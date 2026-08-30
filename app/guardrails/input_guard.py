"""Inbound guardrails.

Four checks run before a customer message is allowed to reach the agent loop,
cheapest first so an obvious rejection never costs an LLM call:

1. **Rate limit.** Per session, sliding window.
2. **Shape.** Length and emptiness.
3. **Deterministic pattern match.** Known injection phrasings and requests for
   internal state. Fast, free, and auditable.
4. **Prompt Guard 2 classifier.** Meta's jailbreak detector, called through the
   same provider. Catches novel phrasings the pattern list does not know about.

Layers 3 and 4 exist together on purpose. The regex list is precise but only
knows what we thought of; the classifier generalises but is a model, and models
have false negatives. Neither is trusted alone.

The critical point is what these guardrails are *for*. They reduce noise and
give governance a record. They are not the thing preventing data disclosure:
that is the WHERE clause in `app/services/orders.py`. A guardrail that can be
talked around is a filter, and this one is designed as a filter over a hard
boundary, never as the boundary itself.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from app.agent.llm import LLMError, llm_client
from app.config import settings
from app.guardrails.redaction import redact

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 2000

#: Phrasings that attempt to override the assistant's instructions or extract
#: its configuration. Matched case-insensitively against the raw message.
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("instruction_override", re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.?!]{0,40}"
        r"\b(?:previous|prior|above|earlier|all|your|any)\b[^.?!]{0,30}"
        r"\b(?:instruction|prompt|rule|guideline|direction|constraint)s?\b", re.I)),
    ("prompt_extraction", re.compile(
        r"\b(?:show|reveal|print|repeat|output|display|tell me|what (?:is|are))\b[^.?!]{0,40}"
        r"\b(?:system prompt|initial prompt|your instructions|your rules|your prompt|"
        r"tool schema|function definitions?)\b", re.I)),
    ("role_reassignment", re.compile(
        r"\b(?:you are now|from now on you|act as|pretend (?:to be|you are)|roleplay as|"
        r"simulate being)\b[^.?!]{0,50}\b(?:dan|developer mode|admin|root|unrestricted|"
        r"jailbroken|no rules|without restrictions)\b", re.I)),
    ("sql_injection", re.compile(
        r"(?:\bunion\s+select\b|\bdrop\s+table\b|\bdelete\s+from\b|\binsert\s+into\b|"
        r"\bselect\s+\*\s+from\b|--\s*$|\bor\s+1\s*=\s*1\b)", re.I)),
    ("schema_probe", re.compile(
        r"\b(?:database|db|table|schema|column|sql query|orm)\b[^.?!]{0,30}"
        r"\b(?:structure|schema|names?|list|dump|describe|show)\b", re.I)),
    ("privilege_escalation", re.compile(
        r"\b(?:all|other|another|every|someone else'?s?|any)\b\s+"
        r"(?:customers?|users?|people'?s?|accounts?)\b[^.?!]{0,25}"
        r"\b(?:orders?|data|details?|emails?|addresses)\b", re.I)),
]


@dataclass(slots=True)
class GuardDecision:
    """Outcome of the inbound checks."""

    allowed: bool
    rule: str = "clean"
    action: str = "allow"          # allow | block | warn
    score: float = 0.0
    detail: str = ""
    safe_message: str = ""         # redacted, for logging and storage
    user_message: str = ""         # what to show if blocked
    redactions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """Per-key sliding window counter.

    Process-local, like the checkout quote store, and for the same reason: it is
    correct for a single-process POC and explicitly called out in the scaling
    notes as something a multi-worker deployment moves to Redis.
    """

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Return `(allowed, seconds_until_retry)`."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            cutoff = now - self.window
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, int(hits[0] + self.window - now))
            hits.append(now)
            return True, 0

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


rate_limiter = SlidingWindowRateLimiter(
    settings.rate_limit_requests, settings.rate_limit_window_seconds
)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

REFUSAL_MESSAGE = (
    "I can only help with the Aurelia catalogue, your own orders, and purchases on this "
    "account. I can't share how I'm configured or look up anyone else's information. "
    "Is there something I can find or check for you?"
)


def check_patterns(message: str) -> tuple[str, str] | None:
    """Return `(rule, matched_text)` for the first pattern that fires."""
    for rule, pattern in INJECTION_PATTERNS:
        match = pattern.search(message)
        if match:
            return rule, match.group(0)[:160]
    return None


async def classify_injection(message: str) -> float:
    """Prompt Guard 2 jailbreak probability, or -1.0 when unavailable.

    A guardrail that fails closed on a provider hiccup would take the whole
    assistant down; one that fails silently open leaves no record. We return a
    sentinel, let the deterministic layer stand alone, and log the degradation
    so an operator can see that the classifier was not consulted.
    """
    try:
        raw = await llm_client.classify(settings.guard_injection_model, message, timeout=8.0)
        return float(raw.strip())
    except (LLMError, ValueError, TypeError) as exc:
        logger.warning("guardrail.classifier_unavailable", extra={"error": str(exc)})
        return -1.0


async def screen_input(message: str, session_id: str) -> GuardDecision:
    """Run the inbound guardrail chain. Cheapest checks first."""
    redaction = redact(message)
    safe = redaction.text

    allowed, retry_after = rate_limiter.check(session_id)
    if not allowed:
        return GuardDecision(
            allowed=False, rule="rate_limit", action="block", detail=f"retry_after={retry_after}s",
            safe_message=safe, redactions=redaction.findings,
            user_message=(
                f"You're sending messages faster than I can answer them. "
                f"Give me about {retry_after} seconds and try again."
            ),
        )

    stripped = message.strip()
    if not stripped:
        return GuardDecision(
            allowed=False, rule="empty_message", action="block", safe_message=safe,
            user_message="I didn't catch that. What can I help you find?",
        )
    if len(stripped) > MAX_MESSAGE_CHARS:
        return GuardDecision(
            allowed=False, rule="message_too_long", action="block",
            detail=f"length={len(stripped)}", safe_message=safe[:500],
            user_message=(
                f"That message is longer than I can process ({len(stripped):,} characters, "
                f"limit {MAX_MESSAGE_CHARS:,}). Could you shorten it?"
            ),
        )

    pattern_hit = check_patterns(stripped)
    if pattern_hit:
        rule, matched = pattern_hit
        logger.warning(
            "guardrail.injection_pattern",
            extra={"session_id": session_id, "rule": rule, "matched": matched},
        )
        return GuardDecision(
            allowed=False, rule=rule, action="block", score=1.0, detail=matched,
            safe_message=safe, redactions=redaction.findings, user_message=REFUSAL_MESSAGE,
        )

    # An empty guard model name disables the classifier layer outright. That is
    # the correct setting when the configured provider does not host a
    # purpose-built jailbreak classifier - a local Ollama runtime, say, where
    # Prompt Guard is not available. Without this, every single turn would fire
    # a request that is guaranteed to 404, adding latency and log noise to buy
    # nothing. The deterministic pattern layer above still runs either way, and
    # authorisation never depended on this layer in the first place.
    if settings.guard_enabled and llm_client.available and settings.guard_injection_model.strip():
        score = await classify_injection(stripped)
        if score >= settings.guard_injection_threshold:
            logger.warning(
                "guardrail.injection_classifier",
                extra={"session_id": session_id, "score": score},
            )
            return GuardDecision(
                allowed=False, rule="prompt_guard_classifier", action="block", score=score,
                detail=f"jailbreak probability {score:.4f}", safe_message=safe,
                redactions=redaction.findings, user_message=REFUSAL_MESSAGE,
            )
        if score >= 0:
            return GuardDecision(
                allowed=True, rule="clean", action="allow", score=score,
                safe_message=safe, redactions=redaction.findings,
            )

    return GuardDecision(
        allowed=True, rule="clean", action="allow", safe_message=safe,
        redactions=redaction.findings,
    )
