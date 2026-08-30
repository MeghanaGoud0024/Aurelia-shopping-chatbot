"""LLM transport.

Talks to Groq's OpenAI-compatible Chat Completions endpoint over httpx. A vendor
SDK would work, but writing the ~200 lines here buys precise control over the
three things that actually decide whether an agent feels reliable in production:
timeouts, retry policy, and how failures are surfaced to the caller.

Failure philosophy
------------------
An LLM call can fail in ways that are transient (429, 502, a dropped socket) and
ways that are not (401, a malformed request). Retrying the second category
wastes the customer's time and the operator's quota. We classify explicitly and
retry only what is worth retrying, with exponential backoff and jitter, honouring
`Retry-After` when the provider sends it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

#: Providers often state the exact wait in the 429 body ("Please try again in
#: 7.005s"). Honouring that beats guessing with exponential backoff, because
#: a token-per-minute window refills on a schedule we can read rather than
#: probe.
_RETRY_AFTER_IN_BODY = re.compile(r"try again in ([0-9.]+)\s*s", re.I)
#: Upper bound on a single honoured wait. Beyond this the customer is better
#: served by an honest failure than by a request that appears to hang.
MAX_RATE_LIMIT_WAIT_SECONDS = 25.0


class LLMError(RuntimeError):
    """Raised when the provider cannot be reached or refuses the request."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass(slots=True)
class LLMResponse:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    model: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def _parse_arguments(raw: str) -> dict[str, Any]:
    """Parse tool arguments defensively.

    Models occasionally emit arguments that are valid-ish JSON but not an
    object: a bare string, a double-encoded JSON string, or trailing prose. A
    crash here would abort an otherwise recoverable turn, so we salvage what we
    can and let the tool layer's schema validation reject anything genuinely
    unusable.
    """
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return {}
    if isinstance(parsed, str):  # double-encoded
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _provider_wait_hint(response: httpx.Response) -> float | None:
    """Extract how long the provider wants us to wait, in seconds."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = _RETRY_AFTER_IN_BODY.search(response.text or "")
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    reset = response.headers.get("x-ratelimit-reset-tokens")
    if reset:
        parsed = _parse_duration(reset)
        if parsed is not None:
            return parsed
    return None


def _parse_duration(value: str) -> float | None:
    """Parse Groq-style durations such as '7.005s', '1m26.4s', '615ms'."""
    value = value.strip().lower()
    match = re.fullmatch(r"(?:(\d+)m)?([0-9.]+)(ms|s)?", value)
    if not match:
        return None
    minutes, amount, unit = match.groups()
    try:
        seconds = float(amount)
    except ValueError:
        return None
    if unit == "ms":
        seconds /= 1000.0
    if minutes:
        seconds += int(minutes) * 60
    return seconds


#: Pulled out of a 429 error body when the provider states a daily figure the
#: success-path headers never carry, e.g. "... on tokens per day (TPD): Limit
#: 200000, Used 199074, Requested 3363. Please try again in 17m32.78s.".
_DAILY_LIMIT_RE = re.compile(
    r"tokens per day \(TPD\)[:\s]*Limit\s+(?P<limit>\d+),\s*Used\s+(?P<used>\d+)", re.I
)
_TRY_AGAIN_RE = re.compile(r"try again in\s+([0-9hms.]+)", re.I)


@dataclass(slots=True, frozen=True)
class WindowSnapshot:
    """One rate-limit window (e.g. per-minute requests, or tokens) as last
    reported by the provider, plus when we observed it."""

    limit: int
    remaining: int
    reset_in_seconds: float
    observed_at: str  # ISO 8601, UTC


@dataclass(slots=True, frozen=True)
class DailySnapshot:
    """The daily token ceiling, known only when a 429 has stated it - Groq
    does not expose a remaining-today figure on ordinary success responses,
    so this is a fact learned the hard way and cached rather than polled."""

    limit: int
    used: int
    reset_in_seconds: float | None
    observed_at: str


class QuotaTracker:
    """Tracks what is actually knowable about provider quota, and nothing more.

    Two honesty constraints shaped this class:

    1. Groq reports per-minute request/token windows on every response via
       `x-ratelimit-*` headers, so those are genuinely live.
    2. The daily ceiling that actually bit us in testing is *not* in those
       headers on a normal response - it only appears in the text of a 429
       once the ceiling is hit. So the daily figure shown to an operator is
       always "as of the last time we found out", explicitly timestamped,
       never presented as a live number we don't have.

    A locally-summed "tokens used this session" counter fills the gap between
    daily-limit sightings: not authoritative (it doesn't see usage from other
    processes sharing the same key, or from before this process started), but
    directionally useful, and labelled as an estimate rather than a fact.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: WindowSnapshot | None = None
        self._tokens: WindowSnapshot | None = None
        self._daily: DailySnapshot | None = None
        self._session_tokens_used = 0
        self._session_started_at = _now_iso()

    def record_headers(self, headers: httpx.Headers) -> None:
        requests_snap = _parse_window(
            headers, "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
            "x-ratelimit-reset-requests",
        )
        tokens_snap = _parse_window(
            headers, "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-tokens",
        )
        with self._lock:
            if requests_snap is not None:
                self._requests = requests_snap
            if tokens_snap is not None:
                self._tokens = tokens_snap

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self._session_tokens_used += max(0, prompt_tokens) + max(0, completion_tokens)

    def record_error_body(self, status: int, body: str) -> None:
        if status != 429:
            return
        match = _DAILY_LIMIT_RE.search(body)
        if not match:
            return
        reset_in = None
        retry_match = _TRY_AGAIN_RE.search(body)
        if retry_match:
            # The capture can trail into a sentence-ending period ("...32.78s.
            # Need more tokens?"), which _parse_duration's fullmatch then
            # rejects outright - strip it before parsing.
            reset_in = _parse_duration(retry_match.group(1).rstrip("."))
        with self._lock:
            self._daily = DailySnapshot(
                limit=int(match.group("limit")),
                used=int(match.group("used")),
                reset_in_seconds=reset_in,
                observed_at=_now_iso(),
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "per_minute_requests": _window_dict(self._requests),
                "per_minute_tokens": _window_dict(self._tokens),
                "daily": _daily_dict(self._daily),
                "session_tokens_used": self._session_tokens_used,
                "session_started_at": self._session_started_at,
            }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_window(
    headers: httpx.Headers, limit_key: str, remaining_key: str, reset_key: str
) -> WindowSnapshot | None:
    limit_raw = headers.get(limit_key)
    remaining_raw = headers.get(remaining_key)
    reset_raw = headers.get(reset_key)
    if limit_raw is None or remaining_raw is None:
        return None
    reset_seconds = _parse_duration(reset_raw) if reset_raw else 0.0
    try:
        return WindowSnapshot(
            limit=int(limit_raw), remaining=int(remaining_raw),
            reset_in_seconds=reset_seconds or 0.0, observed_at=_now_iso(),
        )
    except ValueError:
        return None


def _window_dict(snap: WindowSnapshot | None) -> dict[str, Any] | None:
    if snap is None:
        return None
    return {
        "limit": snap.limit, "remaining": snap.remaining,
        "reset_in_seconds": round(snap.reset_in_seconds, 1), "observed_at": snap.observed_at,
    }


def _daily_dict(snap: DailySnapshot | None) -> dict[str, Any] | None:
    if snap is None:
        return None
    return {
        "limit": snap.limit, "used": snap.used,
        "remaining": max(0, snap.limit - snap.used),
        "reset_in_seconds": round(snap.reset_in_seconds, 1) if snap.reset_in_seconds is not None else None,
        "observed_at": snap.observed_at,
    }


#: Process-wide, mirroring the llm_client singleton below.
quota_tracker = QuotaTracker()


class LLMClient:
    """Async client for chat completions and single-shot classification."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        if self._client is None:
            if not settings.llm_configured:
                raise LLMError("No LLM API key is configured.", retryable=False)
            self._client = httpx.AsyncClient(
                base_url=settings.llm_base_url.rstrip("/"),
                timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10.0),
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def available(self) -> bool:
        return settings.llm_configured

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not settings.llm_configured:
            raise LLMError("No LLM API key is configured.", retryable=False)
        await self.startup()
        assert self._client is not None

        last_error: LLMError | None = None
        for attempt in range(1, settings.llm_max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = LLMError(f"Could not reach the language model: {exc}", retryable=True)
            else:
                quota_tracker.record_headers(response.headers)
                if response.status_code == 200:
                    return response.json()

                quota_tracker.record_error_body(response.status_code, response.text)
                detail = response.text[:300]
                retryable = response.status_code in RETRYABLE_STATUS
                last_error = LLMError(
                    f"Language model returned {response.status_code}: {detail}",
                    status=response.status_code,
                    retryable=retryable,
                )
                if not retryable:
                    break

                # Honour the provider's own backoff instruction when present:
                # the Retry-After header, the reset hint on the rate-limit
                # headers, or the wait stated in the error body.
                wait = _provider_wait_hint(response)
                if wait is not None and attempt < settings.llm_max_retries:
                    capped = min(wait + 0.4, MAX_RATE_LIMIT_WAIT_SECONDS)
                    logger.warning(
                        "llm.rate_limited",
                        extra={"attempt": attempt, "provider_wait_s": round(wait, 2),
                               "sleeping_s": round(capped, 2)},
                    )
                    await asyncio.sleep(capped)
                    continue

            if attempt < settings.llm_max_retries:
                backoff = min(2 ** (attempt - 1), 8) * (0.7 + random.random() * 0.6)
                logger.warning(
                    "llm.retry",
                    extra={"attempt": attempt, "sleep_seconds": round(backoff, 2),
                           "reason": str(last_error)},
                )
                await asyncio.sleep(backoff)

        raise last_error or LLMError("Language model call failed.", retryable=False)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model or settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format
        effort = reasoning_effort or settings.llm_reasoning_effort
        if effort:
            payload["reasoning_effort"] = effort

        started = time.perf_counter()
        data = await self._post(payload)
        latency_ms = int((time.perf_counter() - started) * 1000)

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        quota_tracker.record_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            raw_args = function.get("arguments") or ""
            calls.append(
                ToolCall(
                    id=raw_call.get("id") or f"call_{len(calls)}",
                    name=function.get("name") or "",
                    arguments=_parse_arguments(raw_args),
                    raw_arguments=raw_args,
                )
            )

        response = LLMResponse(
            content=(message.get("content") or "").strip(),
            reasoning=(message.get("reasoning") or "").strip(),
            tool_calls=calls,
            finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            model=data.get("model") or payload["model"],
        )
        logger.info(
            "llm.completion",
            extra={
                "model": response.model,
                "latency_ms": latency_ms,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "tool_calls": [c.name for c in calls],
                "finish_reason": response.finish_reason,
            },
        )
        return response

    async def classify(self, model: str, text: str, *, timeout: float = 10.0) -> str:
        """Single-shot call to a classifier model such as Prompt Guard.

        Kept separate from `complete` because classifiers return a bare score,
        take no tools, and must fail fast: a guardrail that hangs for 60 seconds
        is worse than a guardrail that is briefly unavailable.

        Deliberately not fed into `quota_tracker`: Groq rate-limits per model,
        and the guard model is a separate, much smaller quota from the main
        chat model. Mixing the two would corrupt the "requests remaining"
        figure the operator actually cares about - the one that blocks replies.
        """
        if not settings.llm_configured:
            raise LLMError("No LLM API key is configured.", retryable=False)
        await self.startup()
        assert self._client is not None
        response = await self._client.post(
            "/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": text}]},
            timeout=timeout,
        )
        if response.status_code != 200:
            raise LLMError(
                f"Classifier {model} returned {response.status_code}",
                status=response.status_code,
                retryable=response.status_code in RETRYABLE_STATUS,
            )
        data = response.json()
        return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""


llm_client = LLMClient()
