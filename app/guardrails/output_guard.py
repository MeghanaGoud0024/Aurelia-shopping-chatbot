"""Outbound guardrails.

Three concerns, in order of severity:

1. **Disclosure.** Internal schema terms, system-prompt content and unredacted
   PII must not reach the browser.
2. **Grounding.** The brief's central requirement is that transactional answers
   come from backend calls, not model guesses. This module enforces that
   mechanically: if a reply makes a transactional claim and no tool ran to
   support it, the reply is replaced.
3. **Truncation.** A reply cut off by the token limit is a broken answer, not a
   short one, and should be labelled as such.

The grounding check is the interesting one, so it is worth being precise about
what it can and cannot do. It verifies that a *class* of claim is backed by a
tool call of the corresponding *class*. It does not verify that the specific
number in the sentence matches the specific number in the tool result; that
would need span-level attribution, which is noted as future work in
docs/ACCURACY_AND_LIMITATIONS.md. What it reliably catches is the failure mode
that actually matters here: the model answering a price, stock or order question
from its own weights when the backend was never consulted.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.guardrails.redaction import (
    contains_internals_leak, contains_schema_leak, redact,
)

logger = logging.getLogger(__name__)

#: Claim classes that require backing tool calls, mapped to the tools that
#: satisfy them. A reply matching the pattern with none of the tools called is
#: ungrounded.
GROUNDING_RULES: list[tuple[str, re.Pattern[str], frozenset[str]]] = [
    (
        "price_claim",
        re.compile(r"(?:costs?|priced?\s+at|for|is|only|just)\s*[$€£]\s?\d", re.I),
        frozenset({"search_products", "get_product_details", "check_availability",
                   "view_cart", "add_to_cart", "update_cart_quantity", "remove_from_cart",
                   "prepare_checkout", "place_order", "get_order_status", "list_my_orders",
                   "track_shipment", "cancel_order", "request_return"}),
    ),
    (
        "stock_claim",
        re.compile(r"\b(?:in stock|out of stock|sold out|we have|available in|"
                   r"\d+\s+(?:left|remaining|in stock))\b", re.I),
        frozenset({"search_products", "check_availability", "get_product_details",
                   "add_to_cart", "view_cart", "prepare_checkout"}),
    ),
    (
        "order_status_claim",
        re.compile(r"\b(?:your order|order\s*#?\s*\d+)\b[^.!?]{0,60}"
                   r"\b(?:is|has been|was|will be)\b[^.!?]{0,40}"
                   r"\b(?:shipped|delivered|packed|confirmed|cancelled|returned|"
                   r"on its way|out for delivery|in transit)\b", re.I),
        frozenset({"get_order_status", "track_shipment", "list_my_orders",
                   "cancel_order", "request_return", "place_order"}),
    ),
    (
        "delivery_date_claim",
        re.compile(r"\b(?:arrive|arriving|delivered|delivery|deliver)\b[^.!?]{0,40}"
                   r"\b(?:on|by|tomorrow|today|within|in\s+\d+\s+days?|"
                   r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
        frozenset({"get_order_status", "track_shipment", "place_order",
                   "list_my_orders", "lookup_policy", "prepare_checkout"}),
    ),
    (
        "policy_claim",
        re.compile(r"\b(?:our (?:policy|policies)|you can return|return window|refund"
                   r"(?:ed|s)? within|warranty (?:period|covers)|free shipping (?:on|over|above))\b", re.I),
        frozenset({"lookup_policy", "get_order_status", "cancel_order", "request_return"}),
    ),
]

DISCLOSURE_REPLACEMENT = (
    "I ran into something I can't share here. Let me get you the right answer another way. "
    "What would you like me to look up?"
)

UNGROUNDED_REPLACEMENT = (
    "I don't want to give you a number I haven't verified. Let me check that against our "
    "systems properly. Could you confirm what you'd like me to look up?"
)

TRUNCATION_SUFFIX = (
    "\n\n(My answer was cut short. Ask me to continue and I'll pick up where I left off.)"
)


@dataclass(slots=True)
class OutputDecision:
    text: str
    allowed: bool = True
    grounded: bool = True
    rule: str = "clean"
    action: str = "allow"          # allow | block | redact | warn
    detail: str = ""
    findings: list[str] = field(default_factory=list)


def screen_output(
    reply: str,
    *,
    tools_called: set[str],
    finish_reason: str = "stop",
) -> OutputDecision:
    """Screen an assistant reply before it is shown to the customer."""
    if not reply.strip():
        return OutputDecision(
            text="I wasn't able to put together an answer for that. Could you rephrase it?",
            allowed=True, grounded=True, rule="empty_reply", action="warn",
        )

    schema_terms = contains_schema_leak(reply)
    if schema_terms:
        logger.error("guardrail.schema_leak_blocked", extra={"terms": schema_terms})
        return OutputDecision(
            text=DISCLOSURE_REPLACEMENT, allowed=False, grounded=True,
            rule="schema_disclosure", action="block",
            detail=f"internal terms in reply: {', '.join(schema_terms)}",
            findings=schema_terms,
        )

    internals = contains_internals_leak(reply)
    if internals:
        logger.error("guardrail.internals_leak_blocked", extra={"terms": internals})
        return OutputDecision(
            text=DISCLOSURE_REPLACEMENT, allowed=False, grounded=True,
            rule="internals_disclosure", action="block",
            detail=f"configuration disclosure: {', '.join(internals)}",
            findings=internals,
        )

    redaction = redact(reply)
    text = redaction.text
    if redaction.redacted:
        logger.warning("guardrail.output_redacted", extra={"findings": redaction.findings})

    for rule, pattern, supporting_tools in GROUNDING_RULES:
        if pattern.search(text) and not (tools_called & supporting_tools):
            logger.error(
                "guardrail.ungrounded_claim",
                extra={"rule": rule, "tools_called": sorted(tools_called)},
            )
            return OutputDecision(
                text=UNGROUNDED_REPLACEMENT, allowed=True, grounded=False,
                rule=rule, action="block",
                detail=(
                    f"reply made a {rule.replace('_', ' ')} but no supporting tool ran "
                    f"(called: {', '.join(sorted(tools_called)) or 'none'})"
                ),
            )

    if finish_reason == "length":
        return OutputDecision(
            text=text + TRUNCATION_SUFFIX, allowed=True, grounded=True,
            rule="truncated", action="warn",
            detail="reply hit the token limit", findings=redaction.findings,
        )

    if redaction.redacted:
        return OutputDecision(
            text=text, allowed=True, grounded=True, rule="pii_redacted",
            action="redact", detail=", ".join(redaction.findings),
            findings=redaction.findings,
        )

    return OutputDecision(text=text, allowed=True, grounded=True)
