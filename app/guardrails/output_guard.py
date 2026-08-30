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
                   "add_to_cart", "view_cart", "prepare_checkout",
                   # Catalogue-shape answers ("we do not stock Gucci, we do have
                   # these brands") are grounded by these too. Omitting them made
                   # a correct refusal look ungrounded.
                   "list_brands", "list_categories"}),
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

#: What to say when a claim was not backed by a tool call. The message is
#: chosen by rule, because a generic "I can't verify that number" is confusing
#: when the customer never asked about a number.
UNGROUNDED_REPLACEMENT: dict[str, str] = {
    "price_claim": (
        "I don't want to quote a price I haven't checked. Let me look that up properly - "
        "which product did you mean?"
    ),
    "stock_claim": (
        "I don't want to tell you something is available without checking stock first. "
        "Which product and size should I look at?"
    ),
    "order_status_claim": (
        "I don't want to tell you where your order is without checking it. Could you confirm "
        "the order number and I'll look it up?"
    ),
    "delivery_date_claim": (
        "I don't want to give you a delivery date I haven't verified. Tell me the order number "
        "and I'll check the live estimate."
    ),
    "policy_claim": (
        "I'd rather quote our actual policy than paraphrase it from memory. Let me look up the "
        "exact wording - what would you like to know?"
    ),
}

#: Shown when the reply contradicted a confirmed-available tool result. Written
#: to recover the conversation rather than merely refuse: the product *is*
#: buyable, so the useful next step is to ask which variant.
#: "I've added it to your bag", "it's in your cart now", "I have put it in",
#: "your cart now contains X". Past/present-tense assertions that a change
#: already happened, or a declarative statement of the cart's current
#: contents - deliberately not matching offers or questions ("shall I add
#: it?", "would you like me to add", "what's in my cart?"), which claim
#: nothing. The "cart/bag now contains" branch was added after a live
#: llama3.2:3b reply - "I'll add it to your bag. Your shopping cart now
#: contains: 1 x ... (Navy)." - slipped past every other branch: "I'll add"
#: is future tense (no claim yet) and "now contains" doesn't use "added" or
#: "is/are ... in", the two verbs the original patterns covered.
_CART_MUTATION_CLAIM = re.compile(
    r"\b(?:I(?:'ve| have)?\s+added"
    r"|(?:has|have)\s+been\s+added"
    r"|added\s+(?:it|that|this|the)\b[^.!?]{0,40}\bto\s+your\s+(?:bag|cart|basket)"
    r"|(?:is|are)\s+now\s+in\s+your\s+(?:bag|cart|basket)"
    r"|(?:cart|bag|basket)\s+(?:now\s+)?(?:contains?|has|includes?|shows?)\b"
    r"|I(?:'ve| have)\s+put\b[^.!?]{0,40}\bin\s+your\s+(?:bag|cart|basket)"
    r"|I(?:'ve| have)?\s+removed)\b",
    re.I,
)

UNBACKED_CART_CLAIM_REPLACEMENT = (
    "I haven't actually added anything yet - I don't want to say your bag changed when it "
    "hasn't. Tell me the size and colour you want and I'll add it properly."
)

AVAILABILITY_CONTRADICTION_REPLACEMENT = (
    "That one is in stock - let me get the details right rather than guess. "
    "Which size and colour would you like?"
)

UNGROUNDED_FALLBACK = (
    "I don't want to tell you something I haven't verified against our systems. "
    "Could you confirm what you'd like me to look up?"
)

#: Typographic characters models reach for that we do not want in output.
#: The system prompt asks for plain ASCII punctuation and the model mostly
#: complies, but "mostly" is not a guarantee and this is a ten-line function.
#: Style rules that can be enforced deterministically should be.
_PUNCTUATION_MAP = str.maketrans({
    "\u2014": "-",   # em dash
    "\u2013": "-",   # en dash
    "\u2012": "-",   # figure dash
    "\u2011": "-",   # non-breaking hyphen
    "\u2010": "-",   # hyphen
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u00a0": " ",   # non-breaking space
    "\u2026": "...",
})


def normalise_punctuation(text: str) -> str:
    """Fold typographic punctuation to plain ASCII."""
    return text.translate(_PUNCTUATION_MAP)


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


#: Internal tool names, discovered from the registry rather than hardcoded so a
#: new tool cannot quietly become un-guarded. Imported lazily inside the
#: function to avoid a circular import (tools -> services -> ... -> guardrails).
def _tool_names() -> frozenset[str]:
    from app.agent.tools import TOOLS_BY_NAME

    return frozenset(TOOLS_BY_NAME)


#: Phrasings that assert a product cannot be bought. Deliberately narrow -
#: only unambiguous unavailability, so "only 3 left" or "out for delivery"
#: never trip it.
_UNAVAILABLE_CLAIM = re.compile(
    r"\b(?:not\s+(?:currently\s+)?(?:available|in\s+stock)"
    r"|no\s+longer\s+(?:available|in\s+stock|stocked)"
    r"|out\s+of\s+stock"
    r"|sold\s+out"
    r"|unavailable"
    r"|we\s+(?:don'?t|do\s+not)\s+(?:have|stock|carry)\s+(?:it|that|this))\b",
    re.I,
)


def contradicts_known_availability(text: str, availability_confirmed: bool) -> bool:
    """True when the reply says "unavailable" about something a tool just
    confirmed is purchasable.

    This is a narrow, deliberately shallow form of span-level grounding. The
    general check elsewhere in this module verifies only that a *class* of
    claim had a *class* of supporting call - it cannot tell whether the
    sentence agrees with what that call returned. That limit is documented in
    ACCURACY_AND_LIMITATIONS.md, and it is exactly the gap a weaker model
    falls through: observed with llama3.2:3b telling a customer a shirt was
    "no longer available" immediately after add_to_cart returned NEEDS_SIZE,
    which is the backend confirming the product exists and asking which
    variant to add.

    Only this one contradiction is checked, because it is the one where being
    wrong is both most likely and most costly: a false "we don't have it"
    ends the conversation and loses the sale, and unlike a wrong price it
    gives the customer no reason to question it. Availability is also the one
    fact the tools return as an unambiguous boolean, so the check is exact
    rather than heuristic.
    """
    return availability_confirmed and bool(_UNAVAILABLE_CLAIM.search(text))


def strip_tool_name_sentences(text: str) -> tuple[str, list[str]]:
    """Remove whole sentences that name an internal tool.

    `recovery_hint` on a ToolError is written for the model - "Call
    list_my_orders to show the orders that do exist" - and a capable model
    paraphrases it into something customer-facing. A smaller one repeats it
    verbatim, tool name and all, which leaks internals and reads as an
    instruction aimed at somebody else.

    Dropping the offending *sentence* rather than blocking the whole reply is
    deliberate: the leak is almost always one trailing suggestion appended to
    an otherwise correct and useful answer, so blocking would throw away good
    information to remove a bad clause. If stripping empties the reply, the
    caller falls back to the generic disclosure message.
    """
    names = _tool_names()
    if not names:
        return text, []

    # Split on sentence boundaries while keeping the delimiter, so rejoining
    # does not mangle spacing or punctuation.
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    found: set[str] = set()
    for part in parts:
        hits = {name for name in names if re.search(rf"\b{re.escape(name)}\b", part)}
        if hits:
            found |= hits
            continue
        kept.append(part)
    if not found:
        return text, []
    return " ".join(kept).strip(), sorted(found)


def screen_output(
    reply: str,
    *,
    tools_called: set[str],
    finish_reason: str = "stop",
    availability_confirmed: bool = False,
    cart_mutated: bool = False,
) -> OutputDecision:
    """Screen an assistant reply before it is shown to the customer.

    `availability_confirmed` is set by the orchestrator when a tool result in
    this turn positively established that a product can be bought. It enables
    the one span-level contradiction check this guard performs; see
    `contradicts_known_availability`.
    """
    reply = normalise_punctuation(reply)
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
    text = normalise_punctuation(redaction.text)
    if redaction.redacted:
        logger.warning("guardrail.output_redacted", extra={"findings": redaction.findings})

    text, leaked_tools = strip_tool_name_sentences(text)
    if leaked_tools:
        logger.warning("guardrail.tool_name_leak_stripped", extra={"tools": leaked_tools})
        if not text.strip():
            return OutputDecision(
                text=DISCLOSURE_REPLACEMENT, allowed=True, grounded=True,
                rule="tool_name_disclosure", action="redact",
                detail=f"stripped internal tool names: {', '.join(leaked_tools)}",
                findings=leaked_tools,
            )

    # A claimed cart change with no successful mutation behind it is among the
    # worst things this assistant can say: the customer believes an item is
    # reserved when nothing happened. Checked against a real success flag
    # rather than "was add_to_cart called", because a *failed* add also puts
    # the tool name in tools_called and would otherwise satisfy the check.
    if _CART_MUTATION_CLAIM.search(text) and not cart_mutated:
        logger.error("guardrail.unbacked_cart_claim", extra={"tools_called": sorted(tools_called)})
        return OutputDecision(
            text=UNBACKED_CART_CLAIM_REPLACEMENT, allowed=True, grounded=False,
            rule="unbacked_cart_claim", action="block",
            detail="reply claimed a cart change with no successful cart mutation in this turn",
        )

    if contradicts_known_availability(text, availability_confirmed):
        logger.error(
            "guardrail.availability_contradiction",
            extra={"tools_called": sorted(tools_called)},
        )
        return OutputDecision(
            text=AVAILABILITY_CONTRADICTION_REPLACEMENT, allowed=True, grounded=False,
            rule="availability_contradiction", action="block",
            detail="reply called a product unavailable that a tool confirmed is in stock",
        )

    for rule, pattern, supporting_tools in GROUNDING_RULES:
        if pattern.search(text) and not (tools_called & supporting_tools):
            logger.error(
                "guardrail.ungrounded_claim",
                extra={"rule": rule, "tools_called": sorted(tools_called)},
            )
            return OutputDecision(
                text=UNGROUNDED_REPLACEMENT.get(rule, UNGROUNDED_FALLBACK),
                allowed=True, grounded=False,
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
