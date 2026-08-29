"""PII detection and redaction.

Applied in two directions and for two different reasons:

* **Inbound, before logging.** A customer may paste a card number into the chat.
  It must not land in a log file, a trace record or the transcript table.
* **Outbound, before display.** Defence in depth. The service layer should never
  emit a card number or an internal identifier, but "should never" is a design
  intention and this is the enforcement.

Detection is regex-based, which means it is precise on structured identifiers
(cards, emails, IBANs) and makes no attempt at unstructured PII such as names.
That limit is stated plainly in docs/ACCURACY_AND_LIMITATIONS.md rather than
papered over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ordering matters: card numbers are tested before phone numbers so a 16-digit
# run is classified as a card rather than a long phone number.
#
# Two patterns are deliberately conservative, because over-redaction is a real
# failure here, not a safe default. A tracking number ("MET4157842762") and an
# order number are exactly the identifiers this assistant exists to discuss, and
# a naive phone pattern eats both:
#
# * `phone` requires separators or an international prefix, so a bare digit run
#   glued to letters is left alone.
# * `card_number` is confirmed with a Luhn checksum before redaction.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Digit run of card length, optionally spaced or hyphenated. Written so the
    # match ends on a digit rather than swallowing a trailing separator.
    ("card_number", re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Requires either an international prefix or at least two separated groups,
    # which is what distinguishes a phone number from a bare reference number.
    ("phone", re.compile(
        r"(?<![\w-])(?:\+\d{1,3}[\s.-]?)?"
        r"(?:\(\d{1,4}\)[\s.-]?|\d{2,4}[\s.-])"
        r"\d{2,4}(?:[\s.-]?\d{2,4}){1,3}(?![\w-])"
    )),
    ("cvv_in_context", re.compile(r"\b(?:cvv|cvc|security\s+code)\D{0,10}(\d{3,4})\b", re.I)),
    ("api_key", re.compile(r"\b(?:sk|gsk|pk|api[_-]?key)[-_][A-Za-z0-9_-]{16,}\b", re.I)),
]

#: Strings that must never appear in a customer-facing reply. These are the
#: internal shapes of the system: table names, ORM classes and SQL verbs.
SCHEMA_TERMS: frozenset[str] = frozenset({
    "product_variants", "order_items", "order_events", "chat_messages",
    "tool_invocations", "guardrail_events", "cart_items", "sqlalchemy",
    "sqlite", "select *", "insert into", "update set", "delete from",
    "drop table", "primary key", "foreign key", "customer_id",
    "price_cents", "list_price_cents", "total_cents", "subtotal_cents",
})

#: Phrases that would disclose the assistant's own construction.
INTERNALS_TERMS: frozenset[str] = frozenset({
    "system prompt", "system message", "my instructions say",
    "you are aurelia", "tool schema", "function schema",
})


def luhn_valid(digits: str) -> bool:
    """Luhn checksum, used to avoid redacting arbitrary long numbers.

    Order numbers, SKUs and tracking numbers are long digit runs too. Without
    the checksum we would redact the very identifiers the assistant exists to
    talk about.
    """
    numbers = [int(d) for d in digits if d.isdigit()]
    if not 13 <= len(numbers) <= 19:
        return False
    checksum = 0
    parity = len(numbers) % 2
    for index, digit in enumerate(numbers):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


@dataclass(slots=True, frozen=True)
class RedactionResult:
    text: str
    findings: list[str]

    @property
    def redacted(self) -> bool:
        return bool(self.findings)


def redact(text: str) -> RedactionResult:
    """Replace detected identifiers with typed placeholders."""
    if not text:
        return RedactionResult(text="", findings=[])

    findings: list[str] = []
    result = text

    for label, pattern in _PATTERNS:
        def _replace(match: re.Match[str], _label: str = label) -> str:
            value = match.group(0)
            if _label == "card_number" and not luhn_valid(value):
                return value  # an order or tracking number, not a card
            if _label == "phone":
                digits = re.sub(r"\D", "", value)
                # Too short to be a phone number, or long enough that it is more
                # likely a reference number we should be showing the customer.
                if not 7 <= len(digits) <= 15:
                    return value
            findings.append(_label)
            return f"[redacted:{_label}]"

        result = pattern.sub(_replace, result)

    return RedactionResult(text=result, findings=sorted(set(findings)))


def contains_schema_leak(text: str) -> list[str]:
    """Return any internal schema terms present in customer-facing text."""
    lowered = text.lower()
    return sorted(term for term in SCHEMA_TERMS if term in lowered)


def contains_internals_leak(text: str) -> list[str]:
    lowered = text.lower()
    return sorted(term for term in INTERNALS_TERMS if term in lowered)
