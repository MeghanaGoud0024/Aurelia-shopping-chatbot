"""Audit trail persistence.

Two things are recorded for every turn, and they answer different questions:

* `ToolInvocation` answers "what evidence supports this sentence?" It holds the
  exact arguments, the exact result and the latency of every backend call.
* `GuardrailEvent` answers "why did the assistant behave that way?" It records
  every guardrail decision, including the ones that allowed the message through,
  because a governance review needs to see the denominator and not just the
  blocks.

Both are written on the same session as the business data, so an audit record
and the transaction it describes commit or roll back together. An audit trail
that can disagree with the ledger is worse than none.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import ChatMessage, Feedback, GuardrailEvent, ToolInvocation

logger = logging.getLogger(__name__)

#: Cap on stored payload size. A full search result is a few kilobytes; storing
#: unbounded blobs would make the audit table the largest thing in the database
#: without adding evidentiary value.
MAX_STORED_PAYLOAD_CHARS = 8_000


def _serialise(payload: Any) -> str:
    try:
        text = json.dumps(payload, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(payload)
    if len(text) > MAX_STORED_PAYLOAD_CHARS:
        return text[:MAX_STORED_PAYLOAD_CHARS] + f'... [truncated, {len(text)} chars total]'
    return text


def record_tool_invocation(
    session: Session,
    *,
    session_id: str,
    turn_id: str,
    correlation_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    status: str,
    latency_ms: int,
    error_message: str = "",
) -> None:
    session.add(
        ToolInvocation(
            session_id=session_id,
            turn_id=turn_id,
            correlation_id=correlation_id,
            tool_name=tool_name,
            arguments_json=_serialise(arguments),
            result_json=_serialise(result),
            status=status,
            error_message=error_message[:400],
            latency_ms=latency_ms,
        )
    )


def record_guardrail_event(
    session: Session,
    *,
    session_id: str,
    turn_id: str,
    stage: str,
    rule: str,
    action: str,
    score: float = 0.0,
    detail: str = "",
) -> None:
    session.add(
        GuardrailEvent(
            session_id=session_id,
            turn_id=turn_id,
            stage=stage,
            rule=rule,
            action=action,
            score=score,
            detail=detail[:400],
        )
    )


def record_message(
    session: Session, *, session_id: str, turn_id: str, role: str, content: str
) -> None:
    """Persist one conversation message.

    Content is expected to be already redacted by the inbound guardrail. This
    function does not redact, so that there is exactly one place responsible for
    it and no ambiguity about whether it ran.
    """
    session.add(
        ChatMessage(
            session_id=session_id, turn_id=turn_id, role=role,
            content=content[:MAX_STORED_PAYLOAD_CHARS],
        )
    )


def record_feedback(
    session: Session,
    *,
    session_id: str,
    turn_id: str,
    rating: str,
    reason: str = "",
    comment: str = "",
) -> Feedback:
    entry = Feedback(
        session_id=session_id, turn_id=turn_id, rating=rating,
        reason=reason[:60], comment=comment[:2000],
    )
    session.add(entry)
    session.flush()
    logger.info(
        "feedback.recorded",
        extra={"session_id": session_id, "turn_id": turn_id, "rating": rating, "reason": reason},
    )
    return entry
