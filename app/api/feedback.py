"""Feedback capture.

The brief asks for a concrete feedback-handling approach, so feedback is stored
against the `turn_id` it refers to. Because every turn already has a full tool
invocation trail keyed by that same id, a "not helpful" rating is not an opaque
complaint: it can be joined back to the exact tool calls, arguments and results
that produced the reply. That join is what makes the signal usable for
prompt and retrieval improvement rather than merely countable.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import Identity, resolve_identity
from app.db.models import Feedback
from app.db.session import get_session
from app.observability.trace import record_feedback
from app.schemas import FeedbackRequest

router = APIRouter(tags=["feedback"])


@router.post("/feedback")
def submit_feedback(
    payload: FeedbackRequest,
    identity: Identity = Depends(resolve_identity),
    session: Session = Depends(get_session),
) -> dict:
    entry = record_feedback(
        session,
        session_id=identity.session_id,
        turn_id=payload.turn_id,
        rating=payload.rating,
        reason=payload.reason,
        comment=payload.comment,
    )
    return {"recorded": True, "feedback_id": entry.id, "turn_id": payload.turn_id}


@router.get("/feedback/summary")
def feedback_summary(session: Session = Depends(get_session)) -> dict:
    rows = session.execute(
        select(Feedback.rating, func.count()).group_by(Feedback.rating)
    ).all()
    counts = {rating: count for rating, count in rows}
    helpful = counts.get("helpful", 0)
    total = sum(counts.values())
    reasons = session.execute(
        select(Feedback.reason, func.count())
        .where(Feedback.rating == "not_helpful", Feedback.reason != "")
        .group_by(Feedback.reason).order_by(func.count().desc()).limit(10)
    ).all()
    return {
        "total": total,
        "helpful": helpful,
        "not_helpful": counts.get("not_helpful", 0),
        "helpful_rate": round(helpful / total, 3) if total else None,
        "top_not_helpful_reasons": [{"reason": r, "count": c} for r, c in reasons],
    }
