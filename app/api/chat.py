"""Chat endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent.orchestrator import orchestrator
from app.api.deps import Identity, resolve_identity
from app.db.session import get_session
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    identity: Identity = Depends(resolve_identity),
    session: Session = Depends(get_session),
) -> ChatResponse:
    """Run one conversational turn.

    Note that `request.session_id` is accepted but never trusted for identity.
    The customer is resolved from the server-side session cookie, so a client
    cannot select whose orders it reads by editing a request body.
    """
    return await orchestrator.run_turn(
        session=session,
        session_id=identity.session_id,
        customer_id=identity.customer_id,
        customer_name=identity.customer_name,
        message=request.message,
    )


@router.post("/chat/reset")
async def reset_conversation(identity: Identity = Depends(resolve_identity)) -> dict:
    """Clear the in-memory conversation history for this session."""
    await orchestrator.reset(identity.session_id)
    logger.info("chat.reset", extra={"session_id": identity.session_id})
    return {"reset": True, "session_id": identity.session_id}
