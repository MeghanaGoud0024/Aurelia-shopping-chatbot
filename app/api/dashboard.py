"""Dashboard endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import Identity, resolve_identity
from app.db.session import get_session
from app.services import dashboard as dashboard_service

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard")
def dashboard(
    identity: Identity = Depends(resolve_identity),
    session: Session = Depends(get_session),
) -> dict:
    """Everything the home view renders, scoped to the signed-in customer."""
    return dashboard_service.build_dashboard(
        session, customer_id=identity.customer_id, session_id=identity.session_id
    )
