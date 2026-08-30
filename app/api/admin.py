"""Operations and governance endpoints.

These back the "Governance" tab in the interface and, more importantly, they are
the answer to "how would you audit this in production". Everything here reads
from the same audit tables the orchestrator writes to during a turn, so what an
operator sees is the record itself and not a parallel reconstruction of it.

In a real deployment these routes sit behind an operator role. That is called
out in `docs/SCALING.md` rather than faked with a hardcoded password here.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.llm import llm_client, quota_tracker
from app.agent.mode import assistant_mode
from app.config import settings
from app.db.models import GuardrailEvent, ToolInvocation
from app.db.session import get_session
from app.retrieval.index import retrieval_service

router = APIRouter(prefix="/ops", tags=["operations"])


@router.get("/health")
def health(session: Session = Depends(get_session)) -> dict:
    """Liveness and readiness in one payload.

    `ready` is false when the retrieval index has not been built or the database
    is empty, because in either case the application is running but cannot
    answer correctly. Reporting "up" in that state is how silent outages happen.
    """
    from app.db.models import Product

    product_count = session.scalar(select(func.count(Product.id))) or 0
    return {
        "status": "ok",
        "ready": retrieval_service.ready and product_count > 0,
        "environment": settings.environment,
        "llm_configured": settings.llm_configured,
        "model": settings.llm_model if settings.llm_configured else "rule-based planner",
        "forced_fallback": assistant_mode.forced_fallback,
        "guardrails_enabled": settings.guard_enabled,
        "catalogue_size": product_count,
        "retrieval": retrieval_service.stats(),
    }


@router.get("/llm-status")
def llm_status() -> dict:
    """Everything actually knowable about provider quota, live.

    This never makes a network call - it reads state accumulated from the
    headers and error bodies of calls the assistant was already making, so
    polling it costs nothing and burns no quota of its own. See
    `app.agent.llm.QuotaTracker` for exactly what is and is not a live figure:
    the per-minute windows are; the daily ceiling is only as fresh as the last
    time a 429 stated it, and is timestamped as such rather than implied to be
    current.
    """
    mode = assistant_mode.snapshot()
    effective_mode = "fallback" if (mode["forced_fallback"] or not llm_client.available) else "live"
    return {
        "configured": llm_client.available,
        "model": settings.llm_model,
        "forced_fallback": mode["forced_fallback"],
        "forced_fallback_changed_at": mode["changed_at"],
        "effective_mode": effective_mode,
        "quota": quota_tracker.snapshot(),
    }


@router.post("/llm-mode")
def set_llm_mode(payload: dict) -> dict:
    """Toggle the manual fallback override.

    Setting forced_fallback=true routes every subsequent turn through the
    deterministic planner regardless of whether the API key works, until
    toggled back. Same tools, same authorisation, same audit trail - only the
    source of the reply's wording changes. See app/agent/fallback.py.
    """
    if "forced_fallback" not in payload or not isinstance(payload["forced_fallback"], bool):
        raise HTTPException(status_code=400, detail="forced_fallback (boolean) is required")
    assistant_mode.set_forced_fallback(payload["forced_fallback"])
    return llm_status()


@router.get("/metrics")
def metrics(session: Session = Depends(get_session)) -> dict:
    """Aggregate operational metrics over the recorded audit trail."""
    total_calls = session.scalar(select(func.count(ToolInvocation.id))) or 0
    by_tool = session.execute(
        select(
            ToolInvocation.tool_name,
            func.count(),
            func.round(func.avg(ToolInvocation.latency_ms), 1),
            func.sum(func.iif(ToolInvocation.status == "error", 1, 0)),
        )
        .group_by(ToolInvocation.tool_name)
        .order_by(func.count().desc())
    ).all()

    guardrails = session.execute(
        select(GuardrailEvent.stage, GuardrailEvent.rule, GuardrailEvent.action, func.count())
        .group_by(GuardrailEvent.stage, GuardrailEvent.rule, GuardrailEvent.action)
        .order_by(func.count().desc())
        .limit(20)
    ).all()

    blocked = session.scalar(
        select(func.count(GuardrailEvent.id)).where(GuardrailEvent.action == "block")
    ) or 0
    total_guard = session.scalar(select(func.count(GuardrailEvent.id))) or 0

    return {
        "tool_calls_total": total_calls,
        "tools": [
            {
                "tool": name, "calls": calls,
                "avg_latency_ms": avg or 0, "errors": errors or 0,
                "error_rate": round((errors or 0) / calls, 3) if calls else 0,
            }
            for name, calls, avg, errors in by_tool
        ],
        "guardrail_events_total": total_guard,
        "guardrail_blocks": blocked,
        "guardrail_block_rate": round(blocked / total_guard, 4) if total_guard else 0,
        "guardrails": [
            {"stage": stage, "rule": rule, "action": action, "count": count}
            for stage, rule, action, count in guardrails
        ],
    }


@router.get("/audit/{turn_id}")
def audit_turn(turn_id: str, session: Session = Depends(get_session)) -> dict:
    """Full evidence trail for one conversational turn.

    This is the reviewer's answer to "where did that sentence come from": the
    exact arguments sent to each backend call and the exact rows that came back.
    """
    invocations = session.scalars(
        select(ToolInvocation).where(ToolInvocation.turn_id == turn_id)
        .order_by(ToolInvocation.id)
    ).all()
    events = session.scalars(
        select(GuardrailEvent).where(GuardrailEvent.turn_id == turn_id)
        .order_by(GuardrailEvent.id)
    ).all()

    def _load(raw: str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    return {
        "turn_id": turn_id,
        "tool_invocations": [
            {
                "tool": inv.tool_name,
                "arguments": _load(inv.arguments_json),
                "result": _load(inv.result_json),
                "status": inv.status,
                "error": inv.error_message or None,
                "latency_ms": inv.latency_ms,
                "at": inv.created_at.isoformat(),
            }
            for inv in invocations
        ],
        "guardrail_events": [
            {
                "stage": e.stage, "rule": e.rule, "action": e.action,
                "score": e.score, "detail": e.detail, "at": e.created_at.isoformat(),
            }
            for e in events
        ],
    }


@router.get("/tools")
def list_tools() -> dict:
    """The tool contract as the model sees it, for review and documentation."""
    from app.agent.tools import TOOLS

    return {
        "count": len(TOOLS),
        "tools": [
            {
                "name": t.name,
                "mutating": t.mutating,
                "renders": t.renders,
                "description": t.description,
                "parameters": sorted((t.parameters.get("properties") or {}).keys()),
                "required": t.parameters.get("required", []),
            }
            for t in TOOLS
        ],
    }


@router.get("/guardrail-events")
def guardrail_events(
    limit: int = Query(50, ge=1, le=200),
    action: str | None = None,
    session: Session = Depends(get_session),
) -> dict:
    statement = select(GuardrailEvent).order_by(GuardrailEvent.id.desc()).limit(limit)
    if action:
        statement = statement.where(GuardrailEvent.action == action)
    rows = session.scalars(statement).all()
    return {
        "events": [
            {
                "id": e.id, "stage": e.stage, "rule": e.rule, "action": e.action,
                "score": e.score, "detail": e.detail, "at": e.created_at.isoformat(),
            }
            for e in rows
        ]
    }
