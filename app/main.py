"""Application entry point."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import admin, catalog, chat, dashboard, feedback, orders
from app.agent.llm import llm_client
from app.config import WEB_DIR, settings
from app.db.seed import seed_database
from app.db.session import init_db, session_scope
from app.logging_setup import configure_logging, correlation_id
from app.retrieval.index import retrieval_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Bring the application to a state where it can answer correctly.

    Ordering matters: schema, then data, then the retrieval index, which is
    built *from* that data. Seeding is idempotent, so a restart against an
    existing database is a no-op rather than a duplicate load. Building the
    index at startup rather than on first request means the first customer does
    not pay for it.
    """
    configure_logging(settings.log_level)
    logger.info(
        "app.starting",
        extra={
            "environment": settings.environment,
            "llm_configured": settings.llm_configured,
            "model": settings.llm_model,
        },
    )

    init_db()
    with session_scope() as session:
        counts = seed_database(session)
        retrieval_service.build(session)

    if settings.llm_configured:
        await llm_client.startup()
    else:
        logger.warning(
            "app.no_llm_key",
            extra={"detail": "Running with the rule-based planner. Set AURELIA_LLM_API_KEY for full quality."},
        )

    logger.info("app.ready", extra={**counts, **retrieval_service.stats()})
    yield

    await llm_client.shutdown()
    logger.info("app.stopped")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Conversational shopping assistant. Every transactional answer is produced by an "
        "explicit backend tool call against the database, never by the language model."
    ),
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


@app.middleware("http")
async def correlation_and_timing(request: Request, call_next):
    """Attach a correlation id to every request and log its outcome.

    The id is returned in `X-Correlation-Id` and stamped on every log line and
    audit record produced while handling the request, so a customer report of
    "the assistant said something odd at 14:32" is traceable to the exact tool
    calls behind it.
    """
    request_id = request.headers.get("x-correlation-id") or uuid.uuid4().hex[:16]
    token = correlation_id.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request.failed", extra={"path": request.url.path})
        correlation_id.reset(token)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Something went wrong on our side.",
                "correlation_id": request_id,
            },
            headers={"X-Correlation-Id": request_id},
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Correlation-Id"] = request_id
    if not request.url.path.startswith("/static"):
        logger.info(
            "request.completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
    correlation_id.reset(token)
    return response


api_prefix = "/api"
app.include_router(chat.router, prefix=api_prefix)
app.include_router(catalog.router, prefix=api_prefix)
app.include_router(orders.router, prefix=api_prefix)
app.include_router(dashboard.router, prefix=api_prefix)
app.include_router(feedback.router, prefix=api_prefix)
app.include_router(admin.router, prefix=api_prefix)


@app.get("/api/session")
def session_info(request: Request) -> dict:
    """Who the interface is acting as, for the header chip."""
    from app.api.deps import resolve_identity
    from fastapi import Response as FastAPIResponse

    from app.db.session import SessionLocal

    response = FastAPIResponse()
    db = SessionLocal()
    try:
        identity = resolve_identity(
            response, db, request.cookies.get("aurelia_session")
        )
        payload = {
            "session_id": identity.session_id,
            "customer_name": identity.customer_name,
            "customer_public_id": identity.customer_public_id,
            "loyalty_tier": identity.loyalty_tier,
            "llm_configured": settings.llm_configured,
            "model": settings.llm_model if settings.llm_configured else "rule-based planner",
        }
    finally:
        db.close()

    json_response = JSONResponse(payload)
    for key, value in response.headers.items():
        if key.lower() == "set-cookie":
            json_response.headers.append(key, value)
    return json_response


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
