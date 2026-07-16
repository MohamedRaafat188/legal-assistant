"""FastAPI application factory.

Thin async wrapper around the Phase 1-4 core (retrieval, agent, citation
guard, Postgres persistence) -- no business logic lives here beyond request
plumbing. Langfuse observability (Phase 6) is wired in as a best-effort,
non-blocking wrapper; the frontend is explicitly out of scope.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from legal_assistant import observability
from legal_assistant.api.routes import auth as auth_routes
from legal_assistant.api.routes import chat as chat_routes
from legal_assistant.api.routes import conversations as conversation_routes
from legal_assistant.api.routes import feedback as feedback_routes
from legal_assistant.config import get_settings
from legal_assistant.db.session import get_engine

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the DB engine on startup; dispose DB + flush Langfuse on shutdown."""
    engine = get_engine()
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    _log.info("legal-assistant API startup: DB reachable")
    yield
    await engine.dispose()
    observability.shutdown()
    _log.info("legal-assistant API shutdown: DB engine disposed, Langfuse flushed")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Legal Assistant API", version="0.6.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_routes.router)
    app.include_router(conversation_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(feedback_routes.router)

    @app.get("/health")
    async def health() -> dict:
        """Cheap liveness/readiness check: confirms the DB is reachable."""
        engine = get_engine()
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return {"status": "ok", "db": "up"}
        except Exception as e:  # noqa: BLE001 -- health check must never raise
            _log.warning("health check DB failure: %s", e)
            return {"status": "degraded", "db": "down"}

    return app


app = create_app()
