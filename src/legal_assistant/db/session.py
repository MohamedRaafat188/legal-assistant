"""Async SQLAlchemy engine/session factory for the Postgres store.

Single place the engine is constructed, so the CLI, Alembic, and (later)
FastAPI all share the same async setup.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from legal_assistant.config import Settings, get_settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use."""
    global _engine
    if _engine is None:
        settings = settings or get_settings()
        _engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return _engine


def get_sessionmaker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory, creating it on first use."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(settings), expire_on_commit=False)
    return _sessionmaker


@asynccontextmanager
async def session_scope(settings: Settings | None = None) -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession, committing on success and rolling back on error."""
    session = get_sessionmaker(settings)()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
