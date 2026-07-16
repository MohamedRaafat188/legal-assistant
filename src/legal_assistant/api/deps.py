"""Shared FastAPI dependencies: per-request DB session, current-user resolution."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from legal_assistant.auth import InvalidTokenError, decode_access_token
from legal_assistant.db.session import get_sessionmaker

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield one AsyncSession per request, committing on success, rolling back on error."""
    session = get_sessionmaker()()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> int:
    """Resolve the requesting user id from the `Authorization: Bearer <token>` header.

    401 on a missing, malformed, expired, or forged token -- this dependency
    is what answers "which user is this request" for every protected route.
    """
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    try:
        return decode_access_token(credentials.credentials)
    except InvalidTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e)) from e
