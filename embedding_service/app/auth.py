"""Bearer-token auth dependency for non-health endpoints."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import get_settings


def require_bearer_token(authorization: str = Header(default="")) -> None:
    """FastAPI dependency: raise 401 unless a matching Bearer token is present."""
    settings = get_settings()
    expected = f"Bearer {settings.embedding_service_token}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing bearer token")
