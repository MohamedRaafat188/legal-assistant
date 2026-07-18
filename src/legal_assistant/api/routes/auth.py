"""POST /auth/register, POST /auth/login -- minimal bcrypt auth + JWT issuance."""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from legal_assistant.api.deps import get_db_session
from legal_assistant.api.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut
from legal_assistant.auth import (
    InvalidCredentialsError,
    UsernameTakenError,
    authenticate,
    create_access_token,
    register,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# In-process per-IP sliding-window rate limit for open self-registration.
# Pilot-scoped hygiene against scripted account creation -- not a substitute
# for a shared/distributed limiter if the API ever runs multi-instance.
_REGISTER_LIMIT = 5
_REGISTER_WINDOW_SECONDS = 600
_register_attempts: dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_register_rate_limit(ip: str) -> None:
    now = time.monotonic()
    attempts = [t for t in _register_attempts[ip] if now - t < _REGISTER_WINDOW_SECONDS]
    if len(attempts) >= _REGISTER_LIMIT:
        _register_attempts[ip] = attempts
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="too many registration attempts, please try again later",
        )
    attempts.append(now)
    _register_attempts[ip] = attempts


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: Request, body: RegisterRequest, session: AsyncSession = Depends(get_db_session)
) -> TokenResponse:
    _check_register_rate_limit(_client_ip(request))

    try:
        user = await register(session, body.username, body.password)
    except UsernameTakenError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, user=UserOut(id=user.id, username=user.username))


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest, session: AsyncSession = Depends(get_db_session)
) -> TokenResponse:
    try:
        user = await authenticate(session, body.username, body.password)
    except InvalidCredentialsError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e)) from e

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, user=UserOut(id=user.id, username=user.username))
