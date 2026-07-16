"""POST /auth/register, POST /auth/login -- minimal bcrypt auth + JWT issuance."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
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


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    body: RegisterRequest, session: AsyncSession = Depends(get_db_session)
) -> TokenResponse:
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
