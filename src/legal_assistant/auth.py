"""Minimal username + hashed-password auth, plus stateless JWT session identity.

Auth stays minimal by design: no refresh tokens, verification, or password
reset (deferred to a later hardening pass). The two non-negotiables are:
passwords are hashed with bcrypt, never stored or logged in plaintext; and
session tokens are signed (HS256) so a request's user identity can't be
forged without `settings.secret_key`.
"""

from __future__ import annotations

import datetime

import bcrypt
import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from legal_assistant.config import Settings, get_settings
from legal_assistant.db.models import User

_JWT_ALGORITHM = "HS256"


class UsernameTakenError(Exception):
    """Raised when registering a username that already exists."""


class InvalidCredentialsError(Exception):
    """Raised when authentication fails (unknown user or wrong password)."""


class InvalidTokenError(Exception):
    """Raised when a session token is missing, malformed, expired, or forged."""


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


async def register(session: AsyncSession, username: str, password: str) -> User:
    """Create a new user with a bcrypt-hashed password. Raises UsernameTakenError on conflict."""
    existing = await session.scalar(select(User).where(User.username == username))
    if existing is not None:
        raise UsernameTakenError(f"username {username!r} is already taken")

    user = User(username=username, password_hash=_hash_password(password))
    session.add(user)
    await session.flush()
    return user


async def authenticate(session: AsyncSession, username: str, password: str) -> User:
    """Verify credentials and return the User. Raises InvalidCredentialsError on failure."""
    user = await session.scalar(select(User).where(User.username == username))
    if user is None or not _verify_password(password, user.password_hash):
        raise InvalidCredentialsError("invalid username or password")
    return user


def create_access_token(user_id: int, settings: Settings | None = None) -> str:
    """Issue a signed JWT carrying the user id and an expiry -- the request's identity proof."""
    settings = settings or get_settings()
    now = datetime.datetime.now(datetime.UTC)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + datetime.timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=_JWT_ALGORITHM)


def decode_access_token(token: str, settings: Settings | None = None) -> int:
    """Validate a session token and return the user id it names. Raises InvalidTokenError."""
    settings = settings or get_settings()
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[_JWT_ALGORITHM])
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError) as e:
        raise InvalidTokenError("invalid or expired session token") from e
