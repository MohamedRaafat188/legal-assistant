"""Pydantic request/response models for the API."""

from __future__ import annotations

import datetime
import re

from pydantic import BaseModel, Field, field_validator

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=256)

    @field_validator("username")
    @classmethod
    def _validate_username(cls, value: str) -> str:
        if not _USERNAME_RE.match(value):
            raise ValueError(
                "username must be 3-32 characters: letters, digits, '_', '.', or '-' only"
            )
        return value


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="محادثة جديدة", max_length=255)


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime.datetime
    updated_at: datetime.datetime


class CitationOut(BaseModel):
    law_name: str
    article_number: int | None
    citation_label: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: list[CitationOut] | None
    created_at: datetime.datetime


class ConversationDetailOut(BaseModel):
    id: int
    title: str
    summary: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    messages: list[MessageOut]


class ChatRequest(BaseModel):
    conversation_id: int
    message: str = Field(min_length=1)


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: int = Field(ge=0, le=1, description="1 for thumbs-up, 0 for thumbs-down")
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    status: str = "recorded"
