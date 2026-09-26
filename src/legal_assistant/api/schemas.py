"""Pydantic request/response models for the API."""

from __future__ import annotations

import datetime
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    # The bis designation, normalized ("مكرر", "مكرر 1", "مكرر ا"), else None
    # -- machine-facing and stable, since it is what identifies which article
    # in a series was cited; citation_label carries the form meant for reading.
    # Defaulted because citations persisted before law 72/2017 was ingested
    # carry no such key.
    article_suffix: str | None = None
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


# --- docgen -------------------------------------------------------------
# Note: no schema here exposes storage_key or document_key. They are
# unguessable by design and must stay server-side.


class DocgenSessionCreateRequest(BaseModel):
    company_type: str
    source_mode: str


class DocgenSessionOut(BaseModel):
    id: int
    company_type: str
    source_mode: str
    status: str
    error: str | None = None
    has_document: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime
    expires_at: datetime.datetime
    page_map: dict | None = None


class DocgenUploadOut(BaseModel):
    id: int
    kind: str
    filename: str
    page_count: int | None


class DocgenArticleOut(BaseModel):
    position: int
    article_number: int
    is_mukarrar: bool
    ordinal_words: str
    article_name: str
    status: str
    span_first: int | None
    span_last: int | None
    possibly_truncated: bool
    source_text: str
    patched_text: str
    patch_ops: list | None
    confidence: float | None
    needs_review: bool
    new_text: str | None


class DocgenSessionDetailOut(DocgenSessionOut):
    uploads: list[DocgenUploadOut]
    articles: list[DocgenArticleOut]
    fields: dict
    warning: str | None = None


class DocgenFieldsPatchRequest(BaseModel):
    fields: dict[str, str]


class DocgenArticlePatchRequest(BaseModel):
    patched_text: str | None = None
    new_text: str | None = None


class DocgenPageMapEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    first: int = Field(alias="from")
    to: int
    # A number ("6", "٦ مكرر") or "التمهيد". Validated by docgen.pages, not here,
    # so every problem comes back together in one Arabic 422.
    article: str


class DocgenPageMapRequest(BaseModel):
    entries: dict[str, DocgenPageMapEntry]
    amended: list[DocgenPageMapEntry]


class DocgenAttendee(BaseModel):
    name: str
    shares: str | None = None
    percentage: str | None = None
    attending: bool = True


class DocgenAttendeesPatchRequest(BaseModel):
    attendees: list[DocgenAttendee]
