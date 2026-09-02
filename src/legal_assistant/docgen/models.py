"""Persistence for docgen sessions.

Sessions are resumable, so everything the review screen shows survives a
reload: extracted articles with their patch ops and confidence, identity
fields with their provenance, and the page classification that produced
them. Uploaded FILES live on disk under `storage_key`, never in the DB --
they carry national ID and passport numbers and are purged after two days.
"""

from __future__ import annotations

import datetime
import enum

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from legal_assistant.db.models import Base


class SessionStatus(enum.StrEnum):
    draft = "draft"
    ocr_running = "ocr_running"
    ready = "ready"
    rendered = "rendered"
    failed = "failed"
    expired = "expired"


class SourceMode(enum.StrEnum):
    # عقد التأسيس only; CR number/date typed by the lawyer. Cheaper, faster,
    # and more accurate -- valid when the عقد has never been amended.
    aoa_only = "aoa_only"
    # عقد + مستخرج سجل تجاري; CR fields extracted and patched into the articles.
    aoa_plus_cr = "aoa_plus_cr"


class UploadKind(enum.StrEnum):
    aoa = "aoa"
    commercial_register = "commercial_register"


def default_expires_at(days: int) -> datetime.datetime:
    """Expiry timestamp `days` from now, timezone-aware."""
    return datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=days)


class DocgenSession(Base):
    __tablename__ = "docgen_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    company_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=SessionStatus.draft.value
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Storage key of the most recently rendered .docx, or None. Cleared by
    # every PATCH so a stale document can never be downloaded after an edit.
    document_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    expires_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    uploads: Mapped[list[DocgenUpload]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    articles: Mapped[list[DocgenArticle]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="DocgenArticle.article_number",
    )
    fields: Mapped[DocgenFields | None] = relationship(
        back_populates="session", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (
        Index("ix_docgen_sessions_user_id", "user_id"),
        Index("ix_docgen_sessions_expires_at", "expires_at"),
    )


class DocgenUpload(Base):
    __tablename__ = "docgen_uploads"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("docgen_sessions.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # Opaque, non-guessable, never returned to the client.
    storage_key: Mapped[str] = mapped_column(String(128), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # [{page, kind, starts_article, confidence}] from the cheap classify pass.
    page_classification: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[DocgenSession] = relationship(back_populates="uploads")

    __table_args__ = (Index("ix_docgen_uploads_session_id", "session_id"),)


class DocgenArticle(Base):
    __tablename__ = "docgen_articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("docgen_sessions.id", ondelete="CASCADE"), nullable=False
    )
    article_number: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal_words: Mapped[str] = mapped_column(String(64), nullable=False)
    # Verbatim OCR output. Never overwritten -- `patched_text` holds the
    # substituted version and the lawyer's correction, so the original stays
    # auditable for the life of the session.
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    patched_text: Mapped[str] = mapped_column(Text, nullable=False)
    patch_ops: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    new_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    session: Mapped[DocgenSession] = relationship(back_populates="articles")

    __table_args__ = (
        Index("ix_docgen_articles_session_id_article_number", "session_id", "article_number"),
    )


class DocgenFields(Base):
    """Identity fields and the attendance list, one row per session.

    JSONB rather than columns because each field carries provenance --
    `{value, source, confidence, conflict}` -- and the attendee list is
    edited as a unit and never queried individually.
    """

    __tablename__ = "docgen_fields"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("docgen_sessions.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    data: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")

    session: Mapped[DocgenSession] = relationship(back_populates="fields")
