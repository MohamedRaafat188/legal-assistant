"""docgen page map

Revision ID: bbd34e8ec646
Revises: ba6ef3d81253
Create Date: 2026-09-26 16:43:23.613365

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bbd34e8ec646"
down_revision: str | Sequence[str] | None = "ba6ef3d81253"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "docgen_sessions",
        sa.Column("page_map", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "docgen_uploads",
        sa.Column("ocr_pages", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.drop_column("docgen_uploads", "page_classification")
    op.add_column(
        "docgen_articles",
        sa.Column("position", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "docgen_articles",
        sa.Column("is_mukarrar", sa.Boolean(), server_default="false", nullable=False),
    )
    op.add_column(
        "docgen_articles",
        sa.Column("status", sa.String(length=16), server_default="found", nullable=False),
    )
    op.add_column("docgen_articles", sa.Column("span_first", sa.Integer(), nullable=True))
    op.add_column("docgen_articles", sa.Column("span_last", sa.Integer(), nullable=True))
    op.add_column(
        "docgen_articles",
        sa.Column("possibly_truncated", sa.Boolean(), server_default="false", nullable=False),
    )
    op.drop_index("ix_docgen_articles_session_id_article_number", table_name="docgen_articles")
    op.create_index(
        "ix_docgen_articles_session_id_position",
        "docgen_articles",
        ["session_id", "position"],
        unique=False,
    )
    op.drop_column("docgen_articles", "selected")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "docgen_articles",
        sa.Column("selected", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.drop_index("ix_docgen_articles_session_id_position", table_name="docgen_articles")
    op.create_index(
        "ix_docgen_articles_session_id_article_number",
        "docgen_articles",
        ["session_id", "article_number"],
        unique=False,
    )
    op.drop_column("docgen_articles", "possibly_truncated")
    op.drop_column("docgen_articles", "span_last")
    op.drop_column("docgen_articles", "span_first")
    op.drop_column("docgen_articles", "status")
    op.drop_column("docgen_articles", "is_mukarrar")
    op.drop_column("docgen_articles", "position")
    op.add_column(
        "docgen_uploads",
        sa.Column("page_classification", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.drop_column("docgen_uploads", "ocr_pages")
    op.drop_column("docgen_sessions", "page_map")
