import datetime

from sqlalchemy import inspect

from legal_assistant.db.models import Base
from legal_assistant.docgen.models import (
    DocgenArticle,
    DocgenFields,
    DocgenSession,
    DocgenUpload,
    SessionStatus,
    SourceMode,
    UploadKind,
    default_expires_at,
)


def test_all_four_tables_are_registered_on_the_shared_base():
    names = set(Base.metadata.tables)
    assert {"docgen_sessions", "docgen_uploads", "docgen_articles", "docgen_fields"} <= names


def test_session_columns():
    columns = {c.name for c in inspect(DocgenSession).columns}
    assert columns == {
        "id",
        "user_id",
        "company_type",
        "source_mode",
        "status",
        "error",
        "document_key",
        "created_at",
        "updated_at",
        "expires_at",
        "page_map",
    }


def test_child_rows_cascade_from_the_session():
    for model, fk_column in (
        (DocgenUpload, "session_id"),
        (DocgenArticle, "session_id"),
        (DocgenFields, "session_id"),
    ):
        column = inspect(model).columns[fk_column]
        foreign_key = next(iter(column.foreign_keys))
        assert foreign_key.column.table.name == "docgen_sessions"
        assert foreign_key.ondelete == "CASCADE"


def test_session_cascades_from_users():
    column = inspect(DocgenSession).columns["user_id"]
    assert next(iter(column.foreign_keys)).ondelete == "CASCADE"


def test_jsonb_columns_are_jsonb():
    from sqlalchemy.dialects.postgresql import JSONB

    assert isinstance(inspect(DocgenUpload).columns["ocr_pages"].type, JSONB)
    assert isinstance(inspect(DocgenSession).columns["page_map"].type, JSONB)
    assert isinstance(inspect(DocgenArticle).columns["patch_ops"].type, JSONB)
    assert isinstance(inspect(DocgenFields).columns["data"].type, JSONB)


def test_enums_carry_the_spec_values():
    assert {s.value for s in SessionStatus} == {
        "draft",
        "ocr_running",
        "ready",
        "rendered",
        "failed",
        "expired",
    }
    assert {m.value for m in SourceMode} == {"aoa_only", "aoa_plus_cr"}
    assert {k.value for k in UploadKind} == {"aoa", "commercial_register"}


def test_default_expires_at_is_two_days_ahead_and_tz_aware():
    expires = default_expires_at(2)
    now = datetime.datetime.now(datetime.UTC)
    assert expires.tzinfo is not None
    assert (
        datetime.timedelta(days=1, hours=23) < expires - now < datetime.timedelta(days=2, hours=1)
    )


def test_articles_are_indexed_by_session_and_number():
    index_columns = {
        tuple(c.name for c in index.columns) for index in inspect(DocgenArticle).local_table.indexes
    }
    assert ("session_id", "position") in index_columns


def test_articles_carry_the_page_map_columns():
    cols = inspect(DocgenArticle).columns
    for name in ("position", "is_mukarrar", "status", "span_first", "span_last",
                 "possibly_truncated"):
        assert name in cols
    assert "selected" not in cols


def test_session_articles_are_ordered_by_declared_position():
    assert "position" in str(DocgenSession.articles.property.order_by[0])
