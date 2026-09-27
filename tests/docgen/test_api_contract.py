import pytest
from fastapi import FastAPI
from fastapi.routing import iter_route_contexts

from legal_assistant.api.routes import docgen as docgen_routes
from legal_assistant.api.schemas import (
    DocgenArticleOut,
    DocgenSessionDetailOut,
    DocgenSessionOut,
    DocgenUploadOut,
)


@pytest.fixture
def routes():
    app = FastAPI()
    app.include_router(docgen_routes.router)
    # FastAPI 0.139 wraps `include_router`-ed routes in a lazy `_IncludedRouter`
    # placeholder on `app.routes`, so a plain `getattr(r, "methods", [])` scan
    # (as older FastAPI versions supported) silently sees nothing for them.
    # `iter_route_contexts` is the public way to get the fully-resolved,
    # flattened route list this test actually needs.
    return {
        (ctx.path, method)
        for ctx in iter_route_contexts(app.routes)
        for method in ctx.methods or []
    }


def test_every_documented_route_exists(routes):
    expected = {
        ("/docgen/sessions", "POST"),
        ("/docgen/sessions/{session_id}/uploads", "POST"),
        ("/docgen/sessions/{session_id}", "GET"),
        ("/docgen/sessions/{session_id}/fields", "PATCH"),
        ("/docgen/sessions/{session_id}/articles/{position}", "PATCH"),
        ("/docgen/sessions/{session_id}/attendees", "PATCH"),
        ("/docgen/sessions/{session_id}/render", "POST"),
        ("/docgen/sessions/{session_id}/document", "GET"),
        ("/docgen/sessions/{session_id}", "DELETE"),
        ("/docgen/sessions/{session_id}/page-map", "PUT"),
        ("/docgen/sessions/{session_id}/uploads/{upload_id}/pages/{page}", "GET"),
    }
    assert expected <= routes


def test_no_response_schema_leaks_a_storage_key():
    for model in (DocgenSessionOut, DocgenSessionDetailOut, DocgenUploadOut, DocgenArticleOut):
        names = set(model.model_fields)
        assert "storage_key" not in names
        assert "document_key" not in names


def test_session_out_reports_document_availability_as_a_boolean():
    assert DocgenSessionOut.model_fields["has_document"].annotation is bool


def test_every_route_requires_authentication():
    for route in docgen_routes.router.routes:
        source = route.endpoint.__code__.co_varnames
        assert "user_id" in source, f"{route.path} does not resolve a user"


def test_the_router_is_mounted_on_the_app():
    from legal_assistant.api.app import create_app

    paths = {ctx.path for ctx in iter_route_contexts(create_app().routes)}
    assert "/docgen/sessions" in paths


def test_article_out_exposes_position_and_flags_not_selected():
    names = set(DocgenArticleOut.model_fields)
    assert {"position", "is_mukarrar", "status", "span_first", "span_last",
            "possibly_truncated"} <= names
    assert "selected" not in names


def test_upload_out_no_longer_reports_page_classification():
    assert "page_classification" not in DocgenUploadOut.model_fields


def test_page_map_request_accepts_the_spec_shape():
    from legal_assistant.api.schemas import DocgenPageMapRequest

    body = DocgenPageMapRequest.model_validate(
        {"entries": {"company_name": {"from": 1, "to": 1, "article": "2"}},
         "amended": [{"from": 3, "to": 4, "article": "٦ مكرر"}]}
    )
    assert body.model_dump(by_alias=True)["amended"][0] == {
        "from": 3, "to": 4, "article": "٦ مكرر"
    }
