"""Regression tests: no synthetic personal-data string may survive into any
part of an `OcrError` raised by `GeminiOcrProvider`, or into the repr of
`PageText`/`PageClassification`.

Table-driven over every failure path so this class of leak (raw model
output, or an underlying exception's message, folded into an exception seen
by a future caller/log formatter) cannot silently return in Tasks 14-15.
Everything here runs offline: `GeminiOcrProvider._llm` is replaced with a
fake object before any `.ainvoke()` happens, so no network call is made.
"""

from __future__ import annotations

import asyncio
import traceback

import pytest

from legal_assistant.docgen.ocr.base import OcrError, PageClassification, PageKind, PageText
from legal_assistant.docgen.ocr.gemini import GeminiOcrProvider
from legal_assistant.docgen.pdf import PageImage

# A synthetic (not real) Egyptian national ID and some Arabic prose, standing
# in for text a scanned عقد تأسيس page or سجل تجارى extraction could contain.
SENSITIVE_ID = "29801011234567"
SENSITIVE_TEXT = f"الرقم القومي {SENSITIVE_ID} محمد أحمد علي"

_IMAGES = [PageImage(page=1, png=b"\x89PNG-fake")]


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLlmReturns:
    """Fake LLM whose `.ainvoke` returns a fixed (non-JSON) response."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, _messages: list[dict]) -> _Response:
        return _Response(self._content)


class _FakeLlmRaises:
    """Fake LLM whose `.ainvoke` raises, mimicking an SDK/HTTP error whose
    message echoes request or response content."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def ainvoke(self, _messages: list[dict]) -> _Response:
        raise self._exc


def _provider(fake_llm) -> GeminiOcrProvider:
    provider = GeminiOcrProvider()
    provider._llm = fake_llm  # bypass the real ChatGoogleGenerativeAI entirely
    return provider


def _assert_sensitive_absent(exc: Exception, sensitive: str) -> None:
    """The core assertion: `sensitive` must appear in NONE of: str(exc),
    repr(exc), str(exc.__cause__), repr(exc.__cause__), or the fully
    formatted traceback (which is what a log handler like Sentry's actually
    captures)."""
    assert sensitive not in str(exc)
    assert sensitive not in repr(exc)
    if exc.__cause__ is not None:
        assert sensitive not in str(exc.__cause__)
        assert sensitive not in repr(exc.__cause__)
    formatted = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    assert sensitive not in formatted


# --- Table of failure paths -------------------------------------------------
#
# Each entry is (label, coroutine-factory). The factory takes no args and
# returns the awaitable to run; it must raise OcrError.

def _non_json_classify():
    provider = _provider(_FakeLlmReturns(f"not json at all {SENSITIVE_TEXT}"))
    return provider.classify_pages(_IMAGES)


def _non_json_extract():
    provider = _provider(_FakeLlmReturns(f"not json at all {SENSITIVE_TEXT}"))
    return provider.extract(_IMAGES)


def _non_json_extract_fields():
    provider = _provider(_FakeLlmReturns(f"not json at all {SENSITIVE_TEXT}"))
    return provider.extract_fields(_IMAGES, schema={"type": "object"})


def _ask_raises_classify():
    provider = _provider(
        _FakeLlmRaises(RuntimeError(f"upstream failure, request body was: {SENSITIVE_TEXT}"))
    )
    return provider.classify_pages(_IMAGES)


def _ask_raises_extract():
    provider = _provider(
        _FakeLlmRaises(RuntimeError(f"upstream failure, request body was: {SENSITIVE_TEXT}"))
    )
    return provider.extract(_IMAGES)


def _ask_raises_extract_fields():
    provider = _provider(
        _FakeLlmRaises(RuntimeError(f"upstream failure, request body was: {SENSITIVE_TEXT}"))
    )
    return provider.extract_fields(_IMAGES, schema={"type": "object"})


def _wrong_shape_classify():
    # Valid JSON, but not the expected list shape.
    provider = _provider(_FakeLlmReturns('{"not": "a list"}'))
    return provider.classify_pages(_IMAGES)


def _wrong_shape_extract():
    provider = _provider(_FakeLlmReturns("[1, 2, 3]"))
    return provider.extract(_IMAGES)


def _wrong_shape_extract_fields():
    provider = _provider(_FakeLlmReturns("[1, 2, 3]"))
    return provider.extract_fields(_IMAGES, schema={"type": "object"})


FAILURE_PATHS = {
    "non_json / classify_pages (_parse_json raise site)": _non_json_classify,
    "non_json / extract (_parse_json raise site)": _non_json_extract,
    "non_json / extract_fields (_parse_json raise site)": _non_json_extract_fields,
    "llm_raises / classify_pages (_ask catch-all raise site)": _ask_raises_classify,
    "llm_raises / extract (_ask catch-all raise site)": _ask_raises_extract,
    "llm_raises / extract_fields (_ask catch-all raise site)": _ask_raises_extract_fields,
    "wrong_shape / classify_pages": _wrong_shape_classify,
    "wrong_shape / extract": _wrong_shape_extract,
    "wrong_shape / extract_fields": _wrong_shape_extract_fields,
}


@pytest.mark.parametrize("label", list(FAILURE_PATHS))
def test_no_sensitive_data_in_any_form_of_the_raised_error(label, caplog):
    factory = FAILURE_PATHS[label]
    with caplog.at_level("DEBUG"):
        with pytest.raises(OcrError) as excinfo:
            asyncio.run(factory())
    exc = excinfo.value
    _assert_sensitive_absent(exc, SENSITIVE_ID)
    _assert_sensitive_absent(exc, SENSITIVE_TEXT)
    # Nothing sensitive was logged either (finding 5: no content-bearing log).
    assert SENSITIVE_ID not in caplog.text
    assert SENSITIVE_TEXT not in caplog.text


def test_non_json_error_message_still_names_the_call_site_and_length():
    """The sanitized message must stay debuggable: which call, how long."""
    provider = _provider(_FakeLlmReturns(f"not json at all {SENSITIVE_TEXT}"))
    with pytest.raises(OcrError) as excinfo:
        asyncio.run(provider.classify_pages(_IMAGES))
    message = str(excinfo.value)
    assert "classify_pages" in message
    assert "length=" in message


def test_ask_failure_message_still_names_the_exception_type():
    provider = _provider(_FakeLlmRaises(RuntimeError("boom")))
    with pytest.raises(OcrError) as excinfo:
        asyncio.run(provider.classify_pages(_IMAGES))
    assert "RuntimeError" in str(excinfo.value)


def test_json_decode_error_cause_is_suppressed_not_chained():
    """`from None` on the _parse_json raise site: the JSONDecodeError (whose
    `.doc` attribute holds the full raw text) must not become `__cause__`."""
    provider = _provider(_FakeLlmReturns(f"not json {SENSITIVE_TEXT}"))
    with pytest.raises(OcrError) as excinfo:
        asyncio.run(provider.classify_pages(_IMAGES))
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


def test_ask_exception_cause_is_suppressed_not_chained():
    provider = _provider(_FakeLlmRaises(RuntimeError(f"body: {SENSITIVE_TEXT}")))
    with pytest.raises(OcrError) as excinfo:
        asyncio.run(provider.classify_pages(_IMAGES))
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


# --- Finding 3: dataclass reprs ---------------------------------------------


def test_page_text_repr_excludes_the_transcribed_text():
    page_text = PageText(page=1, text=SENSITIVE_TEXT, confidence=0.9)
    rendered = repr(page_text)
    assert SENSITIVE_ID not in rendered
    assert SENSITIVE_TEXT not in rendered
    assert "page=1" in rendered
    assert "confidence=0.9" in rendered


def test_page_text_str_also_excludes_the_transcribed_text():
    # dataclasses without a custom __str__ fall back to __repr__.
    page_text = PageText(page=1, text=SENSITIVE_TEXT, confidence=0.9)
    assert SENSITIVE_ID not in str(page_text)


def test_page_classification_repr_is_future_proofed():
    classification = PageClassification(
        page=1, kind=PageKind.body, starts_article=True, confidence=0.9
    )
    rendered = repr(classification)
    assert "page=1" in rendered
    assert "kind=" in rendered


# --- Finding 5: body_pages dedupes -----------------------------------------


def test_body_pages_deduplicates_repeated_page_numbers():
    from legal_assistant.docgen.ocr.base import body_pages

    classifications = [
        PageClassification(2, PageKind.body, False, 0.9),
        PageClassification(2, PageKind.body, False, 0.9),
        PageClassification(1, PageKind.body, False, 0.9),
    ]
    assert body_pages(classifications) == [1, 2]
