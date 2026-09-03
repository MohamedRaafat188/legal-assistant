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


_CHAIN_LINK_ATTRS = {"__cause__", "__context__", "__traceback__", "__suppress_context__"}


def _walk_exception_chain(exc: BaseException) -> list[BaseException]:
    """Every exception reachable from `exc` via `__cause__` or `__context__`,
    recursively, deduplicated by identity. This is what a naive exception
    walker or structured log serializer (e.g. one that emits `__cause__` and
    `__context__` as nested records, or dumps `vars(exc)`) would traverse --
    `from None` on its own only makes `str()`/`repr()`/`traceback.format_*`
    look clean by setting `__suppress_context__`; it does not detach
    `__context__` itself, so a serializer reading that attribute directly
    still walks into it.
    """
    seen: set[int] = set()
    chain: list[BaseException] = []
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)
    return chain


def _assert_sensitive_absent(exc: Exception, sensitive: str) -> None:
    """The core assertion: `sensitive` must appear NOWHERE in the full
    exception graph -- not in `str()`/`repr()` of any exception in the chain
    (walking both `__cause__` and `__context__`, recursively), not in any
    exception's `.args`, not in any other instance attribute an exception
    happens to carry (e.g. `JSONDecodeError.doc`, which holds the complete
    un-parsed raw text and survives `from None` unless `__context__` is
    explicitly cleared), and not in the traceback as formatted by
    `traceback.format_exception` (what most log handlers actually capture).
    """
    for link in _walk_exception_chain(exc):
        assert sensitive not in str(link), f"leaked via str() of {type(link).__name__}"
        assert sensitive not in repr(link), f"leaked via repr() of {type(link).__name__}"
        for arg in link.args:
            assert sensitive not in str(arg), f"leaked via .args of {type(link).__name__}"
        for attr_name, attr_value in vars(link).items():
            if attr_name in _CHAIN_LINK_ATTRS:
                continue
            assert sensitive not in str(attr_value), (
                f"leaked via .{attr_name} of {type(link).__name__}"
            )
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
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


def test_json_decode_error_is_fully_detached_not_just_suppressed():
    """`_parse_json`'s raise site: `from None` alone clears `__cause__` and
    sets `__suppress_context__`, which is enough to make `str()`/`repr()`/
    `traceback.format_exception` look clean -- but does NOT clear
    `__context__`. Without the explicit `error.__context__ = None`, the
    JSONDecodeError (whose `.doc` attribute holds the full raw text,
    including the synthetic ID) remains reachable via `__context__` for any
    serializer that reads it directly. This assertion is non-vacuous against
    the pre-fix code: before the fix, `__context__` here was the
    JSONDecodeError instance, not None.
    """
    provider = _provider(_FakeLlmReturns(f"not json {SENSITIVE_TEXT}"))
    with pytest.raises(OcrError) as excinfo:
        asyncio.run(provider.classify_pages(_IMAGES))
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    # `__context__` is unset because the raise happens outside the `except`
    # block entirely (see gemini.py), not because it was suppressed while
    # still attached -- so `__suppress_context__` has nothing to suppress
    # and is correctly False here. What matters is `__context__ is None`.
    assert excinfo.value.__suppress_context__ is False


def test_ask_exception_is_fully_detached_not_just_suppressed():
    """`_ask`'s catch-all raise site: same requirement as above. Non-vacuous
    against the pre-fix code: before the fix, `__context__` here was the
    caught `RuntimeError` instance (carrying the synthetic ID in `.args`),
    not None.
    """
    provider = _provider(_FakeLlmRaises(RuntimeError(f"body: {SENSITIVE_TEXT}")))
    with pytest.raises(OcrError) as excinfo:
        asyncio.run(provider.classify_pages(_IMAGES))
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert excinfo.value.__suppress_context__ is False


@pytest.mark.parametrize(
    "factory",
    [_wrong_shape_classify, _wrong_shape_extract, _wrong_shape_extract_fields],
)
def test_content_free_raise_sites_have_no_context_either(factory):
    """The three shape-check raise sites (classify_pages/extract/
    extract_fields "did not return a list/object") are never reached from
    inside an active `except` block -- they only run after `_parse_json`
    returns successfully -- so `__context__` is naturally `None` with no
    explicit clearing needed. Confirmed here rather than merely asserted in
    prose.
    """
    with pytest.raises(OcrError) as excinfo:
        asyncio.run(factory())
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


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
