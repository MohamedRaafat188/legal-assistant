import io

import fitz
import pytest

from legal_assistant.docgen.pdf import MAX_DPI, InvalidPdfError, page_count, render_pages


@pytest.fixture
def three_page_pdf() -> bytes:
    document = fitz.open()
    for n in range(3):
        page = document.new_page()
        page.insert_text((72, 144), f"page {n + 1}")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def encrypted_pdf() -> bytes:
    """A password-protected PDF, synthesized in-process (no fixture file)."""
    document = fitz.open()
    document.new_page()
    buffer = io.BytesIO()
    document.save(
        buffer,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="pw123",
        owner_pw="owner123",
    )
    return buffer.getvalue()


def test_page_count(three_page_pdf):
    assert page_count(three_page_pdf) == 3


def test_render_pages_returns_png_bytes_for_every_page(three_page_pdf):
    images = render_pages(three_page_pdf, dpi=72)
    assert [i.page for i in images] == [1, 2, 3]
    assert all(i.png.startswith(b"\x89PNG") for i in images)


def test_render_pages_honours_a_page_subset(three_page_pdf):
    images = render_pages(three_page_pdf, dpi=72, pages=[2])
    assert [i.page for i in images] == [2]


def test_render_pages_rejects_out_of_range_page_numbers(three_page_pdf):
    """An out-of-range page index must raise, not silently drop the page --
    Task 14 feeds render_pages output straight to a vision model, so a
    silently-shrunk result would look like a document with fewer pages than
    it actually has."""
    with pytest.raises(InvalidPdfError, match="99"):
        render_pages(three_page_pdf, dpi=72, pages=[1, 99])


def test_render_pages_error_names_the_actual_page_count(three_page_pdf):
    with pytest.raises(InvalidPdfError, match="3"):
        render_pages(three_page_pdf, dpi=72, pages=[99])


def test_higher_dpi_produces_a_bigger_image(three_page_pdf):
    small = render_pages(three_page_pdf, dpi=72, pages=[1])[0]
    large = render_pages(three_page_pdf, dpi=200, pages=[1])[0]
    assert len(large.png) > len(small.png)


def test_a_non_pdf_raises_invalid_pdf_error():
    with pytest.raises(InvalidPdfError):
        page_count(b"this is not a pdf")


def test_encrypted_pdf_page_count_raises_invalid_pdf_error_not_false_success(encrypted_pdf):
    """page_count must not report a page count on a document it cannot
    actually read without a password -- that is a false-positive success."""
    with pytest.raises(InvalidPdfError, match="encrypt"):
        page_count(encrypted_pdf)


def test_encrypted_pdf_render_pages_raises_invalid_pdf_error_not_raw_valueerror(encrypted_pdf):
    with pytest.raises(InvalidPdfError, match="encrypt"):
        render_pages(encrypted_pdf, dpi=72)


def test_oversized_dpi_raises_invalid_pdf_error_not_raw_pymupdf_error(three_page_pdf):
    with pytest.raises(InvalidPdfError):
        render_pages(three_page_pdf, dpi=20000, pages=[1])


def test_dpi_above_the_module_maximum_is_rejected(three_page_pdf):
    with pytest.raises(InvalidPdfError):
        render_pages(three_page_pdf, dpi=MAX_DPI + 1, pages=[1])


def test_dpi_at_the_module_maximum_is_allowed(three_page_pdf):
    images = render_pages(three_page_pdf, dpi=MAX_DPI, pages=[1])
    assert images[0].page == 1


@pytest.mark.parametrize("dpi", [0, -1, -72])
def test_non_positive_dpi_is_rejected(three_page_pdf, dpi):
    with pytest.raises(InvalidPdfError):
        render_pages(three_page_pdf, dpi=dpi, pages=[1])


def test_negative_page_numbers_are_rejected_not_silently_dropped(three_page_pdf):
    with pytest.raises(InvalidPdfError, match="-1"):
        render_pages(three_page_pdf, dpi=72, pages=[-1, 0, 2])


def test_zero_page_index_is_rejected(three_page_pdf):
    with pytest.raises(InvalidPdfError, match="0"):
        render_pages(three_page_pdf, dpi=72, pages=[0, 2])


def test_empty_pages_list_is_rejected_as_a_caller_bug(three_page_pdf):
    """An explicit request for zero pages is a caller bug, not a valid
    request -- pages=None (all pages) is the correct way to ask for
    everything."""
    with pytest.raises(InvalidPdfError):
        render_pages(three_page_pdf, dpi=72, pages=[])


def test_duplicate_page_indices_are_rejected(three_page_pdf):
    """A caller asking for the same page twice gets an explicit error, not
    a result silently deduplicated (or silently doubled) behind their back."""
    with pytest.raises(InvalidPdfError, match="2"):
        render_pages(three_page_pdf, dpi=72, pages=[2, 2])


def test_pages_none_still_renders_every_page(three_page_pdf):
    images = render_pages(three_page_pdf, dpi=72, pages=None)
    assert [i.page for i in images] == [1, 2, 3]


def test_render_pages_error_messages_contain_no_filesystem_paths(three_page_pdf):
    for bad_pages in ([99], [-3], [0, 99], []):
        with pytest.raises(InvalidPdfError) as excinfo:
            render_pages(three_page_pdf, dpi=72, pages=bad_pages)
        message = str(excinfo.value)
        assert "\\" not in message
        assert "/" not in message


def test_zero_page_document_reports_zero_and_renders_nothing(monkeypatch):
    """PyMuPDF itself refuses to save a zero-page PDF, so a zero-page
    document is simulated at the fitz.open() boundary to exercise the
    same code path without relying on a real (unconstructable) fixture."""

    class _FakeZeroPageDocument:
        page_count = 0
        needs_pass = False

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def load_page(self, *_args, **_kwargs):  # pragma: no cover - never called
            raise AssertionError("load_page must not be called on a zero-page document")

    monkeypatch.setattr(fitz, "open", lambda *a, **k: _FakeZeroPageDocument())
    assert page_count(b"irrelevant") == 0
    assert render_pages(b"irrelevant", dpi=72) == []


def test_severely_truncated_pdf_raises_invalid_pdf_error(three_page_pdf):
    truncated = three_page_pdf[:50]
    with pytest.raises(InvalidPdfError):
        page_count(truncated)
    with pytest.raises(InvalidPdfError):
        render_pages(truncated, dpi=72)
