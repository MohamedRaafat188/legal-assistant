import io

import fitz
import pytest

from legal_assistant.docgen.pdf import InvalidPdfError, page_count, render_pages


@pytest.fixture
def three_page_pdf() -> bytes:
    document = fitz.open()
    for n in range(3):
        page = document.new_page()
        page.insert_text((72, 144), f"page {n + 1}")
    buffer = io.BytesIO()
    document.save(buffer)
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


def test_render_pages_ignores_out_of_range_page_numbers(three_page_pdf):
    images = render_pages(three_page_pdf, dpi=72, pages=[1, 99])
    assert [i.page for i in images] == [1]


def test_higher_dpi_produces_a_bigger_image(three_page_pdf):
    small = render_pages(three_page_pdf, dpi=72, pages=[1])[0]
    large = render_pages(three_page_pdf, dpi=200, pages=[1])[0]
    assert len(large.png) > len(small.png)


def test_a_non_pdf_raises_invalid_pdf_error():
    with pytest.raises(InvalidPdfError):
        page_count(b"this is not a pdf")
