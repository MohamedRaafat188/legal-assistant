"""PDF -> per-page PNG, the only place PyMuPDF is used.

Rendered twice at different DPI: a cheap pass so the model can classify
every page, then a full-fidelity pass over the contract-body pages only.
Paying full OCR price for bank certificates and blank backs is the single
largest avoidable cost in the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import fitz  # PyMuPDF


class InvalidPdfError(ValueError):
    """The uploaded bytes are not a readable PDF."""


@dataclass(frozen=True)
class PageImage:
    """One rendered page. `page` is 1-based, matching what a lawyer counts."""

    page: int
    png: bytes


def _open(pdf_bytes: bytes) -> fitz.Document:
    try:
        return fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:  # noqa: BLE001 -- PyMuPDF raises several types here
        raise InvalidPdfError(f"could not open the uploaded file as a PDF: {e}") from e


def page_count(pdf_bytes: bytes) -> int:
    with _open(pdf_bytes) as document:
        return document.page_count


def render_pages(
    pdf_bytes: bytes, dpi: int, pages: Sequence[int] | None = None
) -> list[PageImage]:
    """Render `pages` (1-based; all pages when None) to PNG at `dpi`.

    Page numbers outside the document are skipped rather than raising -- a
    stale classification referring to a page that no longer exists should
    degrade, not crash the job.
    """
    with _open(pdf_bytes) as document:
        wanted = list(pages) if pages is not None else range(1, document.page_count + 1)
        images: list[PageImage] = []
        for number in wanted:
            if not 1 <= number <= document.page_count:
                continue
            pixmap = document.load_page(number - 1).get_pixmap(dpi=dpi)
            images.append(PageImage(page=number, png=pixmap.tobytes("png")))
        return images
