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

# Every byte handed to this module is attacker-supplied (an HTTP upload), so
# no PyMuPDF-specific exception type may cross the module boundary -- calling
# code is only told to expect InvalidPdfError. PyMuPDF's own Python-level
# errors (including FileDataError/EmptyFileError) are ValueError/RuntimeError
# subclasses, but the lower-level mupdf C++ bindings (e.g. FzErrorLimit for
# "Overly large image") raise fitz.mupdf.FzErrorBase, which does NOT subclass
# RuntimeError -- both families must be caught. TypeError and other
# programming errors are deliberately left to propagate as themselves.
_PYMUPDF_ERRORS = (ValueError, RuntimeError, fitz.mupdf.FzErrorBase)

# A single rendered A4 page at 600 DPI is already ~4960x7015 px (~35M px,
# ~140MB as an uncompressed RGB pixmap before PNG encoding) -- comfortably
# above both of this project's configured render DPIs (docgen_classify_dpi=80,
# docgen_ocr_dpi=220 in Settings) so legitimate use is never affected, but
# bounded so an attacker-controlled dpi on this untrusted-upload path cannot
# force an arbitrarily large in-memory pixmap. This is well below PyMuPDF's
# own internal image-size ceiling (which only starts raising FzErrorLimit
# around dpi=20000 on a similar page), so this module's own guard is what
# fires first, not PyMuPDF's.
MAX_DPI = 600


class InvalidPdfError(ValueError):
    """The uploaded bytes are not a readable PDF."""


@dataclass(frozen=True)
class PageImage:
    """One rendered page. `page` is 1-based, matching what a lawyer counts."""

    page: int
    png: bytes


def _open(pdf_bytes: bytes) -> fitz.Document:
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except _PYMUPDF_ERRORS as e:
        raise InvalidPdfError(f"could not open the uploaded file as a PDF: {e}") from e
    if document.needs_pass:
        document.close()
        raise InvalidPdfError("the uploaded PDF is password-protected (encrypted)")
    return document


def _check_dpi(dpi: int) -> None:
    if dpi <= 0:
        raise InvalidPdfError(f"dpi must be positive, got {dpi}")
    if dpi > MAX_DPI:
        raise InvalidPdfError(f"dpi {dpi} exceeds the maximum allowed ({MAX_DPI})")


def page_count(pdf_bytes: bytes) -> int:
    with _open(pdf_bytes) as document:
        try:
            return document.page_count
        except _PYMUPDF_ERRORS as e:
            raise InvalidPdfError(f"could not read the PDF's page count: {e}") from e


def render_pages(
    pdf_bytes: bytes, dpi: int, pages: Sequence[int] | None = None
) -> list[PageImage]:
    """Render `pages` (1-based; all pages when None) to PNG at `dpi`.

    Page numbers outside the document (including negative or zero) are
    skipped rather than raising -- a stale classification referring to a
    page that no longer exists should degrade, not crash the job. A
    zero-page document simply yields an empty list.
    """
    _check_dpi(dpi)
    with _open(pdf_bytes) as document:
        try:
            wanted = list(pages) if pages is not None else range(1, document.page_count + 1)
            images: list[PageImage] = []
            for number in wanted:
                if not 1 <= number <= document.page_count:
                    continue
                pixmap = document.load_page(number - 1).get_pixmap(dpi=dpi)
                images.append(PageImage(page=number, png=pixmap.tobytes("png")))
            return images
        except _PYMUPDF_ERRORS as e:
            raise InvalidPdfError(f"could not render a page of the PDF: {e}") from e
