"""PDF -> per-page PNG, the only place PyMuPDF is used.

Rendered for two purposes: a low-DPI thumbnail of one page per request
(`pages.thumbnail`), and a full-fidelity render of only the pages the lawyer
mapped.
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
# above both of this project's configured render DPIs (docgen_thumbnail_dpi=60,
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


def _check_pages(pages: Sequence[int], page_count_: int) -> None:
    """Reject any page request that would otherwise silently shrink the
    result -- an out-of-range/zero/negative index, an empty request, or a
    duplicate index. Task 14 feeds render_pages output straight to a vision
    model, so a caller asking for page 99 must get a hard failure, not a
    document that quietly appears to have fewer pages than it does.
    """
    if len(pages) == 0:
        raise InvalidPdfError("pages must not be empty; omit `pages` to render all pages")
    seen: set[int] = set()
    for number in pages:
        if not 1 <= number <= page_count_:
            raise InvalidPdfError(
                f"page {number} is out of range for a document with {page_count_} page(s)"
            )
        if number in seen:
            raise InvalidPdfError(f"page {number} was requested more than once")
        seen.add(number)


def render_pages(
    pdf_bytes: bytes, dpi: int, pages: Sequence[int] | None = None
) -> list[PageImage]:
    """Render `pages` (1-based; all pages when None) to PNG at `dpi`.

    Every page number in `pages` must be in range (1..page_count), unique,
    and `pages` itself must not be empty -- any violation raises
    InvalidPdfError naming the offending index and the document's actual
    page count, rather than silently returning a shorter list than asked
    for. Pass `pages=None` to render every page (including a genuinely
    zero-page document, which yields an empty list).
    """
    _check_dpi(dpi)
    with _open(pdf_bytes) as document:
        try:
            total_pages = document.page_count
        except _PYMUPDF_ERRORS as e:
            raise InvalidPdfError(f"could not read the PDF's page count: {e}") from e
        # Outside any try/except _PYMUPDF_ERRORS: InvalidPdfError is itself a
        # ValueError, one of the _PYMUPDF_ERRORS types, so raising it from
        # inside such a block would get it re-wrapped with the wrong message.
        if pages is not None:
            wanted = list(pages)
            _check_pages(wanted, total_pages)
        else:
            wanted = list(range(1, total_pages + 1))
        try:
            images: list[PageImage] = []
            for number in wanted:
                pixmap = document.load_page(number - 1).get_pixmap(dpi=dpi)
                images.append(PageImage(page=number, png=pixmap.tobytes("png")))
            return images
        except _PYMUPDF_ERRORS as e:
            raise InvalidPdfError(f"could not render a page of the PDF: {e}") from e
