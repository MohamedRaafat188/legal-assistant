# -*- coding: utf-8 -*-
"""Extract clean, logical-order Arabic text from Egyptian-law PDFs.

Engine choice — settled after codepoint-level testing against a rasterized
ground-truth page:

    poppler `pdftotext`  is the ONLY common engine that reorders Arifabic
    BiDi runs to logical (reading) order. pypdf/pdfplumber/PyMuPDF all emit
    visual order (reversed lines or reversed digits), which is unrecoverable
    without re-implementing the BiDi algorithm. So we shell out to pdftotext
    and post-process with `arabic_text.clean_text`.

System dependency: `poppler-utils` must be installed (provides the
`pdftotext` binary). On Debian/Ubuntu / most Railway images:
    apt-get install -y poppler-utils

This module handles extraction ONLY. Splitting pages into article-level
chunks (with book/part/chapter metadata) is the next stage and lives
separately.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from arabic_text import clean_text

# pdftotext separates pages with a form-feed (U+000C).
_PAGE_SEPARATOR = "\x0c"


@dataclass(frozen=True, slots=True)
class PageText:
    """Cleaned text for a single PDF page (page numbers are 1-indexed)."""

    page_number: int
    text: str

    def __bool__(self) -> bool:  # truthy only if there is real content
        return bool(self.text.strip())


class PdftotextNotFoundError(RuntimeError):
    """Raised when the poppler `pdftotext` binary is not on PATH."""


class PdfTextExtractor:
    """Extract logical-order, base-letter Arabic text from a PDF.

    Parameters
    ----------
    keep_raw:
        If True, also return the pre-cleaning text (debugging only).
    timeout_seconds:
        Hard limit for the pdftotext subprocess per document.
    """

    def __init__(self, *, keep_raw: bool = False, timeout_seconds: int = 120) -> None:
        self._binary = shutil.which("pdftotext")
        if self._binary is None:
            raise PdftotextNotFoundError(
                "poppler's `pdftotext` is required. Install poppler-utils "
                "(e.g. `apt-get install -y poppler-utils`)."
            )
        self.keep_raw = keep_raw
        self.timeout_seconds = timeout_seconds

    # -- internal ----------------------------------------------------------
    def _run(self, pdf_path: Path, first: int | None, last: int | None) -> str:
        """Invoke pdftotext and return its raw stdout (UTF-8)."""
        cmd: list[str] = [self._binary]  # type: ignore[list-item]
        if first is not None:
            cmd += ["-f", str(first)]
        if last is not None:
            cmd += ["-l", str(last)]
        cmd += [str(pdf_path), "-"]  # "-" => write to stdout
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=True,
        )
        # pdftotext emits UTF-8; decode explicitly and never crash on a stray byte.
        return proc.stdout.decode("utf-8", errors="replace")

    # -- public API --------------------------------------------------------
    def extract_page(self, pdf_path: str | Path, page_number: int) -> PageText:
        """Extract and clean a single 1-indexed page."""
        path = Path(pdf_path)
        raw = self._run(path, page_number, page_number)
        raw = raw.replace(_PAGE_SEPARATOR, "")
        return PageText(page_number=page_number, text=clean_text(raw))

    def extract_pages(self, pdf_path: str | Path) -> list[PageText]:
        """Extract and clean every page, preserving page order.

        Returns one ``PageText`` per page. Empty pages are kept (as falsy
        ``PageText``) so that page numbering stays aligned with the source.
        """
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        raw = self._run(path, None, None)
        # Split on the form-feed. The final page has no trailing separator.
        raw_pages = raw.split(_PAGE_SEPARATOR)
        # pdftotext appends a trailing form-feed after the last page, yielding
        # one empty tail element; drop it if present.
        if raw_pages and raw_pages[-1].strip() == "":
            raw_pages = raw_pages[:-1]

        return [
            PageText(page_number=i, text=clean_text(chunk))
            for i, chunk in enumerate(raw_pages, start=1)
        ]

    def extract_document_text(self, pdf_path: str | Path) -> str:
        """Whole document as one cleaned string, pages joined by blank lines."""
        return "\n\n".join(p.text for p in self.extract_pages(pdf_path) if p)
