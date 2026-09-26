"""The OCR seam.

Everything upstream of this module deals in `PageText`, never in a provider.
Cloud only, by decision -- `get_provider` exists so the provider can be
swapped without touching callers, not so a local model can be plugged in.
Only pages in the lawyer's page map ever reach `extract`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from legal_assistant.config import Settings, get_settings
from legal_assistant.docgen.pdf import PageImage


class OcrError(RuntimeError):
    """The OCR provider failed or returned something unusable."""


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    confidence: float
    # Page furniture kept out of `text`: running header/footer, page number,
    # form codes, the scanner app's logo. Kept verbatim for audit; never part
    # of an article.
    margins: str = ""

    # `text` is transcribed document content -- possibly a partner's national
    # ID or passport number, copied verbatim per the extraction prompt. The
    # default frozen-dataclass repr would embed it in full in any log line,
    # assertion failure, or debugger frame. Report only its length.
    def __repr__(self) -> str:
        return (
            f"PageText(page={self.page!r}, text=<{len(self.text)} chars>, "
            f"margins=<{len(self.margins)} chars>, confidence={self.confidence!r})"
        )


@runtime_checkable
class OcrProvider(Protocol):
    async def extract(self, images: Sequence[PageImage]) -> list[PageText]: ...

    async def extract_fields(self, images: Sequence[PageImage], schema: dict) -> dict: ...


def get_provider(settings: Settings | None = None) -> OcrProvider:
    """The configured OCR provider."""
    settings = settings or get_settings()
    if settings.ocr_provider == "gemini":
        from legal_assistant.docgen.ocr.gemini import GeminiOcrProvider

        return GeminiOcrProvider(settings)
    raise ValueError(f"unknown ocr_provider: {settings.ocr_provider!r}")
