"""The OCR seam.

Everything upstream of this module deals in `PageText`, never in a provider.
Cloud only, by decision -- `get_provider` exists so the provider can be
swapped without touching callers, not so a local model can be plugged in.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from legal_assistant.config import Settings, get_settings
from legal_assistant.docgen.pdf import PageImage


class OcrError(RuntimeError):
    """The OCR provider failed or returned something unusable."""


class PageKind(enum.StrEnum):
    # Part of the عقد/نظام itself.
    body = "body"
    # Bank certificates and similar trailing paperwork -- never OCR'd at
    # full fidelity, which is where most of the cost saving comes from.
    attachment = "attachment"
    signature = "signature"
    unknown = "unknown"


@dataclass(frozen=True)
class PageClassification:
    page: int
    kind: PageKind
    starts_article: bool
    confidence: float


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    confidence: float


@runtime_checkable
class OcrProvider(Protocol):
    async def classify_pages(
        self, images: Sequence[PageImage]
    ) -> list[PageClassification]: ...

    async def extract(self, images: Sequence[PageImage]) -> list[PageText]: ...

    async def extract_fields(self, images: Sequence[PageImage], schema: dict) -> dict: ...


def body_pages(classifications: Sequence[PageClassification]) -> list[int]:
    """Sorted page numbers of the contract body."""
    return sorted(c.page for c in classifications if c.kind is PageKind.body)


def get_provider(settings: Settings | None = None) -> OcrProvider:
    """The configured OCR provider."""
    settings = settings or get_settings()
    if settings.ocr_provider == "gemini":
        from legal_assistant.docgen.ocr.gemini import GeminiOcrProvider

        return GeminiOcrProvider(settings)
    raise ValueError(f"unknown ocr_provider: {settings.ocr_provider!r}")
