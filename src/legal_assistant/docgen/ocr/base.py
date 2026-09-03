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

    # No field here carries document text today, but this class is rendered
    # by whatever logs/asserts on it (Task 14's orchestrator, in particular).
    # A custom __repr__ means a future text-bearing field must be added to it
    # deliberately to appear in repr() -- the default frozen-dataclass repr
    # would include it silently.
    def __repr__(self) -> str:
        return (
            f"PageClassification(page={self.page!r}, kind={self.kind!r}, "
            f"starts_article={self.starts_article!r}, confidence={self.confidence!r})"
        )


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    confidence: float

    # `text` is transcribed document content -- possibly a partner's national
    # ID or passport number, copied verbatim per the extraction prompt. The
    # default frozen-dataclass repr would embed it in full in any log line,
    # assertion failure, or debugger frame. Report only its length.
    def __repr__(self) -> str:
        return (
            f"PageText(page={self.page!r}, text=<{len(self.text)} chars>, "
            f"confidence={self.confidence!r})"
        )


@runtime_checkable
class OcrProvider(Protocol):
    async def classify_pages(
        self, images: Sequence[PageImage]
    ) -> list[PageClassification]: ...

    async def extract(self, images: Sequence[PageImage]) -> list[PageText]: ...

    async def extract_fields(self, images: Sequence[PageImage], schema: dict) -> dict: ...


def body_pages(classifications: Sequence[PageClassification]) -> list[int]:
    """Sorted, de-duplicated page numbers of the contract body.

    A well-behaved provider never emits two `PageClassification`s for the
    same page, but this is a public function over caller-supplied data, not
    just the Gemini provider's output -- silently returning `[2, 2]` for a
    duplicate is the same silent-data-shape bug this codebase has hit
    before. A duplicate page number is unambiguous (it is still just page 2
    of the body) and de-duplicating is what "sorted page numbers" already
    implies, so this dedupes rather than raising.
    """
    return sorted({c.page for c in classifications if c.kind is PageKind.body})


def get_provider(settings: Settings | None = None) -> OcrProvider:
    """The configured OCR provider."""
    settings = settings or get_settings()
    if settings.ocr_provider == "gemini":
        from legal_assistant.docgen.ocr.gemini import GeminiOcrProvider

        return GeminiOcrProvider(settings)
    raise ValueError(f"unknown ocr_provider: {settings.ocr_provider!r}")
