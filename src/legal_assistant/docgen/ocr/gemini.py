"""Gemini-backed page classification, transcription, and field extraction.

Three calls, each with a different job and a different cost profile:

  classify_pages  -- one cheap low-DPI batch call over every page, so that
                     bank certificates and blank backs never reach the
                     expensive pass.
  extract         -- one call per body page, transcribing VERBATIM. The
                     prompt forbids correcting, completing, or reordering
                     anything, because this text is quoted into a filed legal
                     document as "المادة السادسة قبل التعديل".
  extract_fields  -- the سجل تجاري, read against a JSON schema (one property
                     per numbered box). Read as free text, that 14-box
                     landscape form under a guilloche comes back scrambled.

All three run at temperature=0 via the shared `get_llm` factory.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Sequence

from legal_assistant.config import Settings, get_settings
from legal_assistant.docgen.ocr.base import (
    OcrError,
    PageClassification,
    PageKind,
    PageText,
)
from legal_assistant.docgen.pdf import PageImage
from legal_assistant.llm import get_llm

_log = logging.getLogger(__name__)

_CLASSIFY_PROMPT = """أنت تصنّف صفحات ملف ممسوح ضوئيا لعقد تأسيس شركة مصرية.

لكل صفحة، حدد:
- kind: "body" إذا كانت الصفحة جزءا من نص العقد أو النظام الأساسي نفسه،
  أو "attachment" إذا كانت مستندا مرفقا (شهادة بنكية، إيصال، صورة مستند)،
  أو "signature" إذا كانت صفحة توقيعات أو أختام فقط بلا نص مواد،
  أو "unknown" إذا لم تستطع التحديد.
- starts_article: true إذا كانت الصفحة تبدأ بعنوان مادة (مثل «المادة (٦)»).
- confidence: رقم بين 0 و 1.

أعد JSON فقط، مصفوفة بنفس ترتيب الصفحات المعطاة:
[{"page": <رقم الصفحة>, "kind": "...", "starts_article": true|false, "confidence": 0.0}]
"""

_EXTRACT_PROMPT = """انسخ نص هذه الصفحة من عقد التأسيس نسخا حرفيا كاملا.

قواعد إلزامية:
- لا تصحح أي خطأ إملائي أو نحوي. انسخ ما هو مكتوب.
- لا تكمل أي كلمة ناقصة ولا تخمّن نصا مغطى بختم؛ ضع [غير واضح] مكانه.
- حافظ على الأرقام كما هي (٦ تبقى ٦، و 6 تبقى 6).
- ضع كل عنوان مادة في سطر مستقل، مثل: المادة (٦)
- لا تضف أي شرح أو تعليق أو ترجمة.

أعد JSON فقط: {"text": "...", "confidence": 0.0}
"""

_FIELDS_PROMPT = """استخرج البيانات التالية من مستخرج السجل التجارى المرفق.

- التزم بالمخطط (schema) المعطى حرفيا.
- إذا لم تجد قيمة، اترك الحقل فارغا. لا تخمّن ولا تستنتج.
- انسخ القيم كما تظهر في المستند.

المخطط:
{schema}

أعد JSON فقط مطابقا للمخطط.
"""


def _image_part(image: PageImage) -> dict:
    encoded = base64.b64encode(image.png).decode("ascii")
    return {"type": "image_url", "image_url": f"data:image/png;base64,{encoded}"}


def _parse_json(raw: str) -> object:
    """Parse a model response that may be wrapped in a ```json fence."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise OcrError(f"OCR provider returned non-JSON: {raw[:200]!r}") from e


class GeminiOcrProvider:
    """OcrProvider backed by Gemini. Constructed via `ocr.base.get_provider`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = get_llm(settings=self._settings, model=self._settings.ocr_model)

    async def _ask(self, prompt: str, images: Sequence[PageImage]) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend(_image_part(i) for i in images)
        try:
            response = await self._llm.ainvoke([{"role": "user", "content": content}])
        except Exception as e:  # noqa: BLE001 -- any provider failure is an OcrError
            raise OcrError(f"OCR provider call failed: {e}") from e
        return response.content if isinstance(response.content, str) else str(response.content)

    async def classify_pages(
        self, images: Sequence[PageImage]
    ) -> list[PageClassification]:
        if not images:
            return []
        pages = ", ".join(str(i.page) for i in images)
        raw = await self._ask(f"{_CLASSIFY_PROMPT}\nأرقام الصفحات بالترتيب: {pages}", images)
        payload = _parse_json(raw)
        if not isinstance(payload, list):
            raise OcrError("page classification did not return a list")

        by_page = {}
        for item in payload:
            if not isinstance(item, dict) or "page" not in item:
                continue
            try:
                kind = PageKind(item.get("kind", "unknown"))
            except ValueError:
                kind = PageKind.unknown
            by_page[int(item["page"])] = PageClassification(
                page=int(item["page"]),
                kind=kind,
                starts_article=bool(item.get("starts_article", False)),
                confidence=float(item.get("confidence", 0.0)),
            )

        # A page the model skipped is `unknown`, not silently dropped -- the
        # caller must see that every uploaded page was accounted for.
        return [
            by_page.get(i.page, PageClassification(i.page, PageKind.unknown, False, 0.0))
            for i in images
        ]

    async def extract(self, images: Sequence[PageImage]) -> list[PageText]:
        """Transcribe each page verbatim. Pages are processed concurrently."""
        if not images:
            return []

        async def one(image: PageImage) -> PageText:
            raw = await self._ask(_EXTRACT_PROMPT, [image])
            payload = _parse_json(raw)
            if not isinstance(payload, dict):
                raise OcrError(f"page {image.page}: transcription did not return an object")
            return PageText(
                page=image.page,
                text=str(payload.get("text", "")),
                confidence=float(payload.get("confidence", 0.0)),
            )

        results = await asyncio.gather(*(one(i) for i in images))
        return sorted(results, key=lambda p: p.page)

    async def extract_fields(self, images: Sequence[PageImage], schema: dict) -> dict:
        if not images:
            return {}
        prompt = _FIELDS_PROMPT.format(schema=json.dumps(schema, ensure_ascii=False, indent=2))
        payload = _parse_json(await self._ask(prompt, images))
        if not isinstance(payload, dict):
            raise OcrError("field extraction did not return an object")
        return payload
