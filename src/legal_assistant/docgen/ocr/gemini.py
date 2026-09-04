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


def _text_of(content: object) -> str:
    """Extract the plain text of an `AIMessage.content`.

    Gemini 3-generation models return `content` as a list of content blocks
    (each `{"type": "text", "text": "...", "extras": {...}}`, the "extras"
    carrying an opaque thought-signature token), not a bare string the way
    earlier Gemini models did -- `AIMessage.content`'s declared type has
    always been `str | list[str | dict]`, this call site just never
    exercised the list branch until a live Gemini 3 model did. The previous
    fallback, `str(response.content)`, stringified the whole Python list
    (thought-signature blob included) instead of the transcription text,
    which fed something like `[{'type': 'text', 'text': '...', ...}]` into
    `_parse_json` and made every OCR call fail. Only `type == "text"` blocks
    are kept; any other block type (e.g. a future "thinking" block) is
    silently skipped rather than concatenated in, since only rendered text
    is ever meaningful to `_parse_json`'s caller.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _parse_json(raw: str, *, source: str) -> object:
    """Parse a model response that may be wrapped in a ```json fence.

    `raw` is untrusted model output -- a page transcription or a سجل تجارى
    field extraction -- and may contain a partner's national ID or passport
    number copied verbatim. It must never appear in a raised exception, so
    on failure only the call site (`source`), the response length, and the
    parser's own (content-free) error position are reported.

    `json.JSONDecodeError.__str__` is just a position, but the exception
    object also carries the full un-parsed text on its `.doc` attribute,
    which a structured log formatter that serializes exception attributes
    (not just `str(exc)`) would emit. `raise ... from None` is NOT enough to
    stop that: it clears `__cause__` and sets `__suppress_context__ = True`
    (which is what makes `str()`/`repr()`/`traceback.format_exception()`
    look clean), but it does not clear `__context__` -- and raising from
    inside an `except` block makes the interpreter unconditionally overwrite
    the new exception's `__context__` with the exception currently being
    handled, clobbering even a manual `error.__context__ = None` assigned in
    the same block. The only reliable way to keep `__context__` unset is to
    construct and raise the new error *after* the `try`/`except` statement
    has finished, once this frame is no longer "handling" the
    `JSONDecodeError` -- which is exactly what happens below.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    parse_error_position: int | None = None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        parse_error_position = e.pos
    # Deliberately outside the `except` block -- see the docstring above.
    raise OcrError(
        f"OCR provider ({source}) returned a response that is not valid JSON "
        f"(length={len(raw)} chars, parse error at character {parse_error_position})"
    )


class GeminiOcrProvider:
    """OcrProvider backed by Gemini. Constructed via `ocr.base.get_provider`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = get_llm(settings=self._settings, model=self._settings.ocr_model)

    async def _ask(self, prompt: str, images: Sequence[PageImage]) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend(_image_part(i) for i in images)
        # `str(e)` is NOT safe to interpolate into the error message below:
        # `content` above is the prompt plus the page image(s), and
        # HTTP-client/SDK exceptions commonly echo request or response
        # bodies in their message. Only the exception's type name is kept.
        #
        # The `raise` itself is deliberately placed AFTER this try/except
        # statement, not inside the `except` clause. Raising from inside an
        # `except` block makes the interpreter unconditionally set the new
        # exception's `__context__` to the exception being handled -- even
        # if `__context__` is assigned `None` by hand first, and even with
        # `from None` (which only clears `__cause__`/sets
        # `__suppress_context__ = True`, not `__context__`). That would
        # leave `e` -- and whatever request/response content it carries in
        # `.args` or elsewhere -- reachable via `OcrError.__context__` for
        # any serializer that reads that attribute directly. Raising after
        # the `except` block has already exited avoids that: this frame is
        # no longer "handling" `e` by the time the new exception is raised,
        # so `__context__` stays unset.
        error_type_name: str | None = None
        try:
            response = await self._llm.ainvoke([{"role": "user", "content": content}])
        except Exception as e:  # noqa: BLE001 -- any provider failure is an OcrError
            error_type_name = type(e).__name__
        if error_type_name is not None:
            raise OcrError(f"OCR provider call failed: {error_type_name}")
        return _text_of(response.content)

    async def classify_pages(
        self, images: Sequence[PageImage]
    ) -> list[PageClassification]:
        if not images:
            return []
        pages = ", ".join(str(i.page) for i in images)
        raw = await self._ask(f"{_CLASSIFY_PROMPT}\nأرقام الصفحات بالترتيب: {pages}", images)
        payload = _parse_json(raw, source="classify_pages")
        if not isinstance(payload, list):
            # Not raised from an active exception (no `from e`/implicit
            # context to worry about), and the message names no document
            # content -- explicit `from None` just documents that.
            raise OcrError("page classification did not return a list") from None

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
            payload = _parse_json(raw, source=f"extract page {image.page}")
            if not isinstance(payload, dict):
                # Content-free (page number only); explicit `from None` as above.
                raise OcrError(
                    f"page {image.page}: transcription did not return an object"
                ) from None
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
        payload = _parse_json(await self._ask(prompt, images), source="extract_fields")
        if not isinstance(payload, dict):
            # Content-free; explicit `from None` as above.
            raise OcrError("field extraction did not return an object") from None
        return payload
