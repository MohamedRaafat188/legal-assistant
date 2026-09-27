"""The page map: where, in the uploaded عقد, each value lives.

The lawyer gives every placeholder entry -- and every amended article -- a
contiguous page span and an article (a number, or «التمهيد» for the text
before the first heading). Only those pages are sent to the OCR provider, and
each extractor sees only its own entry's text. This module validates and
(de)serialises the map; it never looks at document text.

Page numbers are 1-based PDF positions as the thumbnails show them, never the
printed footer number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from legal_assistant.docgen import pdf
from legal_assistant.docgen.arabic import normalize_for_match
from legal_assistant.docgen.numbering import ArticleRef, parse_article_ref
from legal_assistant.docgen.parsing.signatures import Concept

PREAMBLE = "التمهيد"
_PREAMBLE_NORM = normalize_for_match(PREAMBLE)
_ALL = frozenset({"shakhs_wahed", "zmm", "masahma"})


@dataclass(frozen=True)
class EntrySpec:
    company_types: frozenset[str]
    preamble_allowed: bool
    # Cross-checked against the lawyer's article; warns, never overrides.
    concept: Concept | None
    label: str  # Arabic, for problems and the page-picker UI


ENTRY_SPECS: dict[str, EntrySpec] = {
    "company_name": EntrySpec(_ALL, True, Concept.COMPANY_NAME, "اسم الشركة"),
    "law_reference": EntrySpec(_ALL, True, None, "رقم القانون وسنته"),
    "company_address": EntrySpec(_ALL, False, Concept.HEAD_OFFICE, "عنوان المركز الرئيسي"),
    "owner_name": EntrySpec(frozenset({"shakhs_wahed"}), False, None, "اسم مالك الشركة"),
    "partners": EntrySpec(frozenset({"zmm"}), False, Concept.CAPITAL, "جدول حصص الشركاء"),
    "issued_capital": EntrySpec(
        frozenset({"zmm", "masahma"}), False, Concept.CAPITAL, "رأس المال المصدر"
    ),
    "shareholders": EntrySpec(frozenset({"masahma"}), True, None, "جدول المؤسسين"),
}


def required_entries(company_type: str) -> list[str]:
    return [n for n, spec in ENTRY_SPECS.items() if company_type in spec.company_types]


@dataclass(frozen=True)
class Entry:
    name: str
    first: int
    last: int
    # None means «التمهيد»: the text before the span's first article heading.
    article: ArticleRef | None

    @property
    def pages(self) -> list[int]:
        return list(range(self.first, self.last + 1))


def _article_json(article: ArticleRef | None) -> str:
    if article is None:
        return PREAMBLE
    return str(article.number) + (" مكرر" if article.mukarrar else "")


@dataclass(frozen=True)
class PageMap:
    entries: dict[str, Entry]
    amended: list[Entry]

    def ocr_pages(self) -> list[int]:
        pages: set[int] = set()
        for entry in [*self.entries.values(), *self.amended]:
            pages.update(entry.pages)
        return sorted(pages)

    def to_json(self) -> dict:
        def one(e: Entry) -> dict:
            return {"from": e.first, "to": e.last, "article": _article_json(e.article)}

        return {
            "entries": {name: one(e) for name, e in self.entries.items()},
            "amended": [one(e) for e in self.amended],
        }

    @classmethod
    def from_json(cls, data: Mapping) -> PageMap:
        """Rebuild an ALREADY-VALIDATED map, as stored on the session."""

        def one(name: str, raw: Mapping) -> Entry:
            article = None if _is_preamble(raw["article"]) else parse_article_ref(raw["article"])
            return Entry(name, int(raw["from"]), int(raw["to"]), article)

        return cls(
            entries={n: one(n, r) for n, r in data["entries"].items()},
            amended=[one(f"amended:{i}", r) for i, r in enumerate(data["amended"])],
        )


class PageMapError(ValueError):
    """The submitted map is unusable. `problems` lists every reason, in Arabic."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def _is_preamble(value: object) -> bool:
    return isinstance(value, str) and normalize_for_match(value) == _PREAMBLE_NORM


def _parse_entry(
    name: str,
    label: str,
    raw: object,
    page_count: int,
    preamble_allowed: bool,
    problems: list[str],
) -> Entry | None:
    if not isinstance(raw, Mapping):
        problems.append(f"«{label}»: البيانات غير مكتملة.")
        return None
    first, last, article_raw = raw.get("from"), raw.get("to"), raw.get("article")
    ok = True
    if not (isinstance(first, int) and isinstance(last, int)):
        problems.append(f"«{label}»: أدخل رقم الصفحة الأولى والأخيرة.")
        ok = False
    elif not 1 <= first <= last <= page_count:
        problems.append(
            f"«{label}»: نطاق الصفحات من {first} إلى {last} غير صالح؛ "
            f"الملف من {page_count} صفحة."
        )
        ok = False

    article: ArticleRef | None = None
    if _is_preamble(article_raw):
        if not preamble_allowed:
            problems.append(
                f"«{label}» ({name}): يجب إدخال رقم المادة، ولا يصح اختيار «التمهيد»."
            )
            ok = False
    else:
        article = parse_article_ref(article_raw) if isinstance(article_raw, str) else None
        if article is None:
            problems.append(f"«{label}»: رقم المادة غير صالح. مثال: 6 أو ٦ مكرر.")
            ok = False
    return Entry(name, first, last, article) if ok else None


def validate_page_map(raw: Mapping, company_type: str, page_count: int) -> PageMap:
    """Validate a submitted map against the company type and the عقد's page count.

    Collects EVERY problem before raising, so the lawyer fixes the form in one
    pass rather than one error at a time.
    """
    problems: list[str] = []
    raw = raw if isinstance(raw, Mapping) else {}
    raw_entries = raw.get("entries") if isinstance(raw.get("entries"), Mapping) else {}
    raw_amended = raw.get("amended") if isinstance(raw.get("amended"), list) else []

    required = required_entries(company_type)
    for name in raw_entries:
        if name not in required:
            problems.append(f"الحقل «{name}» لا ينطبق على هذا النوع من الشركات.")

    entries: dict[str, Entry] = {}
    for name in required:
        spec = ENTRY_SPECS[name]
        if name not in raw_entries:
            problems.append(f"«{spec.label}» ({name}): أدخل الصفحات والمادة.")
            continue
        entry = _parse_entry(
            name, spec.label, raw_entries[name], page_count, spec.preamble_allowed, problems
        )
        if entry is not None:
            entries[name] = entry

    amended: list[Entry] = []
    if not raw_amended:
        problems.append("أدخل مادة واحدة على الأقل للتعديل.")
    seen: set[ArticleRef] = set()
    for index, raw_entry in enumerate(raw_amended):
        entry = _parse_entry(
            f"amended:{index}",
            f"المادة المعدلة رقم {index + 1}",
            raw_entry,
            page_count,
            False,
            problems,
        )
        if entry is None:
            continue
        if entry.article in seen:
            problems.append(
                f"المادة {_article_json(entry.article)} مذكورة أكثر من مرة للتعديل."
            )
            continue
        seen.add(entry.article)
        amended.append(entry)

    if problems:
        raise PageMapError(problems)
    return PageMap(entries=entries, amended=amended)


def thumbnail(pdf_bytes: bytes, page: int, dpi: int) -> bytes:
    """One page as PNG, rendered locally. Never stored: it carries the same
    personal data as the upload, so it is rendered per request instead."""
    return pdf.render_pages(pdf_bytes, dpi=dpi, pages=[page])[0].png
