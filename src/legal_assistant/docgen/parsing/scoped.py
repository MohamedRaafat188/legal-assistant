"""Apply the page map: give each extractor only its own entry's text.

Two filters, in order: the entry's page span, then its article (or the
preamble). Nothing outside survives, so a company name on a bank certificate
two pages away can never be read as THE company name. Everything fails closed
into a flag; nothing is searched for elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from legal_assistant.docgen.pages import ENTRY_SPECS, Entry, PageMap
from legal_assistant.docgen.parsing.articles import (
    ExtractedArticle,
    LookupStatus,
    find_by_ref,
    preamble_text,
    segment_articles,
)
from legal_assistant.docgen.parsing.commercial_register import (
    Party,
    parse_party_table,
    split_capital,
)
from legal_assistant.docgen.parsing.signatures import states
from legal_assistant.docgen.parsing.values import current_value, law_reference, owner_name


@dataclass(frozen=True)
class ScopedText:
    status: LookupStatus
    text: str | None = None
    article: ExtractedArticle | None = None
    truncated: bool = False

    def __repr__(self) -> str:  # text may carry a national ID -- length only
        size = len(self.text) if self.text is not None else None
        return f"ScopedText(status={self.status!r}, text=<{size} chars>)"


def span_text(page_texts: Mapping[int, str], entry: Entry) -> str:
    return "\n".join(page_texts.get(p, "") for p in entry.pages)


def scope(page_texts: Mapping[int, str], entry: Entry) -> ScopedText:
    text = span_text(page_texts, entry)
    if entry.article is None:
        preamble = preamble_text(text)
        if not preamble:
            return ScopedText(LookupStatus.not_found)
        return ScopedText(LookupStatus.found, text=preamble)
    lookup = find_by_ref(segment_articles(text), entry.article)
    if lookup.status is not LookupStatus.found:
        return ScopedText(lookup.status)
    return ScopedText(
        LookupStatus.found,
        text=lookup.article.body,
        article=lookup.article,
        truncated=lookup.truncated,
    )


@dataclass
class AoaExtraction:
    values: dict[str, str | None] = field(default_factory=dict)
    parties: list[Party] = field(default_factory=list)
    # entry name -> flags: not_found, article_not_found, ambiguous,
    # signature_mismatch.
    flags: dict[str, list[str]] = field(default_factory=dict)


# entry -> the scalar fields it fills
ENTRY_FIELDS: dict[str, tuple[str, ...]] = {
    "company_name": ("company_name",),
    "law_reference": ("law_number", "law_year"),
    "company_address": ("company_address",),
    "owner_name": ("owner_name",),
    "partners": (),
    "issued_capital": ("issued_capital", "capital"),
    "shareholders": (),
}


def extract_aoa(
    page_texts: Mapping[int, str], page_map: PageMap, company_type: str
) -> AoaExtraction:
    out = AoaExtraction()
    for name, entry in page_map.entries.items():
        for field_name in ENTRY_FIELDS[name]:
            out.values.setdefault(field_name, None)
        flags: list[str] = []
        scoped = scope(page_texts, entry)
        if scoped.status is LookupStatus.not_found:
            flags.append("article_not_found")
        elif scoped.status is LookupStatus.ambiguous:
            flags.append("ambiguous")
        else:
            concept = ENTRY_SPECS[name].concept
            if concept is not None and entry.article is not None and not states(
                scoped.text, concept
            ):
                flags.append("signature_mismatch")
            if not _fill(name, scoped.text, out):
                flags.append("not_found")
        out.flags[name] = flags
    return out


def _fill(name: str, text: str, out: AoaExtraction) -> bool:
    """Extract `name`'s value(s) from `text` into `out`. False when absent."""
    if name == "law_reference":
        ref = law_reference(text)
        if ref:
            out.values["law_number"], out.values["law_year"] = ref
        return ref is not None
    if name == "owner_name":
        out.values["owner_name"] = owner_name(text)
        return out.values["owner_name"] is not None
    if name in ("partners", "shareholders"):
        out.parties = parse_party_table(text)
        return bool(out.parties)
    if name == "issued_capital":
        _authorized, issued = split_capital(text)
        out.values["issued_capital"] = issued
        out.values["capital"] = current_value("capital", text) or issued
        return issued is not None
    value = current_value(name, text) or None
    out.values[name] = value
    return value is not None
