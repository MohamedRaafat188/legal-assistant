"""Citation guard: the mechanical backstop that enforces "cite only what was retrieved."

Plain Python, not AI. The system prompt instructs the model to retrieve
before citing -- that is a nudge, not a guarantee. This module is what
actually guarantees it: every citation in the model's answer is checked
against the set of articles retrieved during the conversation, and anything
that doesn't match is caught before it can reach the user.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from legal_assistant.rag.prompts import CITATION_CONTRACT_AR
from legal_assistant.rag.retrieval import RetrievedArticle, normalize_article_number

REGENERATE_INSTRUCTION_AR = """\
تنبيه: تحتوي إجابتك السابقة على استشهاد بمادة لم يتم استرجاعها فعلياً في هذه \
المحادثة. هذا غير مقبول إطلاقاً. أعد صياغة إجابتك الآن بحيث تستشهد حصراً \
بالمواد الواردة في السياق المسترجع أعلاه، ولا شيء غيرها. إن لم تجد في السياق \
المسترجع ما يكفي للإجابة، فاذكر ذلك صراحة بدلاً من الاستشهاد بأي مادة أخرى.
"""

FALLBACK_MESSAGE_AR = (
    "عذراً، لم أتمكن من التحقق من دقة الاستشهادات القانونية اللازمة للإجابة عن "
    "هذا السؤال استناداً إلى المواد المسترجعة فعلياً. يُرجى إعادة صياغة السؤال "
    "أو تحديد المادة أو الموضوع القانوني المقصود بدقة أكبر."
)

_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EASTERN_ARABIC_INDIC_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

_INLINE_ARTICLE_RE = re.compile(
    r"(?:ال)?ماد[ةه]\s*(?:رقم)?\s*[()]*\s*(?P<num>[0-9٠-٩۰-۹]+)\s*[()]*"
    r"(?P<mukarrar>\s*مكرر)?"
)


def _digits_to_ascii(text: str) -> str:
    return text.translate(_ARABIC_INDIC_DIGITS).translate(_EASTERN_ARABIC_INDIC_DIGITS)


def normalize_law_name(name: str) -> str:
    """Light normalization for law-name matching: fold alef/ya/ta-marbuta variants, strip tatweel."""
    if not name:
        return ""
    text = name
    text = text.replace("ـ", "")  # strip tatweel
    for ch in "أإآٱ":  # أ إ آ ٱ
        text = text.replace(ch, "ا")  # -> ا
    text = text.replace("ى", "ي")  # ALEF MAKSURA -> YEH
    text = text.replace("ی", "ي")  # FARSI YEH -> YEH
    text = text.replace("ة", "ه")  # TEH MARBUTA -> HEH
    text = re.sub(r"\s+", " ", text).strip()
    return text


ArticleKey = tuple[str, int, bool]  # (normalized_law_name, article_number, is_mukarrar)


@dataclass
class AllowedSet:
    """The set of articles retrieved during a conversation -- the ONLY thing citable."""

    numbered: set[ArticleKey] = field(default_factory=set)
    # For مواد الإصدار / any article with no number: matched by exact citation_label.
    unnumbered_labels: set[str] = field(default_factory=set)
    # All citation_labels ever retrieved (numbered or not), for lenient structured matching.
    all_labels: set[str] = field(default_factory=set)

    def add(self, article: RetrievedArticle) -> None:
        law_key = normalize_law_name(article.law_name)
        self.all_labels.add(article.citation_label)
        if article.article_number is None:
            self.unnumbered_labels.add(article.citation_label)
        else:
            self.numbered.add((law_key, article.article_number, False))

    def add_many(self, articles: list[RetrievedArticle]) -> None:
        for a in articles:
            self.add(a)

    def contains_numbered(self, law_name: str, article_number: int, is_mukarrar: bool) -> bool:
        return (normalize_law_name(law_name), article_number, is_mukarrar) in self.numbered

    def contains_label(self, citation_label: str) -> bool:
        return citation_label in self.all_labels


@dataclass
class CitationCheck:
    law_name: str
    article_number: int | None
    citation_label: str
    is_valid: bool
    reason: str = ""


@dataclass
class InlineRef:
    raw_match: str
    article_number: int
    is_mukarrar: bool
    is_verified: bool


@dataclass
class GuardResult:
    is_valid: bool
    valid_citations: list[CitationCheck]
    hallucinated_citations: list[CitationCheck]
    unverified_inline_refs: list[InlineRef]
    report: str


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_answer_json(raw_text: str) -> dict:
    """Parse the model's JSON answer, tolerating a markdown code fence wrapper."""
    cleaned = _strip_code_fence(raw_text)
    return json.loads(cleaned)


def _check_structured_citations(citations: list[dict], allowed: AllowedSet) -> list[CitationCheck]:
    checks: list[CitationCheck] = []
    for c in citations:
        law_name = c.get("law_name", "")
        raw_number = c.get("article_number")
        citation_label = c.get("citation_label", "")

        if raw_number is None:
            is_valid = allowed.contains_label(citation_label)
            checks.append(
                CitationCheck(
                    law_name=law_name,
                    article_number=None,
                    citation_label=citation_label,
                    is_valid=is_valid,
                    reason="" if is_valid else "no retrieved article matches this citation_label",
                )
            )
            continue

        try:
            number, is_mukarrar = normalize_article_number(raw_number)
        except ValueError:
            checks.append(
                CitationCheck(law_name, None, citation_label, False, "unparseable article_number")
            )
            continue

        is_valid = allowed.contains_numbered(law_name, number, is_mukarrar)
        checks.append(
            CitationCheck(
                law_name=law_name,
                article_number=number,
                citation_label=citation_label,
                is_valid=is_valid,
                reason="" if is_valid else "article not retrieved in this conversation",
            )
        )
    return checks


def _scan_inline_refs(answer_text: str, allowed: AllowedSet) -> list[InlineRef]:
    """Soft scan: flag inline «المادة ن» mentions not in the retrieved set (review-level)."""
    refs: list[InlineRef] = []
    for m in _INLINE_ARTICLE_RE.finditer(answer_text):
        num_str = _digits_to_ascii(m.group("num"))
        if not num_str.isdigit():
            continue
        number = int(num_str)
        is_mukarrar = bool(m.group("mukarrar"))
        verified = any(number == n and is_mukarrar == mk for (_law, n, mk) in allowed.numbered)
        refs.append(InlineRef(raw_match=m.group(0), article_number=number, is_mukarrar=is_mukarrar, is_verified=verified))
    return refs


def verify(answer_json: dict, allowed: AllowedSet) -> GuardResult:
    """Verify a parsed answer JSON against the conversation's retrieved-context set."""
    answer_text = answer_json.get("answer_text", "")
    citations = answer_json.get("citations", []) or []

    checks = _check_structured_citations(citations, allowed)
    valid = [c for c in checks if c.is_valid]
    hallucinated = [c for c in checks if not c.is_valid]
    inline_refs = _scan_inline_refs(answer_text, allowed)
    unverified_inline = [r for r in inline_refs if not r.is_verified]

    is_valid = len(hallucinated) == 0

    lines = [f"citations: {len(valid)} valid, {len(hallucinated)} hallucinated"]
    for c in hallucinated:
        lines.append(f"  HALLUCINATED: law={c.law_name!r} article={c.article_number} ({c.reason})")
    if unverified_inline:
        lines.append(f"inline refs not in retrieved set (review-level): {len(unverified_inline)}")
        for r in unverified_inline:
            lines.append(f"  UNVERIFIED INLINE: {r.raw_match!r}")
    report = "\n".join(lines)

    return GuardResult(
        is_valid=is_valid,
        valid_citations=valid,
        hallucinated_citations=hallucinated,
        unverified_inline_refs=unverified_inline,
        report=report,
    )


def guarded_generate(
    generate_fn: Callable[[str | None], str],
    allowed: AllowedSet,
) -> tuple[dict, GuardResult]:
    """Verify -> on hard failure regenerate once with the corrective instruction -> else fallback.

    `generate_fn(extra_instruction)` must return the model's raw text answer
    (the JSON contract response); `extra_instruction` is None on the first
    call and REGENERATE_INSTRUCTION_AR on the retry. No answer with an
    unverified citation is ever returned.
    """
    raw = generate_fn(None)
    try:
        answer_json = parse_answer_json(raw)
    except (json.JSONDecodeError, ValueError):
        answer_json = {"answer_text": raw, "citations": []}

    result = verify(answer_json, allowed)
    if result.is_valid:
        return answer_json, result

    raw_retry = generate_fn(REGENERATE_INSTRUCTION_AR)
    try:
        answer_json_retry = parse_answer_json(raw_retry)
    except (json.JSONDecodeError, ValueError):
        answer_json_retry = {"answer_text": raw_retry, "citations": []}

    result_retry = verify(answer_json_retry, allowed)
    if result_retry.is_valid:
        return answer_json_retry, result_retry

    fallback_json = {"answer_text": FALLBACK_MESSAGE_AR, "citations": []}
    return fallback_json, result_retry


__all__ = [
    "CITATION_CONTRACT_AR",
    "REGENERATE_INSTRUCTION_AR",
    "FALLBACK_MESSAGE_AR",
    "AllowedSet",
    "CitationCheck",
    "InlineRef",
    "GuardResult",
    "parse_answer_json",
    "normalize_law_name",
    "verify",
    "guarded_generate",
]
