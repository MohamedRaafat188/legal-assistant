"""Identify what an article is ABOUT, independent of its number.

المركز الرئيسي is المادة (٥) in one sample and المادة (٦) in another, so
signatures no longer locate articles -- the lawyer's page map does, one
number/span at a time. `states` cross-checks the lawyer's page map (it warns
when an entry's article does not look like what they said it is, but never
overrides their choice), and `classify` picks which field a declared article
is the patch target for.

Matching is keyword-based and deliberately dumb: no LLM, no embeddings.
A wrong concept here only mis-targets a patch, and the patcher then fails
closed because the old value will not be found verbatim.
"""

from __future__ import annotations

import enum

from legal_assistant.docgen.arabic import normalize_for_match
from legal_assistant.docgen.parsing.articles import ExtractedArticle


class Concept(enum.StrEnum):
    COMPANY_NAME = "company_name"
    HEAD_OFFICE = "head_office"
    CAPITAL = "capital"
    PURPOSE = "purpose"
    DURATION = "duration"


# Each concept needs ANY of its phrases to appear in the normalized body.
# Phrases are written in ordinary spelling; both sides go through
# normalize_for_match, so hamza/ya/teh-marbuta variants are covered.
_SIGNATURES: dict[Concept, tuple[str, ...]] = {
    Concept.COMPANY_NAME: ("اسم الشركة", "تسمى الشركة", "التسمية"),
    Concept.HEAD_OFFICE: (
        "المركز الرئيسي",
        "مركز الشركة الرئيسي",
        "المقر الرئيسي",
        "موطنها القانوني",
        "موطن الشركة",
    ),
    Concept.CAPITAL: ("رأس مال الشركة", "رأس المال", "راس المال المصدر", "رأسمال"),
    Concept.PURPOSE: ("غرض الشركة", "أغراض الشركة"),
    Concept.DURATION: ("مدة الشركة", "مدة هذه الشركة"),
}

_NORMALIZED = {
    concept: tuple(normalize_for_match(p) for p in phrases)
    for concept, phrases in _SIGNATURES.items()
}


def classify(article: ExtractedArticle) -> Concept | None:
    """The concept this article states, or None if it matches nothing.

    Concepts are tested in declaration order, so an article that mentions
    both رأس المال and المركز الرئيسي is classified by whichever comes first
    in `_SIGNATURES`.
    """
    body = normalize_for_match(article.body)
    for concept, phrases in _NORMALIZED.items():
        if any(phrase in body for phrase in phrases):
            return concept
    return None


def states(text: str, concept: Concept) -> bool:
    """Whether `text` says `concept`. The page-map cross-check: warns when the
    lawyer's article does not look like what they said it is; never overrides."""
    body = normalize_for_match(text)
    return any(phrase in body for phrase in _NORMALIZED[concept])
