# -*- coding: utf-8 -*-
"""Amendments to law 159/1981 that its source edition cannot carry.

The source PDF is a consolidated annotated edition current to 2018, so every
amendment after that date is missing from it by construction -- not an
extraction defect, and nothing `law159_glyphs` or `law159_structure` could
recover. Those are supplied here, and each one is recorded in `law159_audit.md`
as an owner-supplied intervention rather than something read off the page.

Same principle as `law72_corrections.apply_patches`: the replacement text comes
from the project owner, is applied against an explicit anchor, and fails loudly
rather than guessing if the anchor no longer matches.

Formatting follows the source's own convention for a repeal -- the note sits at
the top of the article, immediately after the heading, as it does for the nine
articles the source itself marks ملغاة. The difference is that the source drops
the repealed text entirely, while these articles still carry their pre-repeal
wording, which is worth keeping: a lawyer asking after a repealed provision
usually needs to know what it said. The trailing clause on the note is
editorial framing, added so the surviving text cannot be misread as current.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostSourceRepeal:
    """A repeal enacted after the source edition was consolidated."""

    law_number: int
    law_year: int
    note: str
    # A distinctive phrase from the article as parsed, so a re-extraction that
    # changes the text cannot silently attach this note to the wrong wording.
    anchor: str

    def record(self) -> dict:
        return {
            "kind": "repeal",
            "law_number": self.law_number,
            "law_year": self.law_year,
            "scope": None,
            "note": self.note,
            "source": "project owner (post-dates the source edition)",
        }


# Keyed by (article_number, article_suffix), matching Article.key.
POST_SOURCE_AMENDMENTS: dict[tuple[int, str | None], PostSourceRepeal] = {
    (94, None): PostSourceRepeal(
        law_number=194,
        law_year=2020,
        note=(
            "ملغاة بالقانون رقم ١٩٤ لسنة ٢٠٢٠ بشأن إصدار قانون البنك المركزي "
            "والجهاز المصرفي المنشور بالجريدة الرسمية العدد ٣٧ مكرر (و) في "
            "١٥ /٩/ ٢٠٢٠ . وفيما يلي نص المادة قبل إلغائها:"
        ),
        anchor="لا يجوز لعضو مجلس",
    ),
}


class AnchorNotFoundError(RuntimeError):
    """The article no longer contains the text this correction was written against."""


def apply(articles: list) -> list[tuple[object, PostSourceRepeal]]:
    """Apply every post-source amendment in place; return what was applied.

    Raises if an anchor is missing or an article is not found, so a change in
    the extraction surfaces as a build failure instead of a silently skipped
    repeal.
    """
    by_key = {a.key: a for a in articles if a.kind == "substantive"}
    applied: list[tuple[object, PostSourceRepeal]] = []

    for key, amendment in POST_SOURCE_AMENDMENTS.items():
        article = by_key.get(key)
        if article is None:
            raise AnchorNotFoundError(f"article {key} not found; cannot apply its repeal")
        if amendment.anchor not in article.body:
            raise AnchorNotFoundError(
                f"anchor {amendment.anchor!r} missing from article {key} -- "
                "the source text changed, so this correction must be re-reviewed"
            )
        article.body = f"{amendment.note}\n\n{article.body}"
        article.amendments = [amendment.record(), *article.amendments]
        article.status = "repealed"
        applied.append((article, amendment))

    return applied
