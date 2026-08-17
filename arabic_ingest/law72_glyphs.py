"""Glyph-level repairs for the investment-law PDF family.

Same class of fix as `arabic_ingest/glyph_repair.py`: the shapes drawn on the
page are correct Arabic; only the codepoints the fonts report are wrong. None
of these change what the text *says*.

The two Gazette amendment PDFs (141/2019, 160/2023) are typeset with a Persian
-flavoured font and full justification, which produces four defects that the
base 72/2017 PDF does not have:

1. Extended Arabic-Indic digits (U+06F0-U+06F9) mixed with the ordinary
   Arabic-Indic ones (U+0660-U+0669). They are the same digits from a
   different Unicode block. This one is not cosmetic -- article numbers are
   parsed from these, so "مادة (۱۷)" would otherwise not match article 17.
2. Farsi yeh (U+06CC) and keheh (U+06A9) standing in for Arabic yeh and kaf.
3. Tatweel (U+0640) inserted *inside* words to stretch them for justification
   ("الحـوافز"). Purely typographic; in the base law's headings tatweel is
   decorative but still not part of any word.
4. A combining hamza-below (U+0655) detached from its alef and left in front
   of the preceding waw, so "وإدارته" extracts as "ٕوادارته".

Plus one diacritic-placement artifact: fathatan lands one letter too early
("خصًما" for "خصمًا"). The base letters are already correct, so this only
affects how the text reads on screen -- but `body_faithful` is what a lawyer
sees quoted, so it is worth repairing. The rule is deliberately narrow: it
fires only on letter + fathatan + letter + (alef|hamza|alef-maqsura), which
is the corrupted shape; the correct shape (letter + fathatan + alef) is left
untouched.
"""
from __future__ import annotations

import re

LETTER = "ء-ي"

# 1. extended Arabic-Indic -> Arabic-Indic, and Western -> Arabic-Indic.
#    Law 141/2019 is typeset with Western digits while the base law and
#    160/2023 use Arabic-Indic ones. Leaving both in a single consolidated law
#    would mean "المادة 11" and "المادة ١١" both occur, which breaks the
#    article-number matching the retriever and citation guard rely on.
_EXT_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹0123456789", "٠١٢٣٤٥٦٧٨٩٠١٢٣٤٥٦٧٨٩")
# 2. Persian letterforms -> Arabic
_PERSIAN = str.maketrans({"ی": "ي", "ک": "ك"})
# 3. justification tatweel
_TATWEEL = "ـ"
# 4. detached hamza-below before a waw+alef cluster
_DETACHED_HAMZA = re.compile("ٕوا")
# fathatan one letter early: "خصًما" -> "خصمًا", "أًيا" -> "أيًا"
_EARLY_TANWIN = re.compile(rf"([{LETTER}])ً([{LETTER}])([اءى])")
# ...and the word-final hamza variant: "بناًء" -> "بناءً". Restricted to hamza
# on purpose: "letter + fathatan + alef" (كتابًا, مكررًا) is the CORRECT shape,
# so including alef here would corrupt every properly-written tanwin.
_EARLY_TANWIN_FINAL = re.compile(rf"([{LETTER}])ً(ء)(?![{LETTER}])")
# "مكررًا" whose raa and alef were swapped around the tanwin
_MUKARRAR = re.compile("مكرً(ار|را)")


def repair_pre_clean(text: str) -> str:
    """Codepoint substitutions that must happen before `clean_text`.

    These are all whole-character swaps or deletions, so they are unaffected
    by the detached-diacritic spacing that `clean_text` later repairs.
    """
    text = text.translate(_EXT_DIGITS)
    text = text.translate(_PERSIAN)
    text = text.replace(_TATWEEL, "")
    text = _DETACHED_HAMZA.sub("وإ", text)
    return text


# The 141/2019 font drops the space before a kaf-initial word. Enumerated
# rather than pattern-matched: a general "split before kaf" rule would happily
# cut correct words in half.
_KAF_GLUE = {
    "عجزكل": "عجز كل",
    "وذلككله": "وذلك كله",
    "أيًاكان": "أيًا كان",
    "علىكل": "على كل",
    "ويُنفذكقانون": "ويُنفذ كقانون",
}
# Justification split a word across a space: "قـ اررا" -> (tatweel gone) "ق اررا".
_SPLIT_QARAR = re.compile(r"\bق\s+اررا\b")
# The Gazette running header, repeated at the top of every page of 160/2023.
_GAZETTE_HEADER = re.compile(
    r"[٠-٩]*\s*الجريدة الرسمية\s*[–-]\s*العدد\s*[٠-٩]+\s*مكرر\s*فى\s*[٠-٩]+\s*"
    r"\S+\s*سنة\s*[٠-٩]+\s*[٠-٩]*"
)


def strip_page_furniture(text: str) -> str:
    """Remove the repeated Gazette masthead so it cannot land inside an article."""
    return _GAZETTE_HEADER.sub(" ", text)


def repair_post_clean(text: str) -> str:
    """Diacritic-placement repairs, which need `clean_text` to have run first.

    BiDi extraction leaves a space between a letter and its combining mark;
    `clean_text` re-attaches them. Running these patterns any earlier would
    see "خص ًما" and match nothing.
    """
    text = _MUKARRAR.sub("مكررًا", text)
    for glued, split in _KAF_GLUE.items():
        text = text.replace(glued, split)
    text = _SPLIT_QARAR.sub("قرارًا", text)
    # Twice: a single word can carry two early tanwins, and the first pass
    # consumes the window the second one needs.
    for _ in range(2):
        text = _EARLY_TANWIN.sub("\\1\\2ً\\3", text)
    text = _EARLY_TANWIN_FINAL.sub("\\1\\2ً", text)
    return text
