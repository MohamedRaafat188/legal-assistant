# -*- coding: utf-8 -*-
"""Glyph-level repairs for the Companies Law 159/1981 PDF.

Same class of fix as `glyph_repair.py` and `law72_glyphs.py`: the shapes drawn
on the page are correct Arabic; only the codepoints the extractor reports are
wrong. None of these change what the text *says*.

This source is a fully-justified, annotated consolidated edition, and its
extraction carries five defects the other law PDFs do not:

1. **The raa/zayn transposition.** A `ر` or `ز` followed by `ا` inside a word
   comes out as a spurious space, then the alef, then the letter:
   `المرافق` -> `الم ارفق`, `الجزاءات` -> `الج ازءات`, `قرارات` -> `ق ار ارت`
   (twice in one word). 286 occurrences over 129 distinct sites. See
   `_repair_transposition` for why this cannot be a blanket rule.
2. **Stranded digit runs.** Extracted with `-layout`, xpdf brackets every
   embedded LTR run in directional marks, so `law72_extract.logicalize_line`
   and `repair_mirrors` put the parens back from the source's own markup
   rather than by guesswork. What survives that is the article heading
   `(مادة ١)`, whose digit still sits outside its own closer: `(مادة )١`.
3. **Detached hamza-below** (U+0655) left in front of the preceding letter, so
   `وإذا` extracts as `ٕواذا`. Identical to the law-72 defect.
4. **Justification tatweel** (U+0640) inside words, and Farsi yeh/keheh
   (U+06CC/U+06A9) standing in for the Arabic letters.
5. **Two letter-level transpositions** the font produces consistently:
   `مستبدلة` -> `مسبتدلة` (28x) and `رأس` -> `أرس` (63x). Both are enumerated
   rather than generalised -- `أرسل` is a real word that a `أرس` -> `رأس` rule
   would destroy, so only the three tokens actually observed are repaired.

What is NOT repaired: `الجراءم` for `الجرائم` (a hamza-seat error in the
source, not a font defect) and the cross-reference in مادة ١٢٩ مكرراً "٧" that
points at «المادة (١٣٩ مكرراً "٢")» where the provision it describes lives in
١٢٩ مكرراً "٢". Both are surfaced by `find_substantive_defects` and left in
place: correcting a statute is not this pipeline's job.
"""
from __future__ import annotations

import re

LETTER = "ء-ي"

# -- 1. codepoint-level swaps, applied before `clean_text` --------------------

_PERSIAN = str.maketrans({"ی": "ي", "ک": "ك"})
_TATWEEL = "ـ"
# The hamza-below that belongs to the alef of the *following* word.
_DETACHED_HAMZA = re.compile("ٕوا")
# Whatever hamza-below survives that rule is a stray mark on a letter that
# never carried one; it renders as a dot the source does not draw.
_STRAY_HAMZA_BELOW = "ٕ"


def repair_pre_clean(text: str) -> str:
    """Codepoint substitutions that must happen before `clean_text`.

    All whole-character swaps or deletions, so they are unaffected by the
    detached-diacritic spacing `clean_text` later repairs.
    """
    text = text.translate(_PERSIAN)
    text = text.replace(_TATWEEL, "")
    text = _DETACHED_HAMZA.sub("وإ", text)
    text = text.replace(_STRAY_HAMZA_BELOW, "")
    return text


# -- 2. the raa/zayn transposition -------------------------------------------

# A blanket `\s+ا[رز]` -> transposed rule is unsafe: Arabic form-VIII words
# genuinely begin with «ارت», so `الذي ارتضاه` would become `الذيراتضاه`.
# Following `glyph_repair.py`, the bounded universe was enumerated and
# classified. The rule can only fire on a token preceded by letter+space, and
# 65 distinct such tokens occur in this document. Exactly three are genuine
# words rather than transposition artifacts:
REVIEWED_GENUINE_TOKENS = frozenset({"ارتضاه", "ارتباط", "ارتكابه"})

# ...and these are the other 62, every one an artifact whose repair yields a
# real word (`ارقبين` -> `المراقبين`, `ازنية` -> `الميزانية`, `ارعى` ->
# `يراعى`). Held as data so that a token outside this audited universe -- a
# genuinely new word in a re-extraction -- is reported instead of silently
# rewritten. The rule is still applied to it, since ~95% of these are
# artifacts, but the audit says so.
REVIEWED_ARTIFACT_TOKENS = frozenset({
    "ار", "ارء", "ارءات", "ارءاته", "ارءاتها", "ارءم", "ارؤها", "ارئب", "ارئم", "اربع",
    "اربعة", "ارت", "ارته", "ارتها", "ارح", "ارحة", "ارحل", "ارخى", "ارر", "اررا",
    "اررات", "ارره", "اررها", "ارض", "ارضات", "ارضها", "ارعاة", "ارعى", "ارغ", "ارف",
    "ارفق", "ارق", "ارقب", "ارقبا", "ارقبة", "ارقبو", "ارقبي", "ارقبين", "ارقها", "ارك",
    "اركمي", "اركهم", "ارم", "ارمة", "ارها", "از", "ازءات", "ازد", "ازرة", "ازل",
    "ازلة", "ازم", "ازمات", "ازماتها", "ازمية", "ازنة", "ازنية", "ازول", "ازولة",
    "ازوله", "ازولها", "ازيا",
})

# Two artifacts landed at the *start* of a word, where there is no preceding
# fragment to rejoin -- `زاد` -> `ازد`, `زاول` -> `ازول`. These are the only
# two in the document, so they are repaired by name.
WORD_INITIAL_TRANSPOSITIONS = {"ازد": "زاد", "ازول": "زاول"}

_TRANSPOSED = re.compile(rf"([{LETTER}])\s+ا([رز])")
# Only tokens in this position are reachable by the rule, so only these need
# reviewing -- an `ازدهاره` that opens a line is never at risk.
_REACHABLE_TOKEN = re.compile(rf"[{LETTER}]\s+(ا[رز][{LETTER}]*)")


def _repair_transposition(text: str) -> tuple[str, list[str]]:
    """Undo the raa/zayn transposition, returning (text, unreviewed_tokens)."""
    unreviewed = [
        t
        for t in _REACHABLE_TOKEN.findall(text)
        if t not in REVIEWED_GENUINE_TOKENS and t not in REVIEWED_ARTIFACT_TOKENS
    ]

    # Shield the genuine words so the general rule cannot reach them.
    shields = {w: f"\x00{i}\x00" for i, w in enumerate(sorted(REVIEWED_GENUINE_TOKENS))}
    for word, token in shields.items():
        text = text.replace(word, token)

    for bad, good in WORD_INITIAL_TRANSPOSITIONS.items():
        text = re.sub(rf"(?<![{LETTER}]){bad}(?![{LETTER}])", good, text)

    # To a fixpoint: `قرارات` extracts as `ق ار ارت`, two artifacts in one word,
    # and the first pass consumes the window the second one needs.
    previous = None
    while previous != text:
        previous = text
        text = _TRANSPOSED.sub(r"\1\2ا", text)

    for word, token in shields.items():
        text = text.replace(token, word)
    return text, unreviewed


# -- 3. stranded digit runs ---------------------------------------------------

# Deliberately NOT a global "(" <-> ")" swap. Under `-layout` the directional
# marks survive, so `law72_extract.repair_mirrors` reorders brackets using the
# source's own bracketing -- principled, where a blanket mirror would be a
# guess. This only mops up the shape that walker leaves behind: an LTR digit
# run sitting past its own closer, `(مادة )١` -> `(مادة ١)`.
_STRANDED_DIGITS = re.compile(r"\(([^()\n]*?)\)\s*([٠-٩]+)")


def _repair_parens(text: str) -> str:
    text = _STRANDED_DIGITS.sub(r"(\1\2)", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


# -- 4. enumerated letter transpositions -------------------------------------

# Only the tokens actually observed. `أرس` -> `رأس` is deliberately not a
# general rule: `أرسل` and `أرسى` are real words it would corrupt.
ENUMERATED_FIXES = {
    "مسبتدلة": "مستبدلة",
    "أرسمالها": "رأسمالها",
    "أرسمال": "رأسمال",
    "أرس": "رأس",
}


def _apply_enumerated(text: str) -> tuple[str, list[tuple[str, str, int]]]:
    log: list[tuple[str, str, int]] = []
    for bad, good in ENUMERATED_FIXES.items():
        pattern = rf"(?<![{LETTER}]){bad}(?![{LETTER}])"
        text, count = re.subn(pattern, good, text)
        if count:
            log.append((bad, good, count))
    return text, log


# -- 5. defects reported rather than repaired --------------------------------

_SUBSTANTIVE_DEFECTS: list[tuple[str, str]] = [
    (
        "الجراءم",
        "hamza seat: the source writes الجراءم for الجرائم in مادة ١٦٤ مكرراً. "
        "Left as written -- it is a typo in the source text, not a font defect.",
    ),
    (
        '١٣٩ مكرراً',
        'مادة ١٢٩ مكرراً "٧" cross-refers to «المادة (١٣٩ مكرراً "٢")», but article '
        '139 has no مكرر and the prohibition it describes is in ١٢٩ مكرراً "٢". '
        "Reported, not corrected: rewriting a cross-reference is rewriting the law.",
    ),
]


def find_substantive_defects(text: str) -> list[tuple[str, str]]:
    """Defects that are wrong in the source itself, so must not be guessed."""
    return [(marker, note) for marker, note in _SUBSTANTIVE_DEFECTS if marker in text]


# -- public entry point -------------------------------------------------------


def repair_post_clean(text: str) -> tuple[str, dict]:
    """Run every post-`clean_text` repair, returning (text, audit).

    `audit["unreviewed_ar_tokens"]` is non-empty only if a re-extraction turned
    up an ا[رز]-initial token outside the classified universe. Those are still
    repaired by the general rule (they are artifacts ~95% of the time), but
    they are surfaced so a new word cannot be silently altered.
    """
    text, unreviewed = _repair_transposition(text)
    text = _repair_parens(text)
    text, enumerated = _apply_enumerated(text)
    return text, {
        "unreviewed_ar_tokens": sorted(set(unreviewed)),
        "enumerated_fixes": enumerated,
        "substantive_defects": find_substantive_defects(text),
    }
