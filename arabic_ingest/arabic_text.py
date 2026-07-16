# -*- coding: utf-8 -*-
"""Arabic text normalization for the Egyptian-law RAG corpus.

Two distinct normalization levels, used for two different purposes:

1. ``clean_text``  -> the FAITHFUL text.
   Fixes only *extraction artifacts*: presentation-form glyphs, stray
   BiDi control characters, and spacing damage. It does NOT change letter
   identity, so the output is still an exact textual rendering of the law.
   This is what we store for **display and citation** — a lawyer must see
   the article exactly as enacted.

2. ``normalize_for_embedding`` -> the RETRIEVAL text.
   Applies standard Arabic IR letter-folding (tatweel removal, Farsi-yeh ->
   Arabic-yeh, optional alef/ya/ta-marbuta folding) on top of the faithful
   text. This copy is what we embed and index, so that surface spelling
   variants collapse to one form and retrieval recall goes up.

Store BOTH on every chunk. Embed the second; show the first.
"""
from __future__ import annotations

import logging
import re
import unicodedata

_log = logging.getLogger(__name__)

# --- Private Use Area (PUA) glyph repair ------------------------------------
# The doPDF export of this gazette embeds a font that maps several glyphs into
# the Unicode Private Use Area (U+E000–U+F8FF). NFKC cannot fold these (PUA has
# no decomposition), so without repair they survive as opaque boxes that break
# both display and search. Identified empirically from surrounding-word context
# and confirmed visually against rasterized pages:
#
#   * 509 of 512 occurrences are VOWEL DIACRITICS the font stored privately.
#     These are stripped anyway in the retrieval copy, so their only effect is
#     on display vowelization. The exact haraka for a couple of rare glyphs is
#     a best-effort identification (noted below); getting it slightly wrong is
#     low-stakes because Arabic reads correctly without full vocalization.
#   * Only U+F001 is a real letter cluster (the ligature لمج in "المجني") and
#     MUST be restored to preserve the word.
#
# Any PUA codepoint NOT in this table is removed but logged at WARNING level, so
# a new source document with a different font surfaces loudly instead of
# silently corrupting the corpus.
_PUA_REMAP: dict[int, str] = {
    0xE823: "\u064B",  # FATHATAN  ً  (اعتبارًا, يومًا, بناءً) — visually confirmed
    0xE821: "\u064F",  # DAMMA     ُ  (on verb prefixes: يُعمل, يُصدر) — best effort
    0xE825: "\u0651",  # SHADDA    ّ  (التصرّف, المضرّة)
    0xE82C: "\u0651",  # SHADDA    ّ  (يفصّل, تُضمّن)
    0xE820: "\u0651",  # SHADDA    ّ  (تُضمّن)
    0xF001: "\u0644\u0645\u062C",  # ligature لمج (in "المجني") — real letters
    0xF220: "",        # decorative end-of-line mark — drop
}

_PUA_RANGE = range(0xE000, 0xF900)  # BMP Private Use Area


def repair_pua(text: str) -> str:
    """Replace font-specific Private Use Area glyphs with real Unicode.

    Known glyphs are remapped via ``_PUA_REMAP``; unknown PUA codepoints are
    dropped and logged so new fonts are caught rather than silently corrupting.
    """
    if not any(ord(c) in _PUA_RANGE for c in text):
        return text
    out: list[str] = []
    unknown: set[str] = set()
    for ch in text:
        cp = ord(ch)
        if cp in _PUA_REMAP:
            out.append(_PUA_REMAP[cp])
        elif cp in _PUA_RANGE:
            unknown.add(hex(cp))  # drop, but remember to warn
        else:
            out.append(ch)
    if unknown:
        _log.warning("Unmapped PUA glyphs dropped (add to _PUA_REMAP): %s",
                     ", ".join(sorted(unknown)))
    return "".join(out)

# --- BiDi / directional formatting characters ------------------------------
# These are invisible layout controls that poppler leaves in the stream.
# They carry no linguistic content and break naive string matching, so we
# drop them entirely.
_DIRECTIONAL_CONTROLS: frozenset[int] = frozenset(
    {
        0x200E,  # LEFT-TO-RIGHT MARK
        0x200F,  # RIGHT-TO-LEFT MARK
        0x202A,  # LEFT-TO-RIGHT EMBEDDING
        0x202B,  # RIGHT-TO-LEFT EMBEDDING
        0x202C,  # POP DIRECTIONAL FORMATTING
        0x202D,  # LEFT-TO-RIGHT OVERRIDE
        0x202E,  # RIGHT-TO-LEFT OVERRIDE
        0x2066,  # LEFT-TO-RIGHT ISOLATE
        0x2067,  # RIGHT-TO-LEFT ISOLATE
        0x2068,  # FIRST STRONG ISOLATE
        0x2069,  # POP DIRECTIONAL ISOLATE
        0x061C,  # ARABIC LETTER MARK
        0x200B,  # ZERO WIDTH SPACE
        0x200C,  # ZERO WIDTH NON-JOINER
        0x200D,  # ZERO WIDTH JOINER
        0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
    }
)

# Combining Arabic marks (harakat + tanwin + superscript alef). NFKC/BiDi
# reordering sometimes inserts a space *before* these, detaching a tanwin
# from its host letter ("نافذ ًا" instead of "نافذًا"). We re-attach them.
_COMBINING_MARKS = "\u064B-\u0652\u0670\u0653-\u0655"

# Arabic-Indic digits -> ASCII, for parsing law/article numbers only.
# (The stored text keeps the original Arabic-Indic digits.)
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
_SPACE_BEFORE_MARK = re.compile(rf"[ \t]+([{_COMBINING_MARKS}])")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([،؛؟])")

# Spacing repairs for BiDi-glued clause numbers and punctuation. Arabic-Indic
# digits are U+0660–U+0669; base Arabic letters U+0621–U+064A. We insert a
# space at digit↔letter boundaries (never digit↔digit, so numbers stay intact)
# and after a comma/semicolon glued to the next letter.
_DIGIT_THEN_LETTER = re.compile(r"(?<=[\u0660-\u0669])(?=[\u0621-\u064A])")
_LETTER_THEN_DIGIT = re.compile(r"(?<=[\u0621-\u064A])(?=[\u0660-\u0669])")
_COMMA_GLUED = re.compile(r"([،؛])(?=[\u0621-\u064A])")


def strip_directional_controls(text: str) -> str:
    """Remove invisible BiDi / zero-width formatting characters."""
    return text.translate({cp: None for cp in _DIRECTIONAL_CONTROLS})


def clean_text(raw: str) -> str:
    """Turn raw ``pdftotext`` output into faithful, storable Arabic.

    Pipeline: NFKC (folds presentation forms -> base letters) -> drop BiDi
    controls -> repair diacritic/punctuation spacing -> collapse runaway
    whitespace. Letter identity is preserved.
    """
    # Repair font-specific Private Use Area glyphs BEFORE NFKC (NFKC cannot
    # fold PUA — it has no decomposition — so unmapped glyphs would survive).
    text = repair_pua(raw)
    # NFKC is the critical step: it maps the ~1300 presentation-form glyphs
    # per page (U+FB50–U+FDFF, U+FE70–U+FEFF) back to their base letters.
    text = unicodedata.normalize("NFKC", text)
    text = strip_directional_controls(text)

    # Repair spacing damage left by BiDi reordering.
    text = _SPACE_BEFORE_MARK.sub(r"\1", text)   # re-attach tanwin/harakat
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)  # no space before ، ؛ ؟

    # BiDi extraction glues clause numbers to the following word
    # ("١٢نوفمبر", ".١أعضاء") and drops the space after commas
    # ("العمد،ومشايخ"). These harm tokenization for the embedder and hurt
    # readability, without carrying meaning — repair them. Letter identity is
    # untouched; only spaces are inserted.
    text = _DIGIT_THEN_LETTER.sub(" ", text)   # ٥٤مكرر  -> ٥٤ مكرر
    text = _LETTER_THEN_DIGIT.sub(" ", text)   # المادة٣٠٨ -> المادة ٣٠٨
    text = _COMMA_GLUED.sub(r"\1 ", text)      # العمد،ومشايخ -> العمد، ومشايخ

    # Collapse whitespace without destroying paragraph structure.
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    # Trim trailing spaces on each line.
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def normalize_for_embedding(
    text: str,
    *,
    fold_alef: bool = True,
    fold_yaa: bool = True,
    fold_taa_marbuta: bool = False,
    strip_harakat: bool = True,
) -> str:
    """Produce the retrieval/embedding copy from already-cleaned text.

    Defaults follow standard Arabic IR practice and are tuned for recall on
    lawyer queries, which are rarely diacritized or spelled with hamza care:

    * ``fold_alef``       أ إ آ ٱ -> ا      (hamza on alef is often dropped)
    * ``fold_yaa``        ی ى -> ي           (Farsi-yeh is a font artifact here;
                                              final alef-maqsura varies by typist)
    * ``fold_taa_marbuta`` ة -> ه            (OFF by default: it can conflate
                                              distinct legal terms; enable only
                                              if eval shows it helps)
    * ``strip_harakat``   remove diacritics  (queries are undiacritized)

    Always removes tatweel (ـ), which is purely decorative.
    """
    # Farsi -> Arabic letter forms (these appear only because of the masthead
    # font, never as meaningful spelling).
    text = text.replace("\u06CC", "\u064A")  # ARABIC LETTER FARSI YEH -> YEH
    text = text.replace("\u06A9", "\u0643")  # ARABIC LETTER KEHEH -> KAF
    text = text.replace("\u0640", "")        # remove TATWEEL

    if fold_alef:
        for ch in "أإآٱ":
            text = text.replace(ch, "ا")
    if fold_yaa:
        text = text.replace("\u0649", "\u064A")  # ALEF MAKSURA -> YEH
    if fold_taa_marbuta:
        text = text.replace("\u0629", "\u0647")  # TEH MARBUTA -> HEH
    if strip_harakat:
        text = re.sub(rf"[{_COMBINING_MARKS}]", "", text)

    text = _MULTISPACE.sub(" ", text)
    return text.strip()


_ASCII_TO_ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def ascii_digits_to_arabic(text: str) -> str:
    """Map 0-9 to ٠-٩. Use for lawyer-facing DISPLAY strings (headers, labels)."""
    return text.translate(_ASCII_TO_ARABIC_DIGITS)


def arabic_digits_to_ascii(text: str) -> str:
    """Map ٠-٩ to 0-9. Use for *parsing* numbers, not for storage."""
    return text.translate(_ARABIC_INDIC_DIGITS)


def normalize_for_match(text: str) -> str:
    """Canonical normalization for *matching* structural keywords/ordinals.

    This is the SINGLE function to use whenever text is compared against a
    pattern (book/part/chapter ordinals, keyword detection). Both sides of any
    comparison MUST pass through this exact function — normalizing only one
    side silently breaks matches (e.g. folding the text 'الأول' -> 'الاول' but
    leaving the pattern 'الأول' unfolded drops the match with no error). Keeping
    it centralized here guarantees both sides use identical rules.
    """
    return normalize_for_embedding(
        text,
        fold_alef=True,
        fold_yaa=True,
        fold_taa_marbuta=False,
        strip_harakat=True,
    )


# --- Clause (بند) structure reconstruction ---------------------------------
# Egyptian articles enumerate بنود (clauses). BiDi extraction (a) flattens each
# onto shared/ wrapped lines and (b) moves the marker's trailing punctuation to
# the LEFT of its number ("- ١", ".١", "-١٠" all mean "١-"). We rebuild the
# structure on the *flattened* body: put each بند on its own line and restore
# the marker to canonical logical form. A بند is only recognized at a clause
# boundary (start, or after . : ؛) — this is what stops wrapped cross-reference
# number-lists ("٣٠٨ من قانون…") from being mis-split, since those start with a
# bare digit with no leading dash/dot at a boundary.
_ARD = "\u0660-\u0669"  # Arabic-Indic digits ٠-٩
# Feminine/masculine ordinal words used as major clause markers (أولاً …).
_ORD_CLAUSE = ("أولا", "ثانيا", "ثالثا", "رابعا", "خامسا", "سادسا",
               "سابعا", "ثامنا", "تاسعا", "عاشرا")
_ORD_CLAUSE_ALT = "|".join(_ORD_CLAUSE)

# Clause numbers appear as Arabic-Indic digits (Law 174) OR ASCII digits (Law
# 131 -- a source-font quirk); accept both. Digit identity is preserved as-is.
_DIGITS = rf"0-9{_ARD}"

# Numbered بند: boundary, optional dot/dash, digits, (optional dup punct), space.
_CLAUSE_NUM = re.compile(rf"(^|[.:؛])\s*[.\-]\s*([{_ARD}]+)\s*[-.]?\s+")
# Parenthesized number: boundary then ")(١" / ") (١" / ")١(" / canonical "(١)".
_CLAUSE_PAREN_NUM = re.compile(rf"(^|[.:؛])\s*[()]+\s*([{_DIGITS}]+)\s*[()]*\s+")
# Ordinal word marker, usually parenthesized: "(أولاً)". Harakat/tanwin can sit
# anywhere inside the word (أولاً vs ثانيًا place the tanwin differently), so we
# allow optional combining marks between every letter.
_HARAKAT = r"[\u064B-\u0652\u0670]*"
_ord_flex = "|".join(
    _HARAKAT.join(re.escape(ch) for ch in word) + _HARAKAT for word in _ORD_CLAUSE
)
_CLAUSE_ORD = re.compile(
    rf"(^|[.:؛])\s*[()]*\s*(?P<w>{_ord_flex})\s*[()]*\s+"
)
# BiDi mirrors and spaces the parens around a single-letter enumerator, so the
# source "(أ)" is extracted as ") أ (" (closing paren, letter, opening paren).
# Correct text never brackets a lone letter as ")X(", so this reversed form is an
# unambiguous mirror artifact — repair it everywhere, independent of clause-
# boundary detection (some markers follow a bare space, not . : ؛). Tatweel and
# harakat on the letter are dropped for a canonical marker.
_MIRRORED_LETTER_PAREN = re.compile(rf"\)\s*([ء-ي])ـ?{_HARAKAT}\s*\(")

# BiDi also moves BOTH parens in front of a parenthesized number and glues it to
# the next word: source "(2)" is extracted as ") (2" (close, open, digit, then no
# space). Reversed parens ")(" before a digit never occur in correct text, so this
# is an unambiguous mirror artifact -- restore "(2) " everywhere. The trailing
# space un-glues the following word. Digit identity (ASCII vs Arabic) is preserved.
_MIRRORED_NUM_PAREN = re.compile(rf"\)\s*\(\s*([{_DIGITS}]+)")

# Alphabetic بند marker on its own line: a single Arabic letter in parentheses
# ("(أ)") at a clause boundary. Runs AFTER _MIRRORED_LETTER_PAREN, so the parens
# are already canonical; this only adds the line break, matching the numeric and
# ordinal clause handlers.
_CLAUSE_PAREN_LETTER = re.compile(
    rf"(^|[.:؛])\s*\(\s*([ء-ي])ـ?{_HARAKAT}\s*\)\s+"
)
# Dash bullet (non-numbered): boundary then a dash then a non-digit word.
_CLAUSE_DASH = re.compile(rf"(^|[.:؛])\s*-\s*(?=[^\d\s{_ARD}])")


def format_clause_structure(body: str) -> str:
    """Put each بند on its own line with a canonical marker, on a flattened body.

    Order matters: numbered and parenthesized-number markers are handled before
    the bare dash bullet, so "- ١" is read as a number (→ "١-"), not a bullet.
    """
    def _num(m: "re.Match[str]") -> str:
        lead = m.group(1)
        nl = "\n" if lead else ""
        return f"{lead}{nl}{m.group(2)}- "

    def _paren(m: "re.Match[str]") -> str:
        lead = m.group(1)
        nl = "\n" if lead else ""
        return f"{lead}{nl}({m.group(2)}) "

    def _ord(m: "re.Match[str]") -> str:
        lead = m.group(1)
        nl = "\n" if lead else ""
        return f"{lead}{nl}({m.group('w')}) "

    def _paren_letter(m: "re.Match[str]") -> str:
        lead = m.group(1)
        nl = "\n" if lead else ""
        return f"{lead}{nl}({m.group(2)}) "

    def _dash(m: "re.Match[str]") -> str:
        lead = m.group(1)
        nl = "\n" if lead else ""
        return f"{lead}{nl}- "

    body = _MIRRORED_LETTER_PAREN.sub(r"(\1)", body)
    body = _MIRRORED_NUM_PAREN.sub(r"(\1) ", body)
    body = _CLAUSE_NUM.sub(_num, body)
    body = _CLAUSE_PAREN_NUM.sub(_paren, body)
    body = _CLAUSE_ORD.sub(_ord, body)
    body = _CLAUSE_PAREN_LETTER.sub(_paren_letter, body)
    body = _CLAUSE_DASH.sub(_dash, body)
    # Tidy: no space before the newline, single spaces after.
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n[ \t]+", "\n", body)
    return body.strip()
