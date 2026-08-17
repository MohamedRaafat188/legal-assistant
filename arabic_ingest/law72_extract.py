"""Logical-order extraction for the investment-law PDF family (72/2017 + amendments).

Why this file exists instead of reusing `pdf_extractor.py` unchanged
--------------------------------------------------------------------
Laws 131 and 174 were extracted with poppler's `pdftotext` in its default
mode, which fully reorders Arabic BiDi runs to logical order. This machine
ships *xpdf's* `pdftotext` (Glyph & Cog 4.00, bundled with Git for Windows),
and on these particular CID-keyed fonts:

  * default mode  -> drops the Arabic entirely (emits only spaces/punctuation)
  * `-raw`        -> Arabic in pure visual order (unusable)
  * `-layout`     -> Arabic words in LOGICAL order, but each embedded LTR run
                     (numbers, and the punctuation attached to them) is dumped
                     in left-to-right screen order.

`-layout` is therefore the only usable mode, and its one defect is
mechanically repairable *because xpdf brackets every such run* in
RLE (U+202B) / PDF (U+202C) marks. The outermost bracket is the RTL line
itself; a *nested* bracket is exactly one embedded LTR run. So we walk the
nesting, and for depth >= 2 we reverse the whitespace-separated tokens and
rotate each token's leading punctuation back to trailing.

Concretely, a source phrase "articles 12, 48, and a new clause" is emitted with
the two numbers in screen order (",48 ,12") and logicalizes back to "12, 48,".

This repair MUST run before `arabic_text.clean_text`, which strips the
directional marks the repair depends on.

Two further source-specific notes:
  * Law 160/2023's PDF has a damaged xref table; `repair_pdf` rewrites it
    with pypdf first.
  * This pdftotext build mangles non-ANSI paths, so PDFs are copied to ASCII
    filenames before extraction.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ARABIC_INGEST = Path(r"E:\DL projects\Legal Assistant\arabic_ingest")
sys.path.insert(0, str(ARABIC_INGEST))

from arabic_text import clean_text  # noqa: E402
from law72_glyphs import (  # noqa: E402
    repair_post_clean,
    repair_pre_clean,
    strip_page_furniture,
)

PAGE_SEP = "\x0c"
RLE, LRE, PDF_MARK = "‫", "‪", "‬"

# Punctuation that BiDi rotates to the front of an embedded LTR run.
_LEADING_PUNCT = "،؛:.,"


def repair_pdf(src: Path, dst: Path) -> Path:
    """Rewrite a PDF through pypdf so a damaged xref table stops blocking xpdf."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(src), strict=False)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    with dst.open("wb") as fh:
        writer.write(fh)
    return dst


def _logicalize_run(run: str) -> str:
    """Turn one screen-order LTR run back into logical order.

    Tokens are reversed, and any punctuation BiDi moved to the head of a token
    is rotated back to its tail. A token that is *only* punctuation is left
    alone -- it already sits at the boundary where it belongs.
    """
    tokens = run.split()
    if not tokens:
        return run
    out = []
    for tok in reversed(tokens):
        stripped = tok.lstrip(_LEADING_PUNCT)
        if stripped and stripped != tok:
            lead = tok[: len(tok) - len(stripped)]
            tok = stripped + lead
        out.append(tok)
    return " ".join(out)


def logicalize_line(line: str) -> str:
    """Repair every nested LTR run in one extracted line.

    Depth 1 is the RTL line itself and is already in logical order; only
    depth >= 2 runs are screen-ordered and need reversing.
    """
    if RLE not in line and LRE not in line:
        return line
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in line:
        if ch in (RLE, LRE):
            if depth == 1:  # entering a nested run -- start buffering
                buf = []
            elif depth >= 2:
                buf.append(ch)
            depth += 1
            continue
        if ch == PDF_MARK:
            depth -= 1
            if depth == 1:  # leaving a nested run -- flush it repaired
                out.append(_logicalize_run("".join(buf)))
                buf = []
            elif depth >= 2:
                buf.append(ch)
            continue
        (buf if depth >= 2 else out).append(ch)
    out.extend(buf)
    return "".join(out)


# --- mirrored bracket/quote repair -----------------------------------------
# A bracket pair wrapping an LTR run is mirrored by BiDi: both delimiters end
# up adjacent and *before* the run. So a source "(23)" extracts as ")(23" and
# a source '"1"' as '""1'. A doubled delimiter immediately preceding a number
# never occurs in correct text, so these are unambiguous artifacts. A percent
# sign travels with the number and belongs after it.
_AR_D = "٠-٩"
_LETTER = "ء-ي"
_PCT = "[%٪]"
_MIRRORED_QUOTES = re.compile(rf'"\s*"\s*({_PCT}?)\s*([{_AR_D}]+)')
_MIRRORED_PARENS = re.compile(rf"\)+\s*\(\s*({_PCT}?)\s*([{_AR_D}]+)")
# A *word* run wrapped the same way, e.g. the issuance-article headers.
_MIRRORED_WORD_PAREN = re.compile(r"\)([^()\n]{3,40}?)\(")
# ...and a lone enumerator letter: the source "(أ)" extracts as ")أ(". Correct
# text never brackets a single letter in reversed parens, so this is safe.
_MIRRORED_LETTER_PAREN = re.compile(rf"\)\s*([{_LETTER}])\s*\(")
# An unpaired opening paren glued to a trailing number is the same mirror with
# its closer lost across a run boundary; restore both delimiters.
_ORPHAN_OPEN_PAREN = re.compile(rf"(?<=[{_LETTER}])\(\s*([{_AR_D}]+)(?![{_AR_D})])")
# A bracketed number can also come out with BOTH delimiters the same way round
# -- "(٣٥٪(" or ")٥٥٪)" -- when the run boundary falls inside the brackets.
# Requiring digits in between keeps this from touching ordinary parentheses.
_SAME_DELIM_OPEN = re.compile(rf"\((\s*[{_AR_D}]+\s*{_PCT}?)\(")
_SAME_DELIM_CLOSE = re.compile(rf"\)(\s*[{_AR_D}]+\s*{_PCT}?)\)")
# ...and the closing paren can drift a space to the right of its number.
_SPACED_CLOSE_PAREN = re.compile(rf"([{_AR_D}])\s+\)")
# BiDi leaves a delimited number's colon detached: '"22" :' -> '"22":'.
_COLON_AFTER_BRACKET = re.compile(r'(["\)])\s+:')
# ...and glues a definition colon to the following word.
_COLON_GLUED = re.compile(rf":(?=[{_LETTER}])")
# A closing bracket can end up glued to the next word: "(١٣)من" -> "(١٣) من".
_CLOSE_PAREN_GLUED = re.compile(rf"\)(?=[{_LETTER}])")


def repair_mirrors(text: str) -> str:
    text = _MIRRORED_QUOTES.sub(lambda m: f'"{m.group(2)}{m.group(1)}"', text)
    text = _MIRRORED_PARENS.sub(lambda m: f"({m.group(2)}{m.group(1)})", text)
    text = _MIRRORED_WORD_PAREN.sub(r"(\1)", text)
    text = _MIRRORED_LETTER_PAREN.sub(r"(\1)", text)
    text = _ORPHAN_OPEN_PAREN.sub(r" (\1)", text)
    text = _SAME_DELIM_OPEN.sub(r"(\1)", text)
    text = _SAME_DELIM_CLOSE.sub(r"(\1)", text)
    text = _SPACED_CLOSE_PAREN.sub(r"\1)", text)
    text = _COLON_AFTER_BRACKET.sub(r"\1:", text)
    text = _COLON_GLUED.sub(": ", text)
    text = _CLOSE_PAREN_GLUED.sub(") ", text)
    return text


def extract_pages(pdf: Path) -> list[str]:
    """Cleaned, logical-order text for every page of ``pdf``."""
    proc = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", "-layout", str(pdf), "-"],
        capture_output=True, check=True,
    )
    raw = proc.stdout.decode("utf-8", errors="replace")
    # NFKC folds the ~1300 presentation-form glyphs per page back to base
    # letters and leaves the directional marks the repair needs untouched.
    raw = unicodedata.normalize("NFKC", raw)

    pages = raw.split(PAGE_SEP)
    if pages and not pages[-1].strip():
        pages = pages[:-1]

    out = []
    for page in pages:
        fixed = "\n".join(logicalize_line(ln) for ln in page.split("\n"))
        # Glyph repairs run before clean_text so that tatweel is gone and the
        # extended-block digits are ordinary Arabic-Indic ones by the time
        # clean_text's spacing rules and repair_mirrors look for digits.
        fixed = repair_pre_clean(fixed)
        cleaned = repair_post_clean(repair_mirrors(clean_text(fixed)))
        # The Gazette masthead repeats on every page of 160/2023 and would
        # otherwise be spliced into the middle of any article that spans a
        # page break.
        out.append(strip_page_furniture(cleaned).strip())
    return out


def prepare(data_dir: Path, work: Path) -> dict[str, Path]:
    """Copy the Arabic-named source PDFs to ASCII paths, repairing 160/2023."""
    work.mkdir(parents=True, exist_ok=True)
    names = {
        "base72": "قانون الاستثمار رقم 72 لسنة 2017.pdf",
        "amend141": "قانون رقم 141 لسنة 2019.pdf",
        "amend160": "قانون رقم 160 لسنة 2023.pdf",
    }
    paths = {}
    for key, name in names.items():
        dst = work / f"{key}.pdf"
        shutil.copyfile(data_dir / name, dst)
        if key == "amend160":
            dst = repair_pdf(dst, work / "amend160_fixed.pdf")
        paths[key] = dst
    return paths


# Intermediates (ASCII-named PDF copies, per-stage text dumps) are build
# artefacts, not corpus data -- they live under one directory that can be
# deleted and regenerated at any time.
BUILD_DIR = Path(__file__).parent / "law72_intermediates"

DATA_DIR = Path(r"E:\DL projects\Legal Assistant\data") / "قانون الاستثمار"


if __name__ == "__main__":
    paths = prepare(DATA_DIR, BUILD_DIR / "pdf")
    for key, pdf in paths.items():
        pages = extract_pages(pdf)
        text = "\n".join(
            f"===== PAGE {i} =====\n{p}" for i, p in enumerate(pages, 1)
        )
        (BUILD_DIR / f"{key}_fixed.txt").write_text(text, encoding="utf-8")
        print(f"{key}: pages={len(pages)} chars={len(text)}")
