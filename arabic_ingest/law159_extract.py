"""Logical-order extraction for the Companies Law 159/1981 PDF.

The BiDi machinery is imported from `law72_extract` rather than duplicated:
`logicalize_line` and `repair_mirrors` are properties of *this pdftotext build*
(xpdf / Glyph & Cog 4.00, which emits each embedded LTR run bracketed in
RLE/PDF marks under `-layout`), not of the investment law. Law 159 is typeset
differently but extracted by the same binary, so the same walker applies.

What is specific to this document is handled in `law159_glyphs`: a raa/zayn
transposition, two enumerated letter transpositions, and the stranded digit
run in `(مادة ١)` headings.

Scope of the corpus
-------------------
The PDF is 107 pages: the law itself, then a `مذكرة إيضاحية` (explanatory
memorandum to the draft) that begins mid-page 84 and runs to the end. The
memorandum is commentary on the bill, not enacted text -- it is legislative
history, and quoting it as law would be wrong. `split_body` cuts there, so
only the enacted text ever reaches the corpus.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ARABIC_INGEST = Path(r"E:\DL projects\Legal Assistant\arabic_ingest")
sys.path.insert(0, str(ARABIC_INGEST))

from arabic_text import clean_text  # noqa: E402
from law72_extract import PAGE_SEP, logicalize_line, repair_mirrors  # noqa: E402
from law159_glyphs import repair_post_clean, repair_pre_clean  # noqa: E402

BUILD_DIR = Path(__file__).parent / "law159_intermediates"
DATA_DIR = Path(r"E:\DL projects\Legal Assistant\data") / "قانون الشركات"
SOURCE_PDF = "قانون الشركات رقم 159 لسنة 1981.pdf"

# The heading that opens the explanatory memorandum. Matched on the two words
# together so the phrase cannot collide with a passing mention.
MEMORANDUM_MARKER = "مذكرة إيضاحية"


def prepare(data_dir: Path = DATA_DIR, work: Path | None = None) -> Path:
    """Copy the Arabic-named source PDF to an ASCII path.

    This pdftotext build mangles non-ANSI paths and fails with an I/O error
    before it ever opens the file, so the copy is not optional.
    """
    work = work or (BUILD_DIR / "pdf")
    work.mkdir(parents=True, exist_ok=True)
    dst = work / "law159.pdf"
    shutil.copyfile(data_dir / SOURCE_PDF, dst)
    return dst


def extract_pages(pdf: Path) -> tuple[list[str], list[dict]]:
    """Cleaned, logical-order text per page, plus a per-page glyph-repair audit.

    The audit is returned per page rather than merged because the memorandum
    pages are discarded by `split_body`; folding them in would report
    "unreviewed" tokens for text that never reaches the corpus.
    """
    proc = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", "-layout", str(pdf), "-"],
        capture_output=True,
        check=True,
    )
    raw = proc.stdout.decode("utf-8", errors="replace")
    # NFKC folds the presentation-form glyphs back to base letters and leaves
    # the directional marks the walker needs untouched.
    raw = unicodedata.normalize("NFKC", raw)

    pages = raw.split(PAGE_SEP)
    if pages and not pages[-1].strip():
        pages = pages[:-1]

    out: list[str] = []
    audits: list[dict] = []
    for page in pages:
        fixed = "\n".join(logicalize_line(ln) for ln in page.split("\n"))
        # Pre-clean first so tatweel is gone and the Persian letterforms are
        # Arabic ones by the time clean_text's spacing rules and repair_mirrors
        # look at the text.
        fixed = repair_pre_clean(fixed)
        cleaned, audit = repair_post_clean(repair_mirrors(clean_text(fixed)))
        out.append(cleaned.strip())
        audits.append(audit)

    return out, audits


def merge_audits(audits: list[dict]) -> dict:
    """Fold per-page audits into one report."""
    totals: dict[tuple[str, str], int] = {}
    for audit in audits:
        for bad, good, count in audit["enumerated_fixes"]:
            totals[(bad, good)] = totals.get((bad, good), 0) + count
    return {
        "unreviewed_ar_tokens": sorted({t for a in audits for t in a["unreviewed_ar_tokens"]}),
        "enumerated_fixes": sorted((bad, good, n) for (bad, good), n in totals.items()),
        "substantive_defects": sorted({d for a in audits for d in a["substantive_defects"]}),
    }


def split_body(pages: list[str], audits: list[dict] | None = None):
    """Return (pages of enacted text, [per-page audits], 1-indexed memorandum page).

    The memorandum begins partway through a page, so that page is truncated at
    the marker rather than dropped -- article 184 ends a few lines above it.
    """
    for index, page in enumerate(pages):
        position = page.find(MEMORANDUM_MARKER)
        if position != -1:
            body = pages[:index] + [page[:position].rstrip()]
            kept = audits[: index + 1] if audits is not None else []
            return body, kept, index + 1
    raise ValueError(f"{MEMORANDUM_MARKER!r} not found -- refusing to guess where the law ends")


def extract_body() -> tuple[list[str], dict, int]:
    """Extract, repair and cut in one call: (body pages, audit, memorandum page)."""
    pages, audits = extract_pages(prepare())
    body, kept, memo_page = split_body(pages, audits)
    return body, merge_audits(kept), memo_page


if __name__ == "__main__":
    body, audit, memo_page = extract_body()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    dump = "\n".join(f"===== PAGE {i} =====\n{p}" for i, p in enumerate(body, 1))
    (BUILD_DIR / "law159_body.txt").write_text(dump, encoding="utf-8")
    print(f"enacted text: {len(body)} pages (memorandum starts on page {memo_page})")
    print(f"chars: {sum(len(p) for p in body)}")
    print(f"enumerated fixes: {audit['enumerated_fixes']}")
    print(f"unreviewed ا[رز] tokens: {audit['unreviewed_ar_tokens'] or 'none'}")
    for marker, note in audit["substantive_defects"]:
        print(f"  DEFECT LEFT IN PLACE: {marker} -- {note}")
