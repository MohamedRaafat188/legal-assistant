"""Regenerate the docgen text fixture that can be derived mechanically.

Only the blank GAFI مساهمة نموذج has a PDF text layer. Run this after
replacing that sample, then eyeball the diff -- the committed fixture is what
the parser tests assert against, so it must not change silently.

Usage: python scripts/docgen_make_fixtures.py
"""

from __future__ import annotations

import pathlib

import fitz  # PyMuPDF

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "نموذج-عقد-شركة-مساهمة-ق-159-نهائى.pdf"
TARGET = ROOT / "tests" / "fixtures" / "docgen" / "masahma_namuzag.txt"


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"sample not found: {SOURCE}")
    doc = fitz.open(SOURCE)
    pages = [page.get_text("text") for page in doc]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text("\n\f\n".join(pages), encoding="utf-8")
    print(f"wrote {TARGET} ({len(pages)} pages, {TARGET.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
