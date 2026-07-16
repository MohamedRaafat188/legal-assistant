"""Extract normalize_for_embedding + its exact dependencies from the ingestion
source, byte-for-byte, and write embedding_service/app/normalize.py.

Avoids any risk of manually retyping combining-Arabic-diacritic ranges.
"""
from pathlib import Path

SRC = Path(r"E:\DL projects\Legal Assistant\arabic_ingest\arabic_text.py")
DST = Path(r"E:\DL projects\Legal Assistant\embedding_service\app\normalize.py")

src_text = SRC.read_text(encoding="utf-8")

# Grab the exact lines defining _COMBINING_MARKS, _MULTISPACE, and normalize_for_embedding.
marker_start = "_COMBINING_MARKS = "
marker_multispace = "_MULTISPACE = re.compile"
func_start = "def normalize_for_embedding("
func_end_marker = "\n\n\n_ASCII_TO_ARABIC_DIGITS"  # first thing after the function

start_idx = src_text.index(marker_start)
combining_line_end = src_text.index("\n", start_idx) + 1
combining_line = src_text[start_idx:combining_line_end]

ms_idx = src_text.index(marker_multispace)
ms_line_end = src_text.index("\n", ms_idx) + 1
multispace_line = src_text[ms_idx:ms_line_end]

func_idx = src_text.index(func_start)
func_end_idx = src_text.index(func_end_marker, func_idx)
func_body = src_text[func_idx:func_end_idx].rstrip() + "\n"

header = '''"""Arabic normalization for the retrieval/embedding text copy.

Byte-for-byte extracted from the ingestion project's arabic_text.py
(normalize_for_embedding and its exact dependencies), via
scripts/extract_normalize.py, to guarantee no transcription drift in the
combining-Arabic-diacritic ranges. Query text must be folded identically to
how document text was folded at ingestion time -- any drift here silently
degrades retrieval, so do not hand-edit this independently of the ingestion
source; re-run the extraction script instead.
"""

from __future__ import annotations

import re

'''

out = header + combining_line + multispace_line + "\n\n" + func_body

DST.write_text(out, encoding="utf-8")
print("Wrote", DST)
print("----")
print(out)
