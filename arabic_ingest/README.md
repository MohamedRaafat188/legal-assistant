# arabic_ingest — Egyptian-law PDF ingestion (Phase 1)

Turns an official-gazette law PDF into clean, article-level structured data,
ready for the RAG chunker (Phase 2).

## Folder contents

Everything is flat in this one folder. No subfolders, no `pip install`.

**Library modules** (imported by the tools; you don't run these directly):
- `arabic_text.py` — cleans Arabic PDF text (fixes glyph shapes, Private-Use
  font glyphs, BiDi spacing, clause-number/comma spacing) and provides the two
  normalization levels: faithful (for display/citation) and embedding (for
  retrieval).
- `pdf_extractor.py` — reads the PDF page-by-page in correct logical order
  (wraps poppler's `pdftotext`).
- `structure.py` — detects books / parts / chapters / articles and the
  issuance articles (مواد الإصدار).
- `articles.py` — the single source of truth for slicing pages into article
  records (used by both the preview tool and the chunker).

**Runnable tools:**
- `inspect_corpus.py` — prints the structural report and writes the structure
  tree as JSON.
- `preview_articles.py` — extracts every article body with its context and
  writes them as JSON for review (thin wrapper over `articles.py`).
- `chunker.py` — produces the final vector-DB-ready chunks (metadata header +
  faithful/normalized text + citation label + metadata) as JSON.

**Generated outputs** (safe to delete/regenerate):
- `structure_law174.json` — the book/part/chapter/article tree.
- `articles_preview.json` — all 552 articles (6 issuance + 546 substantive)
  with body text and metadata.
- `chunks_law174.json` — the final article-level chunks ready for embedding.

## One-time setup

The extractor needs poppler's `pdftotext` binary on your PATH.

- Conda: `conda install -c conda-forge poppler`
- Otherwise: download the poppler Windows build, unzip, add its `\bin` to PATH.

Verify: `pdftotext -v` should print a version.

## Tool 1 — inspect the structure

```
python inspect_corpus.py "PATH\TO\law.pdf"
python inspect_corpus.py "PATH\TO\law.pdf" --json structure_law174.json
```
Prints counts (should be 6 books / 23 parts / 46 chapters / 546 articles for
Law 174/2025), an article-integrity check (complete, gap-free, no duplicates),
and the full hierarchy tree. Use this to confirm a new law parsed correctly.

## Tool 2 — preview the article bodies

```
python preview_articles.py "PATH\TO\law.pdf" --json articles_preview.json
```
Writes one record per article to `articles_preview.json`. Each record:

| field | meaning |
|-------|---------|
| `article_type` | `"substantive"` (مادة N) or `"issuance"` (مواد الإصدار) |
| `article_number` | 1..546 for substantive; `null` for issuance |
| `issuance_number` | 1..6 for issuance; `null` otherwise |
| `citation_label` | canonical Arabic citation, e.g. `المادة (26) من قانون الإجراءات الجنائية` |
| `law_name` / `law_number` / `law_year` | law-level identity |
| `book_*` / `part_*` / `chapter_*` | structural context (number + title) |
| `page_start` / `page_end` | source pages (traceability) |
| `char_count` | body length |
| `body` | the faithful article text |

Open the JSON to eyeball articles before chunking.

## Using the library in your own code

```python
from pdf_extractor import PdfTextExtractor
from structure import detect_structure

pages = PdfTextExtractor().extract_pages("law.pdf")
st = detect_structure(pages)
print(st.n_books, st.n_articles, st.article_integrity().is_complete)
```

## Notes / current limitations

- `pdftotext` (poppler) is the only extraction engine that yields correct
  logical-order Arabic; the module refuses to run without it.
- Clause (بند) structure is reconstructed: each بند is put on its own line and
  its marker restored to canonical form — numbered `١-`, dash bullets `- `,
  ordinals `(أولاً)`. Two source-level limits remain: a بند whose marker was
  lost/scrambled in the PDF can't be recovered, and a transition paragraph with
  no marker stays attached to the preceding بند. Content is always complete.
- Law-level metadata (name/number/year) is currently hard-coded in
  `preview_articles.py` for this one sample law; the real pipeline will pass it
  in per source document.


## Phase 2 — RAG core (embeddings + vector DB)

Model: **BGE-M3** (hybrid dense + sparse). DB: **Qdrant** (self-hosted, hybrid search).

New modules:
- `config.py` — env-driven config (model, dims, Qdrant URL, collection).
- `embeddings.py` — `BGEM3Embedder`: dense + sparse encoders (lazy model load).
- `vector_store.py` — `LawVectorStore`: Qdrant collection (dense+sparse named
  vectors), idempotent upserts, hybrid RRF search, `filter_by(...)` helpers.
- `ingest.py` — pipeline CLI: chunks JSON -> embeddings -> Qdrant.

### Install
```
pip install -r requirements.txt
```
First run downloads the BGE-M3 weights (~2.3 GB) from HuggingFace.

### Ingest a law — local dev (no Docker needed)
By default the store is EMBEDDED on disk (folder `./qdrant_storage`), running
inside the Python process. Just run:
```
python chunker.py "PATH\to\law.pdf" --json chunks_law174.json
python ingest.py chunks_law174.json --recreate
```
Embedded mode is single-process (you can't ingest and serve at the same time)
and has no payload indexes — fine for local dev at this corpus size.

### Later — run against a Qdrant server (for the API / deployment)
Start a server (needs Docker Desktop running, or a Railway Qdrant), then point
the pipeline at it via env var:
```
docker run -p 6333:6333 qdrant/qdrant
set QDRANT_URL=http://localhost:6333
python ingest.py chunks_law174.json --recreate
```

Each point stores a dense + sparse vector and a payload with `body_faithful`
(the only text a citation may quote), `citation_label`, `header`, and all
filterable metadata. Retrieval fuses dense + sparse with RRF and supports
metadata filters (e.g. `filter_by(law_number=174, book_number=1)`).
