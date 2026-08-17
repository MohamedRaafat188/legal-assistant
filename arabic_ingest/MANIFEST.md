# arabic_ingest — verified baseline

Complete, self-consistent snapshot of the Egyptian-law RAG pipeline
(Phase 1 ingestion + Phase 2 ingestion half). All modules import cleanly
together; regenerated artifacts match the current code.

## Verify consistency after any edit
    python -c "import chunker, ingest, vector_store, articles; print('all imports OK')"

## Modules (library)
    arabic_text.py     Arabic cleaning + faithful/normalized normalization
    pdf_extractor.py   poppler pdftotext extraction (logical-order Arabic)
    structure.py       book/part/chapter/article + issuance detection
    articles.py        single source of truth for slicing articles
    config.py          env-driven settings (model, Qdrant, collection)
    embeddings.py      BGE-M3 hybrid (dense + sparse) encoder
    vector_store.py    Qdrant collection, hybrid RRF search, filters, client factory

## Tools (run these)
    inspect_corpus.py    structure report + structure_law174.json
    preview_articles.py  article bodies for review + articles_preview.json
    chunker.py           final chunks -> chunks_law174.json
    ingest.py            embed + upsert chunks into Qdrant

## Law 72/2017 (الاستثمار) — consolidated with its amendments
The investment law needed its own extraction path: this machine's `pdftotext`
is xpdf's, which drops the Arabic entirely on these fonts in default mode, and
the base PDF is an unofficial retyped copy rather than a Gazette scan. The
modules below are that path; the shared library above is reused wherever it fits.

    law72_extract.py         xpdf -layout extraction + BiDi numeric-run repair
    law72_glyphs.py          digit-block / Persian-letterform / tatweel repairs
    law72_corrections.py     reviewed typo table + the article 32/33 patch
    law72_structure.py       article slicing for this law's header conventions
    law72_amendments.py      amendment model + the splicing engine
    law72_amendment_map.py   what 141/2019 and 160/2023 change, and where
    build_law72.py           run the whole pipeline -> chunks_law72.json + audit
    verify_law72.py          coverage, schema-parity and amendment checks

    python build_law72.py && python verify_law72.py

Chunks carry two fields laws 131/174 do not: `metadata.article_suffix`
("مكرر") and `metadata.amendments`. **Read `law72_audit.md` before ingesting** —
it lists every edit made to the source text.

## Artifacts (generated; safe to delete/regenerate)
    structure_law174.json   6 books / 23 parts / 46 chapters / 546 articles
    articles_preview.json    552 article records (6 issuance + 546 substantive)
    chunks_law174.json       552 vector-DB-ready chunks
    chunks_law72.json        106 chunks (10 issuance + 94 articles + 2 مكرر)
    law72_audit.md           every intervention made to the 72/2017 source
    law72_intermediates/     per-stage text dumps (git-ignored)

## Requirements
    pip install -r requirements.txt      (+ poppler system dependency)
