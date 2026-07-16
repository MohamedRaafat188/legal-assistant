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

## Artifacts (generated; safe to delete/regenerate)
    structure_law174.json   6 books / 23 parts / 46 chapters / 546 articles
    articles_preview.json    552 article records (6 issuance + 546 substantive)
    chunks_law174.json       552 vector-DB-ready chunks

## Requirements
    pip install -r requirements.txt      (+ poppler system dependency)
