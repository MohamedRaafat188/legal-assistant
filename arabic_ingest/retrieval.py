# -*- coding: utf-8 -*-
"""Retrieval pipeline — the query-side counterpart to ingestion.

Flow:  query text
   ->  normalize_for_embedding  (SAME normalization applied to documents at
       ingest time — this MUST match, or dense/sparse recall silently drops)
   ->  BGE-M3 encode (dense + sparse)
   ->  Qdrant hybrid search (RRF fusion) with optional metadata filters
   ->  RetrievedArticle results, each carrying the FAITHFUL text and the
       canonical citation label for the assistant to quote verbatim.

The assistant layer (Phase 3) consumes RetrievedArticle. Verifiable citation is
enforced structurally here: every result exposes `citation_label` and
`body_faithful` straight from the stored chunk, so the LLM is handed the exact
citation and exact text and never has to (or should) invent either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qdrant_client import models

import config
from arabic_text import normalize_for_embedding
from embeddings import BGEM3Embedder, HybridVec
from vector_store import LawVectorStore, make_client, filter_by


@dataclass(slots=True)
class RetrievedArticle:
    """One search hit: citation-ready, with faithful text and provenance."""
    chunk_id: str
    citation_label: str
    body_faithful: str        # the ONLY text a citation may quote
    header: str
    score: float
    article_type: str
    article_number: Optional[int]
    article_status: str
    repealed_range: Optional[str]
    law_name: str
    law_number: int
    law_year: int
    division_number: Optional[int]
    division_title: Optional[str]
    book_number: Optional[int]
    book_title: Optional[str]
    part_number: Optional[int]
    part_title: Optional[str]
    part_kind: str
    chapter_number: Optional[int]
    chapter_title: Optional[str]
    section_number: Optional[int]
    section_title: Optional[str]
    subsection_title: Optional[str]
    page_start: int
    page_end: int

    def to_context_block(self) -> str:
        """Render this article for injection into the LLM prompt.

        The citation label, ancestry, and faithful body are presented
        together so the model cites the exact stored string, never one
        composed from memory. A repealed article is flagged explicitly and
        up front -- the LLM must never present it as still in force.
        """
        ancestry = " ▸ ".join(
            x for x in (
                self.division_title, self.book_title,
                "الباب التمهيدي" if self.part_kind == "preamble" else self.part_title,
                self.chapter_title, self.section_title, self.subsection_title,
            ) if x
        )
        lines = [f"[{self.citation_label}]"]
        if ancestry:
            lines.append(ancestry)
        if self.article_status == "repealed":
            lines.append("⚠ ملغاة — هذه المادة لم تعد سارية")
        lines.append(self.body_faithful)
        return "\n".join(lines)


class Retriever:
    """Hybrid (dense + sparse) retriever over the law corpus."""

    def __init__(self, embedder: BGEM3Embedder, store: LawVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def _encode_query(self, query: str) -> HybridVec:
        # CRITICAL: identical normalization to the document side.
        return self.embedder.encode_query(normalize_for_embedding(query))

    @staticmethod
    def _build_filter(
        law_number: Optional[int],
        law_year: Optional[int],
        article_number: Optional[int],
        book_number: Optional[int],
        article_type: Optional[str],
    ) -> Optional[models.Filter]:
        f = filter_by(
            law_number=law_number, law_year=law_year,
            article_number=article_number, book_number=book_number,
            article_type=article_type,
        )
        return f if f.must else None  # None => unfiltered

    @staticmethod
    def _to_article(point: models.ScoredPoint | models.Record) -> RetrievedArticle:
        p = point.payload or {}
        return RetrievedArticle(
            chunk_id=p.get("chunk_id", ""),
            citation_label=p.get("citation_label", ""),
            body_faithful=p.get("body_faithful", ""),
            header=p.get("header", ""),
            # Records from a filter-only lookup (vector_store.filter_only)
            # carry no relevance score -- there's nothing to rank, the
            # filter already pinned an exact match.
            score=getattr(point, "score", 1.0),
            article_type=p.get("article_type", ""),
            article_number=p.get("article_number"),
            article_status=p.get("article_status", "active"),
            repealed_range=p.get("repealed_range"),
            law_name=p.get("law_name", ""),
            law_number=p.get("law_number", 0),
            law_year=p.get("law_year", 0),
            division_number=p.get("division_number"), division_title=p.get("division_title"),
            book_number=p.get("book_number"), book_title=p.get("book_title"),
            part_number=p.get("part_number"), part_title=p.get("part_title"),
            part_kind=p.get("part_kind", "normal"),
            chapter_number=p.get("chapter_number"), chapter_title=p.get("chapter_title"),
            section_number=p.get("section_number"), section_title=p.get("section_title"),
            subsection_title=p.get("subsection_title"),
            page_start=p.get("page_start", 0), page_end=p.get("page_end", 0),
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        law_number: Optional[int] = None,
        law_year: Optional[int] = None,
        article_number: Optional[int] = None,
        book_number: Optional[int] = None,
        article_type: Optional[str] = None,
        prefetch_limit: int = 30,
    ) -> list[RetrievedArticle]:
        """Retrieve the top articles for a query, optionally metadata-filtered."""
        qvec = self._encode_query(query)
        qfilter = self._build_filter(
            law_number, law_year, article_number, book_number, article_type
        )
        points = self.store.hybrid_search(
            qvec, limit=limit, prefetch_limit=prefetch_limit, query_filter=qfilter
        )
        return [self._to_article(pt) for pt in points]

    def lookup_article(
        self,
        article_number: int,
        *,
        article_type: Optional[str] = None,
        law_number: Optional[int] = None,
        law_year: Optional[int] = None,
        book_number: Optional[int] = None,
        limit: int = 5,
    ) -> list[RetrievedArticle]:
        """Fetch a specific article by its number, no embedding involved.

        Use this instead of `search` when the caller already knows the exact
        article_number (e.g. parsed via query_intent.extract_article_reference):
        hybrid semantic/lexical search routinely misranks "give me article N"
        queries, because a bare article number carries almost no distinguishing
        signal once normalized -- an exact metadata filter is both faster and
        strictly more correct here.

        Substantive and issuance articles are numbered in disjoint namespaces
        stored in DIFFERENT payload fields (article_number vs issuance_number
        -- see structure.py): an issuance article has article_number=null.
        article_type therefore picks which field `article_number` is matched
        against; if unknown, match either (an issuance and a substantive
        article never share the same number by coincidence in practice, but
        this stays correct even if they did).
        """
        common = filter_by(law_number=law_number, law_year=law_year,
                            book_number=book_number)
        if article_type == "issuance":
            qfilter = models.Filter(
                must=[*common.must, models.FieldCondition(
                    key="issuance_number", match=models.MatchValue(value=article_number)
                )]
            )
        elif article_type == "substantive":
            qfilter = models.Filter(
                must=[*common.must, models.FieldCondition(
                    key="article_number", match=models.MatchValue(value=article_number)
                )]
            )
        else:
            qfilter = models.Filter(
                must=common.must,
                should=[
                    models.FieldCondition(key="article_number", match=models.MatchValue(value=article_number)),
                    models.FieldCondition(key="issuance_number", match=models.MatchValue(value=article_number)),
                ],
            )
        records = self.store.filter_only(qfilter, limit=limit)
        return [self._to_article(r) for r in records]


def smart_search(
    retriever: Retriever,
    query: str,
    limit: int = 10,
    **filter_kwargs,
) -> list[RetrievedArticle]:
    """Route a query to exact lookup or ranked search, whichever fits.

    If the query names a specific article ("نص المادة ٢ من قانون ...", "المادة
    الثانية من مواد الإصدار"), fetch it by metadata filter instead of ranking
    -- see Retriever.lookup_article for why. If the caller already passed an
    explicit article_number filter, trust that instead of re-parsing the
    query and fall back to ranked search (there's only ~1 match to filter to
    anyway).
    """
    from query_intent import extract_article_reference

    ref = None if filter_kwargs.get("article_number") else extract_article_reference(query)
    if ref is not None:
        return retriever.lookup_article(
            ref.article_number,
            article_type=filter_kwargs.get("article_type") or ref.article_type,
            law_number=filter_kwargs.get("law_number") or ref.law_number,
            law_year=filter_kwargs.get("law_year") or ref.law_year,
            book_number=filter_kwargs.get("book_number"),
            limit=limit,
        )
    return retriever.search(query, limit=limit, **filter_kwargs)


def build_retriever(url: Optional[str] = None,
                    collection: Optional[str] = None) -> Retriever:
    """Construct a Retriever from config (real BGE-M3 embedder + Qdrant)."""
    embedder = BGEM3Embedder()
    # Load the embedder's native deps (pyarrow, via FlagEmbedding->datasets)
    # BEFORE the Qdrant client's Rust extension -- loading them in the
    # opposite order crashes with an access violation on Windows. See
    # BGEM3Embedder.warmup.
    embedder.warmup()
    client = make_client(url or config.QDRANT_URL)
    store = LawVectorStore(client, collection or config.COLLECTION_NAME)
    return Retriever(embedder, store)
