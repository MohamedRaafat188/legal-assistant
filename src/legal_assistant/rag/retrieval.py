"""Retrieval over the egyptian_law Qdrant Cloud collection.

Promotes Phase 2's proven retrieval logic (scripts/check_retrieval.py) into
the package: hybrid (dense+sparse, RRF-fused) search with ColBERT rerank for
conceptual queries, and exact metadata-filtered lookup for a known article
number. Citations must be built from `body_faithful`/`clean_text` only --
embeddings and reranking never touch it, they only decide which article to
surface.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from qdrant_client.http import models

from legal_assistant import observability
from legal_assistant.config import Settings, get_settings
from legal_assistant.db.qdrant import get_cloud_client
from legal_assistant.embedding_client import EmbeddingClient
from legal_assistant.rag.law_identity import canonical_law_name

RERANK_CANDIDATES_DEFAULT = 20
TOP_K_DEFAULT = 5

_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# The bis marker: the stem of `article_suffix`, and the substring every
# inflected written form ("مكررًا", "مكرراً") contains.
MUKARRAR = "مكرر"

_HARAKAT_RE = re.compile(r"[ً-ْ]")
_SUFFIX_PUNCT_RE = re.compile(r"[\"'«»()\[\]]")


def normalize_article_suffix(raw: str | None) -> str | None:
    """Fold an article's bis designation to one canonical, comparable form.

    A مكرر article is legally distinct from its base article, and -- as law
    159/1981 makes unavoidable -- distinct from its *siblings* too: مادة ١٢٩
    مكرراً "١" through "٩" are nine separate articles governing one-person
    companies, and مادة ١٣٥ مكرراً "أ" through "د" four more on company
    division. So the suffix cannot be a boolean; it has to carry the whole
    designation, and every way of writing it must fold to the same key:

        "مكرر" / "مكرراً" / "مكررًا"      -> "مكرر"
        'مكرراً "١"' / "مكرر ١" / "مكرر 1" -> "مكرر 1"
        'مكرراً "أ"' / "مكرر أ"            -> "مكرر ا"

    A suffix we do not recognise is kept as-is rather than dropped: an
    unknown designation must never silently match the base article.
    """
    if raw is None:
        return None
    text = str(raw).translate(_ARABIC_INDIC_DIGITS)
    text = text.replace("ـ", "")  # tatweel
    text = _HARAKAT_RE.sub("", text)  # tanween/harakat: مكرراً -> مكررا
    text = _SUFFIX_PUNCT_RE.sub(" ", text)  # the quotes around "١" / "أ"
    for ch in "أإآٱ":
        text = text.replace(ch, "ا")
    text = text.replace("ى", "ي").replace("ی", "ي").replace("ة", "ه")
    text = re.sub(rf"{MUKARRAR}ا?", MUKARRAR, text)  # مكررا -> مكرر
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_article_number(raw: str | int) -> tuple[int, str | None]:
    """Parse a (possibly Arabic-Indic) article number, preserving the مكرر distinction.

    Returns (number, suffix). "163" -> (163, None); "163 مكرر" -> (163, "مكرر");
    '129 مكرراً "5"' -> (129, "مكرر 5"). مكرر articles are legally distinct from
    their base article and from each other, and must never collapse together.

    Only a مكرر remainder is read as a suffix -- trailing prose ("11 من هذا
    القانون") is not a designation and would otherwise fail every lookup.
    """
    if isinstance(raw, int):
        return raw, None
    text = str(raw).translate(_ARABIC_INDIC_DIGITS)
    match = re.search(r"\d+", text)
    if not match:
        raise ValueError(f"No article number found in {raw!r}")
    remainder = text[match.end() :]
    suffix = normalize_article_suffix(remainder) if MUKARRAR in remainder else None
    return int(match.group()), suffix


@dataclass(frozen=True, slots=True)
class RetrievedArticle:
    """One retrieval hit: citation-ready, with faithful text and provenance."""

    chunk_id: str
    citation_label: str
    law_name: str  # canonical full name, e.g. «القانون المدني رقم ١٣١ لسنة ١٩٤٨»
    law_number: int
    law_year: int
    article_number: int | None  # None for مواد الإصدار (enacting provisions)
    # Full bis designation ("مكرر", "مكرر 1", "مكرر ا"), else None. Not a
    # boolean: same-number siblings must stay distinct (see normalize_article_suffix).
    article_suffix: str | None
    article_type: str
    article_status: str
    clean_text: str  # the ONLY text a citation may quote (== body_faithful)
    # Structural header from the payload ("قانون ... / الباب ... / مادة (ن)").
    # Carried for callers and debugging; NOT how articles reach the model --
    # they reach it as serialized ToolArticleResult fields (see rag/tools.py),
    # which is also why `article_status` is a field the system prompt reasons
    # about by name rather than a rendered warning line.
    header: str
    score: float | None  # None for exact-lookup hits (no ranking involved)


def _to_retrieved_article(payload: dict, score: float | None) -> RetrievedArticle:
    law_number = payload.get("law_number", 0)
    law_year = payload.get("law_year", 0)
    return RetrievedArticle(
        chunk_id=payload.get("chunk_id", ""),
        citation_label=payload.get("citation_label", ""),
        law_name=canonical_law_name(law_number, law_year, payload.get("law_name")),
        law_number=law_number,
        law_year=law_year,
        article_number=payload.get("article_number"),
        article_suffix=payload.get("article_suffix"),
        article_type=payload.get("article_type", ""),
        article_status=payload.get("article_status", "active"),
        clean_text=payload.get("body_faithful", ""),
        header=payload.get("header", ""),
        score=score,
    )


class Retriever:
    """Hybrid (dense + sparse + ColBERT rerank) + exact-lookup retriever."""

    def __init__(self, settings: Settings | None = None, trace_id: str | None = None) -> None:
        self._settings = settings or get_settings()
        self._cloud_client = get_cloud_client(self._settings)
        self._embedding_client = EmbeddingClient(self._settings)
        self._collection = self._settings.qdrant_collection_name
        # For observability spans only -- falls back to a fresh id if this
        # Retriever is used standalone (e.g. scripts) without a request trace.
        self._trace_id = trace_id or observability.new_trace_id()

    def search_articles(
        self,
        query_text: str,
        top_k: int = TOP_K_DEFAULT,
        candidate_k: int = RERANK_CANDIDATES_DEFAULT,
        law_number: int | None = None,
    ) -> list[RetrievedArticle]:
        """Conceptual retrieval: embed -> Qdrant hybrid (RRF) -> ColBERT rerank -> top_k.

        Use for topic/conceptual questions, not for "give me article N" queries
        (see get_article_by_number for those -- a bare number carries almost no
        semantic signal and hybrid search routinely misranks it).
        """
        with observability.start_span(
            name="search_articles",
            trace_id=self._trace_id,
            as_type="retriever",
            input={"query_text": query_text, "law_number": law_number},
            metadata={"top_k": top_k, "candidate_k": candidate_k},
        ) as span:
            embedded = self._embedding_client.embed([query_text])[0]

            query_filter = None
            if law_number is not None:
                law_match = models.FieldCondition(
                    key="law_number", match=models.MatchValue(value=law_number)
                )
                query_filter = models.Filter(must=[law_match])

            response = self._cloud_client.query_points(
                collection_name=self._collection,
                prefetch=[
                    models.Prefetch(
                        query=embedded.dense, using="dense", limit=candidate_k, filter=query_filter
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=embedded.sparse.indices, values=embedded.sparse.values
                        ),
                        using="sparse",
                        limit=candidate_k,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=candidate_k,
                with_payload=True,
            )
            candidates = response.points
            if not candidates:
                observability.safe_update(span, output={"result_count": 0})
                return []

            candidate_texts = [
                c.payload.get("text_for_display", c.payload.get("body_faithful", ""))
                for c in candidates
            ]
            rerank_started = time.monotonic()
            rerank_results = self._embedding_client.rerank(query_text, candidate_texts)
            rerank_latency_ms = (time.monotonic() - rerank_started) * 1000

            top = rerank_results[:top_k]
            results = [_to_retrieved_article(candidates[r.index].payload, r.score) for r in top]

            observability.safe_update(
                span,
                output={"result_count": len(results)},
                metadata={
                    "top_k": top_k,
                    "candidate_k": candidate_k,
                    "candidates_returned": len(candidates),
                    "rerank_latency_ms": round(rerank_latency_ms, 1),
                },
            )
            return results

    def get_article_by_number(
        self,
        article_number: int | str,
        law_number: int | None = None,
        limit: int = 50,
    ) -> list[RetrievedArticle]:
        """Exact retrieval by article number, no embedding/ranking involved.

        If `law_number` is None and the number exists in more than one law,
        ALL matches are returned -- never silently pick one (cross-law
        disambiguation). Accepts Arabic-Indic digits and preserves the مكرر
        distinction in full: "163" matches only the base article, "163 مكرر"
        only its bis article, and '129 مكرراً "5"' only that one sibling --
        never a different member of the same series. article_suffix is not
        indexed in Qdrant, so the distinction is applied in Python after the
        exact-number scroll below; `limit` is what has to be generous enough
        to survive that filter, since law 159/1981 puts eleven points on
        article_number 129 alone (base + مكرر + مكرر "١".."٩") before the
        other three laws contribute their own article 129.
        """
        number, suffix = normalize_article_number(article_number)

        with observability.start_span(
            name="get_article_by_number",
            trace_id=self._trace_id,
            as_type="retriever",
            input={"article_number": article_number, "law_number": law_number},
        ) as span:
            must: list[models.FieldCondition] = [
                models.FieldCondition(key="article_number", match=models.MatchValue(value=number))
            ]
            if law_number is not None:
                must.append(
                    models.FieldCondition(
                        key="law_number", match=models.MatchValue(value=law_number)
                    )
                )

            records, _ = self._cloud_client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(must=must),
                limit=limit,
                with_payload=True,
            )

            articles = [_to_retrieved_article(r.payload, score=None) for r in records]
            # A bis reference must match only that exact designation, and a
            # bare number only base points -- never collapse a series together.
            articles = [
                a for a in articles if normalize_article_suffix(a.article_suffix) == suffix
            ]
            observability.safe_update(span, output={"result_count": len(articles)})
            return articles
