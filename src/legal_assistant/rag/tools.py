"""LangChain tool-calling tools wrapping the retrieval module.

The tool descriptions are what the agent uses to choose correctly between
them, so they are written to be precise about when each applies:
`get_article_by_number` for a specific, numbered article; `search_articles`
for conceptual/topic questions with no explicit article number.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from legal_assistant.rag.retrieval import Retriever, RetrievedArticle


class ToolArticleResult(BaseModel):
    """One retrieved article, as returned to the agent."""

    law_name: str
    law_number: int
    law_year: int
    article_number: int | None
    article_type: str
    article_status: str
    citation_label: str
    clean_text: str


def _to_tool_result(article) -> ToolArticleResult:
    return ToolArticleResult(
        law_name=article.law_name,
        law_number=article.law_number,
        law_year=article.law_year,
        article_number=article.article_number,
        article_type=article.article_type,
        article_status=article.article_status,
        citation_label=article.citation_label,
        clean_text=article.clean_text,
    )


def build_retrieval_tools(
    retriever: Retriever | None = None,
    on_retrieve: Callable[[str, list[RetrievedArticle]], None] | None = None,
) -> list:
    """Build the two retrieval tools, bound to a shared Retriever instance.

    `on_retrieve(tool_name, articles)`, if given, fires synchronously every
    time a tool returns results -- this is how the caller (agent.py)
    accumulates the conversation's retrieved-context set without having to
    parse serialized ToolMessage JSON back out of the agent's message list.
    """
    retriever = retriever or Retriever()

    def _notify(tool_name: str, articles: list[RetrievedArticle]) -> None:
        if on_retrieve is not None:
            on_retrieve(tool_name, articles)

    @tool
    def get_article_by_number(
        article_number: int = Field(
            ..., description="The article number, e.g. 5 or 500. Accepts Arabic-Indic digits too."
        ),
        law_number: int | None = Field(
            default=None,
            description=(
                "174 for قانون الإجراءات الجنائية (Law 174/2025), or 131 for القانون المدني "
                "(Law 131/1948). Omit if the law is not specified or unclear in the question -- "
                "if the number exists in both laws, all matches are returned so the caller must "
                "never silently guess which one."
            ),
        ),
    ) -> list[ToolArticleResult]:
        """Fetch a SPECIFIC article by its exact number (e.g. "نص المادة 500 من قانون الإجراءات الجنائية؟",
        "المادة 5 من القانون المدني"). Use this whenever the question names an explicit article
        number, NOT for conceptual/topic questions -- a bare number carries almost no semantic
        signal, so semantic search routinely misranks these. Returns an empty list if the
        article number does not exist (including مواد الإصدار / enacting provisions, which have
        no article number and can only be found via search_articles)."""
        articles = retriever.get_article_by_number(article_number, law_number=law_number)
        _notify("get_article_by_number", articles)
        return [_to_tool_result(a) for a in articles]

    @tool
    def search_articles(
        query_text: str = Field(..., description="The legal question or topic, in Arabic."),
        law_number: int | None = Field(
            default=None,
            description="Restrict to one law (174 or 131) if the question clearly names it; omit otherwise.",
        ),
    ) -> list[ToolArticleResult]:
        """Conceptual/topic search over the two ingested laws (قانون الإجراءات الجنائية رقم ١٧٤
        لسنة ٢٠٢٥ and القانون المدني رقم ١٣١ لسنة ١٩٤٨) using hybrid semantic+lexical retrieval
        with rerank. Use this for questions about a legal concept, procedure, or topic where no
        specific article number is named (e.g. "ما هي شروط التصالح في الجنح؟"). This is also how
        to find مواد الإصدار (enacting provisions), which have no article number and are never
        returned by get_article_by_number."""
        articles = retriever.search_articles(query_text, law_number=law_number)
        _notify("search_articles", articles)
        return [_to_tool_result(a) for a in articles]

    return [get_article_by_number, search_articles]
