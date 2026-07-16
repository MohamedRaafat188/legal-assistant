"""Agent orchestration: LangChain tool-calling agent + unavoidable citation guard.

The agent decides, per turn, whether and which tool to call. Every article
any tool returns (this turn or a prior one, including prior *sessions* --
see below) is accumulated into the conversation's AllowedSet. The agent's
final structured answer is always passed through the citation guard before
being returned -- no answer with an unverified citation ever reaches the
caller.

Cross-session citations (Option A): on load, every historical turn's
persisted `retrieved_context` is replayed into the AllowedSet *before* the
agent runs, so a follow-up today can cite an article retrieved weeks ago in
this same conversation. The running `summary` (if any) is folded into the
system prompt as prose continuity only -- it carries no citation guarantees;
those come entirely from the replayed `retrieved_context`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from legal_assistant import memory, observability
from legal_assistant.config import Settings, get_settings
from legal_assistant.db.models import Message as MessageRow
from legal_assistant.llm import get_llm
from legal_assistant.rag.citation_guard import (
    FALLBACK_MESSAGE_AR,
    AllowedSet,
    guarded_generate,
)
from legal_assistant.rag.prompts import CITATION_CONTRACT_AR, SYSTEM_PROMPT_AR
from legal_assistant.rag.retrieval import RetrievedArticle, Retriever
from legal_assistant.rag.tools import build_retrieval_tools

_log = logging.getLogger(__name__)


class _CitationOut(BaseModel):
    law_name: str
    article_number: int | None
    citation_label: str


class _AnswerFormat(BaseModel):
    """The JSON contract's schema, enforced structurally by the LLM provider."""

    answer_text: str
    citations: list[_CitationOut]


@dataclass
class TurnResult:
    """Everything the CLI (or a future API) needs to report about one turn."""

    answer_text: str
    verified_citations: list[dict]
    guard_report: str
    used_fallback: bool
    tool_calls: list[str] = field(default_factory=list)
    retrieved_article_ids: list[str] = field(default_factory=list)


def _messages_from_history(rows: list[MessageRow]) -> list[BaseMessage]:
    """Convert stored working-memory rows into plain prior-context LangChain messages."""
    out: list[BaseMessage] = []
    for row in rows:
        if row.role == "user":
            out.append(HumanMessage(content=row.content))
        else:
            out.append(AIMessage(content=row.content))
    return out


class LegalAssistantAgent:
    """A conversation with the tool-calling agent, guarded by the citation guard.

    Instances are conversation-scoped and persistence-backed: `create()`
    loads the conversation's full retrieved-context history into the
    AllowedSet and its recent turns + summary into the prompt, and every
    `ask()` call persists the new turn (user message, assistant answer,
    verified citations, this turn's retrieved_context) before returning.
    """

    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        settings: Settings | None = None,
        trace_id: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._user_id = user_id
        self._conversation_id = conversation_id
        # Always a real id, even if tracing turns out to be disabled -- lets
        # the caller (e.g. the API) surface/persist it unconditionally.
        self._trace_id = trace_id or observability.new_trace_id()
        self._allowed = AllowedSet()
        self._messages: list[BaseMessage] = []
        self._summary: str | None = None
        self._retrieved_this_turn: list[tuple[str, list[RetrievedArticle]]] = []

        retriever = Retriever(self._settings, trace_id=self._trace_id)
        self._tools = build_retrieval_tools(retriever, on_retrieve=self._on_retrieve)
        self._llm = get_llm(settings=self._settings)
        self._graph = self._build_graph()

    @classmethod
    async def create(
        cls,
        session: AsyncSession,
        user_id: int,
        conversation_id: int,
        settings: Settings | None = None,
        trace_id: str | None = None,
    ) -> LegalAssistantAgent:
        """Build an agent for an existing conversation, restoring cross-session state."""
        agent = cls(user_id, conversation_id, settings, trace_id=trace_id)
        await agent._load(session)
        return agent

    @property
    def trace_id(self) -> str:
        """The Langfuse trace id for this agent's turn(s) -- always a real id."""
        return self._trace_id

    async def _load(self, session: AsyncSession) -> None:
        loaded = await memory.load_conversation(session, self._conversation_id, self._user_id)
        working = await memory.get_working_memory(session, self._conversation_id, self._user_id)

        self._allowed = loaded.allowed
        self._summary = loaded.summary
        self._messages = _messages_from_history(working.recent_messages)
        self._graph = self._build_graph()

    def _build_graph(self):
        system_prompt = f"{SYSTEM_PROMPT_AR}\n\n{CITATION_CONTRACT_AR}"
        if self._summary:
            system_prompt += f"\n\nملخص المحادثة حتى الآن:\n{self._summary}"

        return create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=system_prompt,
            response_format=_AnswerFormat,
        )

    def _on_retrieve(self, tool_name: str, articles: list[RetrievedArticle]) -> None:
        self._retrieved_this_turn.append((tool_name, articles))
        self._allowed.add_many(articles)
        _log.info("tool_call name=%s retrieved=%d", tool_name, len(articles))

    def _invoke_graph(self, extra_instruction: str | None, callbacks: list | None = None) -> str:
        messages = list(self._messages)
        if extra_instruction:
            messages.append(HumanMessage(content=extra_instruction))

        config = {"callbacks": callbacks} if callbacks else {}
        state = self._graph.invoke({"messages": messages}, config=config)
        self._messages = state["messages"]

        structured = state.get("structured_response")
        if structured is not None:
            return structured.model_dump_json()

        # Fallback if the provider didn't honor response_format for some reason.
        last_ai_text = ""
        for m in reversed(state["messages"]):
            if getattr(m, "type", None) == "ai" and m.content:
                last_ai_text = m.content if isinstance(m.content, str) else str(m.content)
                break
        return last_ai_text

    async def ask(self, session: AsyncSession, user_text: str) -> TurnResult:
        """Run one conversation turn: agent decides tool use, guard verifies, turn is persisted."""
        self._retrieved_this_turn = []
        self._messages.append(HumanMessage(content=user_text))

        with observability.start_span(
            name="chat_turn",
            trace_id=self._trace_id,
            as_type="agent",
            input={"user_text": user_text},
            metadata={"conversation_id": self._conversation_id, "user_id": self._user_id},
        ) as span:
            handler = observability.get_callback_handler()
            callbacks = [handler] if handler else None

            with observability.start_span(
                name="citation_guard", trace_id=self._trace_id, as_type="guardrail"
            ) as guard_span:
                answer_json, guard_result = await asyncio.to_thread(
                    guarded_generate,
                    lambda extra: self._invoke_graph(extra, callbacks=callbacks),
                    self._allowed,
                )
                observability.safe_update(
                    guard_span,
                    output={"report": guard_result.report, "is_valid": guard_result.is_valid},
                )

            tool_calls = [name for name, _ in self._retrieved_this_turn]

            # De-dupe retrieved articles within the turn (a retry may re-call tools).
            seen: set[str] = set()
            turn_articles: list[RetrievedArticle] = []
            for _, arts in self._retrieved_this_turn:
                for a in arts:
                    if a.chunk_id not in seen:
                        seen.add(a.chunk_id)
                        turn_articles.append(a)
            retrieved_ids = [a.chunk_id for a in turn_articles]

            used_fallback = answer_json.get("answer_text") == FALLBACK_MESSAGE_AR

            verified_citations = (
                [
                    {
                        "law_name": c.law_name,
                        "article_number": c.article_number,
                        "citation_label": c.citation_label,
                    }
                    for c in guard_result.valid_citations
                ]
                if not used_fallback
                else []
            )

            answer_text = answer_json.get("answer_text", "")
            retrieved_context = [memory.article_to_context_dict(a) for a in turn_articles]

            observability.score_guard_verdict(self._trace_id, guard_result, used_fallback)
            observability.safe_update(
                span,
                output={"answer_text": answer_text},
                metadata={
                    "tool_calls": tool_calls,
                    "retrieved_article_ids": retrieved_ids,
                    "used_fallback": used_fallback,
                },
            )

        await memory.append_turn(
            session,
            self._conversation_id,
            self._user_id,
            user_text,
            answer_text,
            verified_citations,
            retrieved_context,
            trace_id=self._trace_id,
        )
        await memory.maybe_compact_summary(session, self._conversation_id, self._user_id)

        return TurnResult(
            answer_text=answer_text,
            verified_citations=verified_citations,
            guard_report=guard_result.report,
            used_fallback=used_fallback,
            tool_calls=tool_calls,
            retrieved_article_ids=retrieved_ids,
        )
