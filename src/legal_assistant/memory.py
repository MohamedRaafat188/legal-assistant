"""Conversation store, working memory, and cross-session AllowedSet rebuild.

Option A (locked): every assistant turn persists its `retrieved_context`
(the articles actually retrieved that turn). Loading a conversation replays
every stored `retrieved_context` into a fresh AllowedSet *before* the agent
runs, so a follow-up today can cite an article retrieved weeks ago and every
historical citation stays mechanically verifiable. The running `summary` is
prose continuity only -- it carries no citation guarantees; those live
entirely in `retrieved_context`, so the summary can compress freely.
"""

from __future__ import annotations

import asyncio
import datetime
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from legal_assistant.config import get_settings
from legal_assistant.db.models import Conversation, Message
from legal_assistant.llm import get_llm
from legal_assistant.rag.citation_guard import AllowedSet, normalize_law_name
from legal_assistant.rag.retrieval import RetrievedArticle

# Turns (user+assistant pairs) kept verbatim as "working memory" fed to the agent.
WORKING_MEMORY_TURNS = 6
# Total turns beyond which older-than-working-memory turns get folded into `summary`.
COMPACTION_THRESHOLD_TURNS = 12


class ConversationNotFoundError(Exception):
    """Raised when a conversation doesn't exist or doesn't belong to the requesting user."""


def article_to_context_dict(article: RetrievedArticle) -> dict:
    """Serialize a retrieved article into what's persisted in `retrieved_context`.

    Keeps enough to both rebuild the AllowedSet and re-display the article
    (a faithful clean_text snapshot) without re-fetching from Qdrant.
    """
    return {
        "point_id": article.chunk_id,
        "law_number": article.law_number,
        "law_name": article.law_name,
        "article_number": article.article_number,
        "citation_label": article.citation_label,
        "clean_text": article.clean_text,
    }


def _add_context_dict_to_allowed(allowed: AllowedSet, ctx: dict) -> None:
    citation_label = ctx.get("citation_label", "")
    article_number = ctx.get("article_number")
    allowed.all_labels.add(citation_label)
    if article_number is None:
        allowed.unnumbered_labels.add(citation_label)
    else:
        allowed.numbered.add((normalize_law_name(ctx.get("law_name", "")), article_number, False))


@dataclass
class LoadedConversation:
    """Everything needed to resume a conversation: history, summary, AllowedSet."""

    conversation: Conversation
    messages: list[Message]
    summary: str | None
    allowed: AllowedSet = field(default_factory=AllowedSet)


@dataclass
class WorkingMemory:
    """What's fed to the agent as prior context: recent verbatim turns + running summary."""

    recent_messages: list[Message]
    summary: str | None


async def create_conversation(session: AsyncSession, user_id: int, title: str) -> Conversation:
    """Create a new, empty conversation owned by `user_id`."""
    conversation = Conversation(user_id=user_id, title=title)
    session.add(conversation)
    await session.flush()
    return conversation


async def list_conversations(session: AsyncSession, user_id: int) -> list[Conversation]:
    """List conversations belonging to `user_id`, most recently updated first."""
    result = await session.scalars(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(result.all())


async def _get_owned_conversation(session: AsyncSession, conversation_id: int, user_id: int) -> Conversation:
    conversation = await session.scalar(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    if conversation is None or conversation.user_id != user_id:
        raise ConversationNotFoundError(
            f"conversation {conversation_id} not found for user {user_id}"
        )
    return conversation


async def append_turn(
    session: AsyncSession,
    conversation_id: int,
    user_id: int,
    user_content: str,
    assistant_content: str,
    citations: list[dict],
    retrieved_context: list[dict],
    trace_id: str | None = None,
) -> None:
    """Atomically persist one turn's user + assistant messages and touch `updated_at`."""
    conversation = await _get_owned_conversation(session, conversation_id, user_id)

    session.add(Message(conversation_id=conversation_id, role="user", content=user_content))
    session.add(
        Message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_content,
            citations=citations or None,
            retrieved_context=retrieved_context or None,
            trace_id=trace_id,
        )
    )
    conversation.updated_at = datetime.datetime.now(datetime.UTC)
    await session.flush()


async def find_owned_message_by_trace_id(
    session: AsyncSession, trace_id: str, user_id: int
) -> Message | None:
    """Return the assistant message for `trace_id` if it belongs to `user_id`, else None.

    Used by POST /feedback to enforce the same ownership isolation as the rest
    of the API: a lawyer can only score turns from their own conversations.
    """
    result = await session.scalar(
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Message.trace_id == trace_id, Conversation.user_id == user_id)
    )
    return result


async def load_conversation(session: AsyncSession, conversation_id: int, user_id: int) -> LoadedConversation:
    """Load full ordered history and rebuild the AllowedSet from every stored retrieved_context."""
    conversation = await _get_owned_conversation(session, conversation_id, user_id)
    messages = list(conversation.messages)

    allowed = AllowedSet()
    for m in messages:
        if m.role == "assistant" and m.retrieved_context:
            for ctx in m.retrieved_context:
                _add_context_dict_to_allowed(allowed, ctx)

    return LoadedConversation(
        conversation=conversation, messages=messages, summary=conversation.summary, allowed=allowed
    )


async def get_working_memory(
    session: AsyncSession, conversation_id: int, user_id: int
) -> WorkingMemory:
    """Return every turn not yet folded into `summary`, verbatim, plus the summary itself.

    This window isn't a fixed size: it grows from WORKING_MEMORY_TURNS up to
    2 * WORKING_MEMORY_TURNS turns between compactions (see
    maybe_compact_summary), then drops back down once the next batch is
    folded in. Turns before `summarized_through_turn` are covered by
    `summary` and are deliberately not re-sent here.
    """
    conversation = await _get_owned_conversation(session, conversation_id, user_id)
    messages = list(conversation.messages)
    recent = messages[2 * conversation.summarized_through_turn :]
    return WorkingMemory(recent_messages=recent, summary=conversation.summary)


_SUMMARY_PROMPT_AR = """\
لخّص المحادثة القانونية التالية بإيجاز ودقة باللغة العربية الفصحى، مع الحفاظ \
على استمرارية السياق (ما الذي سُئل وما الذي أُجيب عنه من حيث الموضوع \
والمفاهيم). لا داعي لذكر أرقام المواد أو تفاصيلها الدقيقة -- الاستشهادات \
القانونية محفوظة بشكل منفصل ولا تعتمد على هذا الملخص.

الملخص السابق (إن وجد):
{previous_summary}

المحادثة الإضافية التي يجب دمجها في الملخص:
{turns_text}

اكتب الملخص المحدَّث فقط، دون أي تعليق إضافي.
"""


def _extract_text(content: str | list) -> str:
    """Flatten a LangChain message's content into plain text.

    Gemini responses sometimes come back as a list of content parts (e.g.
    `[{"type": "text", "text": "..."}]`) instead of a bare string.
    """
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)


def _render_turns(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        speaker = "المستخدم" if m.role == "user" else "المساعد"
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)


async def maybe_compact_summary(session: AsyncSession, conversation_id: int, user_id: int) -> bool:
    """Fold the next full batch of WORKING_MEMORY_TURNS turns into `summary`, once enough have piled up.

    Fires once total turns reach COMPACTION_THRESHOLD_TURNS, and again every
    WORKING_MEMORY_TURNS turns after that (e.g. at 12, 18, 24, ...). Each
    firing folds in exactly the batch of turns between
    `summarized_through_turn` and `summarized_through_turn + WORKING_MEMORY_TURNS`
    -- never the whole older history -- so summarization cost stays
    constant per call instead of growing with the conversation.

    Not on the response critical path -- call this after responding to the
    user, or lazily on load. Returns True if a new summary was written.
    """
    conversation = await _get_owned_conversation(session, conversation_id, user_id)
    messages = list(conversation.messages)
    total_turns = sum(1 for m in messages if m.role == "user")
    summarized_through = conversation.summarized_through_turn

    if total_turns < COMPACTION_THRESHOLD_TURNS:
        return False
    if total_turns - summarized_through < 2 * WORKING_MEMORY_TURNS:
        return False

    batch_start_turn = summarized_through
    batch_end_turn = summarized_through + WORKING_MEMORY_TURNS
    batch = messages[2 * batch_start_turn : 2 * batch_end_turn]
    if not batch:
        return False

    prompt = _SUMMARY_PROMPT_AR.format(
        previous_summary=conversation.summary or "(لا يوجد)", turns_text=_render_turns(batch)
    )
    llm = get_llm(model=get_settings().summary_llm_model)
    response = await asyncio.to_thread(llm.invoke, prompt)
    new_summary = _extract_text(response.content)

    conversation.summary = new_summary.strip()
    conversation.summarized_through_turn = batch_end_turn
    await session.flush()
    return True
