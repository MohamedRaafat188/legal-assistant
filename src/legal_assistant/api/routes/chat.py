"""POST /chat -- SSE streaming chat, Option 3 (prose streams, citations withheld until verified).

SSE event protocol (each event is one `event: <name>` + one JSON `data:` line):

    event: token      {"text": "<prose chunk>"}       -- repeated, streamed prose
    event: citations  {"citations": [{"law_name", "article_number", "citation_label"}, ...]}
                                                       -- sent once, AFTER the guard verifies.
                                                          These are the only citations the client
                                                          may ever show.
    event: withdrawn  {"message": "<Arabic fallback>"} -- the guard hard-failed even after one
                                                          regeneration; no prose was streamed for
                                                          this turn. The client shows this message
                                                          in place of any answer.
    event: done       {"conversation_id": <id>, "trace_id": <str|null>}
                                                       -- the turn is persisted; safe to finalize UI.
                                                          trace_id (Langfuse) references this turn for
                                                          POST /feedback; null if tracing is disabled.
    event: error      {"message": "<Arabic user-safe error>"} -- a downstream failure (Qdrant,
                                                          the embedding service, Gemini, DB). No
                                                          turn is persisted when this fires.

Client contract: never render a citation before the `citations` event; on `withdrawn`, the turn
produced no valid answer this round (nothing was streamed to treat as final).

Design note (chunked delivery, not live token streaming): the agent's structured
`{answer_text, citations}` generation and the citation guard both need the complete answer to
run -- there is no partial-answer state that's meaningful to show. So a turn is run to completion
(unchanged `LegalAssistantAgent.ask`, already citation-guard-verified) and *then* its already-verified
`answer_text` is streamed to the client in small chunks for a responsive typing effect. This means
no unverified prose is ever sent, and a client disconnecting mid-stream loses nothing: the turn was
already fully persisted before the first chunk went out.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from legal_assistant.api.deps import get_current_user_id, get_db_session
from legal_assistant.api.schemas import ChatRequest
from legal_assistant.memory import ConversationNotFoundError
from legal_assistant.memory import load_conversation as memory_load_conversation
from legal_assistant.rag.agent import LegalAssistantAgent

router = APIRouter(tags=["chat"])
_log = logging.getLogger(__name__)

_CHUNK_SIZE_CHARS = 12
_CHUNK_DELAY_SECONDS = 0.02

_GRACEFUL_ERROR_AR = (
    "عذراً، حدث خطأ تقني أثناء معالجة سؤالك. يُرجى المحاولة مرة أخرى بعد قليل."
)


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


async def _chat_events(request: Request, session: AsyncSession, user_id: int, body: ChatRequest):
    try:
        try:
            agent = await LegalAssistantAgent.create(session, user_id, body.conversation_id)
            result = await agent.ask(session, body.message)
        except ConversationNotFoundError:
            yield _sse("error", {"message": "المحادثة المطلوبة غير موجودة أو لا تخصك."})
            return
        except Exception:  # noqa: BLE001 -- any downstream failure -> graceful event, never a crash
            _log.exception("chat turn failed: conversation_id=%s", body.conversation_id)
            yield _sse("error", {"message": _GRACEFUL_ERROR_AR})
            return

        if result.used_fallback:
            yield _sse("withdrawn", {"message": result.answer_text})
        else:
            text = result.answer_text
            for i in range(0, len(text), _CHUNK_SIZE_CHARS):
                if await request.is_disconnected():
                    _log.info(
                        "client disconnected mid-stream, conversation_id=%s", body.conversation_id
                    )
                    return
                yield _sse("token", {"text": text[i : i + _CHUNK_SIZE_CHARS]})
                await asyncio.sleep(_CHUNK_DELAY_SECONDS)
            yield _sse("citations", {"citations": result.verified_citations})

        trace_id = getattr(agent, "trace_id", None)
        yield _sse("done", {"conversation_id": body.conversation_id, "trace_id": trace_id})
    except asyncio.CancelledError:
        raise


@router.post("/chat")
async def chat(
    request: Request,
    body: ChatRequest,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> EventSourceResponse:
    # Authorize up front so a bad/foreign conversation_id 404s cleanly instead
    # of opening a stream that immediately errors.
    try:
        await memory_load_conversation(session, body.conversation_id, user_id)
    except ConversationNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    return EventSourceResponse(_chat_events(request, session, user_id, body))
