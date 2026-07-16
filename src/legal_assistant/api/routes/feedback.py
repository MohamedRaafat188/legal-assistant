"""POST /feedback -- attach a lawyer's thumbs-up/down to the Langfuse trace of one turn.

Feedback and monitoring are the same mechanism: both are scores on a trace.
This endpoint never blocks or breaks on a Langfuse failure -- it is a
separate, best-effort call, gated only by ownership (the trace_id must
belong to a turn in a conversation owned by the current user).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from legal_assistant import observability
from legal_assistant.api.deps import get_current_user_id, get_db_session
from legal_assistant.api.schemas import FeedbackRequest, FeedbackResponse
from legal_assistant.memory import find_owned_message_by_trace_id

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackResponse:
    message = await find_owned_message_by_trace_id(session, body.trace_id, user_id)
    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="لم يتم العثور على هذه المحادثة أو لا تخصك.",
        )

    observability.score_feedback(body.trace_id, body.rating, body.comment)
    return FeedbackResponse()
