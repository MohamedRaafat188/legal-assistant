"""Conversation endpoints: create/list/get, all scoped to the current user."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from legal_assistant.api.deps import get_current_user_id, get_db_session
from legal_assistant.api.schemas import (
    CitationOut,
    ConversationCreateRequest,
    ConversationDetailOut,
    ConversationOut,
    MessageOut,
)
from legal_assistant.memory import (
    ConversationNotFoundError,
    create_conversation,
    list_conversations,
)
from legal_assistant.memory import load_conversation as memory_load_conversation

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation_route(
    body: ConversationCreateRequest,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> ConversationOut:
    conv = await create_conversation(session, user_id, body.title)
    return ConversationOut(
        id=conv.id, title=conv.title, created_at=conv.created_at, updated_at=conv.updated_at
    )


@router.get("", response_model=list[ConversationOut])
async def list_conversations_route(
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> list[ConversationOut]:
    conversations = await list_conversations(session, user_id)
    return [
        ConversationOut(id=c.id, title=c.title, created_at=c.created_at, updated_at=c.updated_at)
        for c in conversations
    ]


@router.get("/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation_route(
    conversation_id: int,
    user_id: int = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db_session),
) -> ConversationDetailOut:
    try:
        loaded = await memory_load_conversation(session, conversation_id, user_id)
    except ConversationNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    return ConversationDetailOut(
        id=loaded.conversation.id,
        title=loaded.conversation.title,
        summary=loaded.summary,
        created_at=loaded.conversation.created_at,
        updated_at=loaded.conversation.updated_at,
        messages=[
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                citations=[CitationOut(**c) for c in m.citations] if m.citations else None,
                created_at=m.created_at,
            )
            for m in loaded.messages
        ],
    )
