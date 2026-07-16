"""Persisted multi-turn CLI for the legal assistant agent.

Login/register, list/select/create a conversation, then chat with full
Postgres-backed persistence: every turn's messages, verified citations, and
retrieved_context are saved, and citations stay verifiable across sessions
(Option A -- see legal_assistant.memory / legal_assistant.rag.agent).

Usage:
    python scripts/ask.py
    python scripts/ask.py --once "ما هو نص المادة 500 من قانون الإجراءات الجنائية؟" \\
        --username lawyer1 --password ... --conversation-id 1
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_assistant.auth import InvalidCredentialsError, UsernameTakenError, authenticate, register
from legal_assistant.db.models import Conversation
from legal_assistant.db.session import session_scope
from legal_assistant.memory import create_conversation, list_conversations
from legal_assistant.rag.agent import LegalAssistantAgent, TurnResult


def print_turn(result: TurnResult) -> None:
    print(f"\nTools called: {result.tool_calls or '(none)'}")
    if result.retrieved_article_ids:
        print(f"Retrieved article IDs: {result.retrieved_article_ids}")
    print(f"\nAnswer:\n{result.answer_text}")
    if result.verified_citations:
        print("\nVerified citations:")
        for c in result.verified_citations:
            print(f"  - {c['citation_label']}")
    else:
        print("\nVerified citations: (none)")
    if result.used_fallback:
        print("\n[GUARD] Fallback returned -- citation could not be verified even after regeneration.")
    print(f"\nGuard report:\n{result.guard_report}")
    print("-" * 70)


async def login_or_register(username: str, password: str, do_register: bool) -> int:
    async with session_scope() as session:
        if do_register:
            user = await register(session, username, password)
            print(f"Registered user {username!r} (id={user.id}).")
        else:
            user = await authenticate(session, username, password)
            print(f"Logged in as {username!r} (id={user.id}).")
        return user.id


async def pick_conversation(user_id: int, new_title: str | None) -> int:
    async with session_scope() as session:
        if new_title:
            conv = await create_conversation(session, user_id, new_title)
            print(f"Created conversation {conv.id!r}: {new_title!r}")
            return conv.id

        conversations: list[Conversation] = await list_conversations(session, user_id)
        if not conversations:
            conv = await create_conversation(session, user_id, "محادثة جديدة")
            print(f"No existing conversations -- created {conv.id!r}.")
            return conv.id

        print("\nYour conversations:")
        for c in conversations:
            print(f"  [{c.id}] {c.title} (updated {c.updated_at})")
        raw = input("Enter a conversation id to resume, or 'new' to start one: ").strip()
        if raw.lower() == "new":
            title = input("Title for the new conversation: ").strip() or "محادثة جديدة"
            conv = await create_conversation(session, user_id, title)
            return conv.id
        return int(raw)


async def run_once(user_id: int, conversation_id: int, question: str) -> None:
    async with session_scope() as session:
        agent = await LegalAssistantAgent.create(session, user_id, conversation_id)
        result = await agent.ask(session, question)
        print_turn(result)


async def run_repl(user_id: int, conversation_id: int) -> None:
    print("Legal Assistant CLI. Type your question in Arabic, or 'exit' to quit.")
    async with session_scope() as session:
        agent = await LegalAssistantAgent.create(session, user_id, conversation_id)
        while True:
            try:
                user_text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                break
            result = await agent.ask(session, user_text)
            print_turn(result)


async def main_async(args: argparse.Namespace) -> None:
    try:
        user_id = await login_or_register(args.username, args.password, args.register)
    except (InvalidCredentialsError, UsernameTakenError) as e:
        print(f"Auth failed: {e}")
        return

    if args.conversation_id is not None:
        conversation_id = args.conversation_id
    else:
        conversation_id = await pick_conversation(user_id, args.new_conversation)

    if args.once:
        await run_once(user_id, conversation_id, args.once)
    else:
        await run_repl(user_id, conversation_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", default=None, help="Ask a single question and exit (non-interactive).")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--register", action="store_true", help="Register a new user instead of logging in.")
    parser.add_argument("--conversation-id", type=int, default=None, help="Resume this conversation id directly.")
    parser.add_argument("--new-conversation", default=None, help="Create a new conversation with this title.")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
