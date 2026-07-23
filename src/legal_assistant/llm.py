"""Gemini LLM factory."""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from legal_assistant.config import Settings, get_settings


def get_llm(
    streaming: bool = False, settings: Settings | None = None, model: str | None = None
) -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model configured for deterministic legal generation.

    temperature=0: citations and legal claims must be reproducible, not
    creative. Streaming is off by default; a later phase turns it on for the
    product API. `model` overrides `settings.llm_model`, e.g. for cheaper
    non-agent tasks like summary compaction.
    """
    settings = settings or get_settings()
    return ChatGoogleGenerativeAI(
        model=model or settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0,
        streaming=streaming,
    )
