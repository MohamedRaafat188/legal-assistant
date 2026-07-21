"""Langfuse Cloud observability: tracing, guard-verdict scores, feedback scores.

Best-effort and non-blocking by design: every call in this module swallows
its own exceptions and logs a warning instead of raising. A Langfuse outage,
missing credentials, or SDK failure must never affect chat or feedback --
callers use this module without wrapping it in their own try/except.

Mental model: a trace is one /chat request; spans inside it are the
meaningful steps (tool choice, retrieval, rerank, generation, guard); scores
attached to a trace are judgments (the guard's own verdict, logged on every
request, and a lawyer's thumbs via /feedback).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from legal_assistant.config import Settings, get_settings

_log = logging.getLogger(__name__)

_client: Any = None
_client_init_attempted = False


class _NoopSpan:
    """Stand-in yielded by start_span when tracing is disabled/unavailable."""

    def update(self, **kwargs: Any) -> None:
        pass

    def score(self, **kwargs: Any) -> None:
        pass


_NOOP_SPAN = _NoopSpan()


def get_client(settings: Settings | None = None) -> Any:
    """Return the process-wide Langfuse client, or None if unconfigured/unavailable.

    Lazily initialized once; a failed init is remembered rather than retried
    on every call, so a misconfigured Langfuse doesn't add latency to every
    request.
    """
    global _client, _client_init_attempted
    if _client_init_attempted:
        return _client
    _client_init_attempted = True

    settings = settings or get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        _log.warning("Langfuse keys not configured; tracing disabled")
        return None

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )
    except Exception:
        _log.warning("Langfuse client init failed; tracing disabled", exc_info=True)
        _client = None
    return _client


def new_trace_id() -> str:
    """Generate a fresh trace id, usable even if tracing turns out to be disabled.

    Pre-generating the id (rather than reading it back from a started span)
    lets the caller surface it to the client immediately, before the trace is
    guaranteed to exist server-side, and reference it later from /feedback.
    """
    try:
        from langfuse import Langfuse

        return Langfuse.create_trace_id()
    except Exception:
        return uuid.uuid4().hex


@contextmanager
def start_span(
    name: str,
    trace_id: str,
    as_type: str = "span",
    input: Any = None,
    metadata: dict | None = None,
) -> Iterator[Any]:
    """Open a Langfuse span as the current OTel span, rooted at `trace_id`.

    Yields a span-like object with `.update(...)` / `.score(...)`; both are
    no-ops if Langfuse is disabled or the call fails, so callers never need
    their own try/except around this.
    """
    client = get_client()
    if client is None:
        yield _NOOP_SPAN
        return

    try:
        with client.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name=name,
            as_type=as_type,
            input=input,
            metadata=metadata,
        ) as span:
            yield span
    except Exception:
        _log.warning("Langfuse span '%s' failed; continuing untraced", name, exc_info=True)
        yield _NOOP_SPAN


@contextmanager
def session_scope(user_id: Any, session_id: Any) -> Iterator[None]:
    """Tag the trace and all its child spans with a Langfuse user_id/session_id.

    Must be entered *before* the trace's root span so the tags propagate to
    every child observation -- Langfuse's session/user views and aggregations
    only include observations carrying these attributes. No-ops if tracing is
    disabled or the SDK call fails.
    """
    client = get_client()
    if client is None:
        yield
        return

    try:
        from langfuse import propagate_attributes

        cm = propagate_attributes(
            user_id=str(user_id) if user_id is not None else None,
            session_id=str(session_id) if session_id is not None else None,
        )
    except Exception:
        _log.warning("Langfuse session/user propagation setup failed", exc_info=True)
        yield
        return

    with cm:
        yield


def safe_update(span: Any, **kwargs: Any) -> None:
    """Update a span, swallowing any Langfuse-side failure."""
    try:
        span.update(**kwargs)
    except Exception:
        _log.warning("Langfuse span update failed", exc_info=True)


def score_guard_verdict(trace_id: str, guard_result: Any, used_fallback: bool) -> None:
    """Log the citation guard's verdict as scores on the trace.

    This is the automated hallucination monitor -- no human needed:
    production hallucination rate, fallback rate, and inline-review
    incidence become filterable/aggregatable in Langfuse from day one.
    """
    client = get_client()
    if client is None:
        return
    try:
        client.create_score(
            trace_id=trace_id,
            name="hallucinated_citations_count",
            value=float(len(guard_result.hallucinated_citations)),
            data_type="NUMERIC",
        )
        client.create_score(
            trace_id=trace_id,
            name="citations_verified_count",
            value=float(len(guard_result.valid_citations)),
            data_type="NUMERIC",
        )
        client.create_score(
            trace_id=trace_id,
            name="inline_unverified_count",
            value=float(len(guard_result.unverified_inline_refs)),
            data_type="NUMERIC",
        )
        client.create_score(
            trace_id=trace_id,
            name="used_fallback",
            value=1.0 if used_fallback else 0.0,
            data_type="NUMERIC",
        )
    except Exception:
        _log.warning("Langfuse guard-verdict scoring failed", exc_info=True)


def score_feedback(trace_id: str, rating: int, comment: str | None) -> bool:
    """Attach a lawyer's thumbs-up/down as a `user_feedback` score on a trace.

    Returns True if the score call was attempted without raising. This is
    best-effort -- Langfuse Cloud does not synchronously confirm the trace
    exists, and this function never raises.
    """
    client = get_client()
    if client is None:
        return False
    try:
        client.create_score(
            trace_id=trace_id,
            name="user_feedback",
            value=float(rating),
            data_type="NUMERIC",
            comment=comment,
        )
        return True
    except Exception:
        _log.warning("Langfuse feedback scoring failed", exc_info=True)
        return False


def get_callback_handler() -> Any:
    """Return a LangChain callback handler bound to the current OTel span, or None.

    Call from inside an active `start_span(...)` block so the handler's
    spans (tool calls, LLM generation) nest under the chat_turn trace.
    Returns None if tracing is disabled/unavailable -- callers pass an empty
    callbacks list in that case.
    """
    client = get_client()
    if client is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:
        _log.warning("Langfuse callback handler init failed", exc_info=True)
        return None


def shutdown() -> None:
    """Flush and release the Langfuse client. Call on FastAPI shutdown."""
    global _client
    if _client is not None:
        try:
            _client.shutdown()
        except Exception:
            _log.warning("Langfuse shutdown failed", exc_info=True)
    _client = None
