"""Expert agent router (Wave B: LangGraph agent + persistent chat sessions).

Endpoints (prefix ``/api/products``, tags ``expert``):

- ``POST /api/products/{product_id}/ask``
    Stream an expert-chat answer as Server-Sent Events (SSE) from the
    LangGraph agent (``run_agent_chat_stream`` → ``astream_events``).
    Optional ``session_id`` in the body continues a persistent session;
    when absent (and the product exists) a new session is created and its
    id is announced first thing in the stream via
    ``data: {"session_id": "<id>"}`` and mirrored in the ``X-Session-Id``
    response header. The user query, tool-call summaries, and the final
    assistant answer are persisted to the session transcript (best-effort;
    never breaks the stream).
- ``POST /api/products/{product_id}/ask/doc``
    Generate a self-contained Markdown document and return it as a
    downloadable file (``Content-Disposition: attachment``).
- ``GET /api/products/{product_id}/chat/sessions``
    List the product's chat sessions (newest first).
- ``GET /api/products/{product_id}/chat/sessions/{session_id}/messages``
    Return the stored transcript of a session.

SSE event contract (fixed; see ``api.agents.expert``):

    data: {"session_id": "<id>"}          (first frame for NEW sessions)
    data: {"status": "retrieving"|"thinking"|"answering"}
    data: {"reasoning": "<model thoughts>"}
    data: {"content": "<answer chunk>"}
    data: {"tool_call": {"name": "<tool>", "args": {...}}}
    data: {"tool_result": {"name": "<tool>", "content": "<summary>"}}
    data: {"error": "<message>"}
    data: [DONE]

All endpoints require an authenticated session (``get_current_user``). The
agent machinery lives in ``api.agents.expert``; this router only does request
parsing, SSE framing, session persistence, and file-response packaging.
``run_agent_chat_stream`` / ``run_agent_doc`` are imported as module
attributes so tests can monkeypatch them on this module.

Threading note: the async ``/ask`` handlers never touch the request-scoped
``db`` session (FastAPI runs sync dependencies on a worker thread while the
async handler + SSE generator run on the event loop — a request-scoped
SQLite session would break its same-thread rule the moment it is queried).
All chat-session DB work goes through short-lived sessions opened and closed
inside the handler/generator itself. The sync session-list endpoints use the
request-scoped session normally (same execution context).
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from api.agents.expert import run_agent_chat_stream, run_agent_doc
from api.expert.deep_research import run_deep_research_stream
from api.auth.deps import get_current_user
from api.db import get_db
from api.expert.types import (
    EVENT_CONTENT,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    ExpertStreamEvent,
)
from api.models import ChatMessageORM, ChatSessionORM, ProductORM, UserORM
from api.utils.rate_limit import enforce_user_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/products", tags=["expert"])

#: Max characters of the query used to title a new chat session.
_SESSION_TITLE_LIMIT = 120
#: Max characters of a tool summary persisted to the transcript.
_TOOL_ROW_LIMIT = 4000
#: Max characters accepted for one query / history message / transcript row.
#: Oversized bodies are rejected with 422 before any agent or DB work.
MAX_QUERY_CHARS = 32_000
#: Max client-provided history messages per request.
MAX_HISTORY_ITEMS = 50


def _sse(payload: Dict[str, Any]) -> str:
    """Format one SSE ``data:`` frame with a JSON payload."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _format_sse(event: ExpertStreamEvent) -> str:
    """Map an expert stream event to an SSE ``data:`` frame.

    Produces a typed frame (``{status}`` / ``{reasoning}`` / ``{content}`` /
    ``{tool_call}`` / ``{tool_result}`` / ``{error}``) keyed by the event
    type. ``tool_call`` / ``tool_result`` events carry a JSON payload in
    ``event.content`` which is re-serialized as a nested JSON object.
    """
    if event.type in (EVENT_TOOL_CALL, EVENT_TOOL_RESULT):
        try:
            payload = json.loads(event.content)
        except (ValueError, TypeError):
            payload = {"content": event.content}
        return _sse({event.type: payload})
    return _sse({event.type: event.content})


class ChatMessage(BaseModel):
    role: str = Field(..., max_length=16)
    content: str = Field(..., max_length=MAX_QUERY_CHARS)


class ExpertAskRequest(BaseModel):
    query: str = Field(..., max_length=MAX_QUERY_CHARS)
    messages: List[ChatMessage] = Field(default_factory=list, max_length=MAX_HISTORY_ITEMS)
    model: Optional[str] = None
    # Optional persistent-session continuation. When omitted, a new session
    # is created and its id is announced in the SSE stream (+ X-Session-Id).
    session_id: Optional[str] = Field(default=None, max_length=64)
    # Deep Research mode (Wave E): route the query through the
    # planner → researcher → synthesizer LangGraph flow (≤5 iterations)
    # instead of the regular react agent. Additive SSE statuses:
    # planning / researching / synthesizing; the final answer arrives as
    # regular content frames.
    deep_research: bool = False

    # Backward-compat: older clients may still send ``use_rlm``. The field is
    # accepted (and ignored) so those clients keep working after the RLM
    # removal; the engine choice is server-side now.
    model_config = {"extra": "ignore"}


def _safe_filename(product_id: str) -> str:
    """Build a safe attachment filename for the doc download."""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", product_id or "product") or "product"
    return f"productarium_{base}_expert.md"


def _new_id(prefix: str) -> str:
    """Generate a unique row id (``<prefix>_<hex ts><random>``).

    The random part carries 128 bits of entropy: session ids are bearer
    capability tokens for the chat endpoints, so they must not be guessable.
    """
    return f"{prefix}_{format(int(time.time()), 'x')}{secrets.token_hex(16)}"


@contextmanager
def _local_session() -> Iterator[Optional[Session]]:
    """Yield a short-lived DB session for streaming-phase persistence.

    Opened and closed inside the caller's execution context (never the
    request thread's session), so SQLite's same-thread rule holds. On any
    failure yields None — persistence is best-effort by design.
    """
    try:
        from api.db import SessionLocal

        session = SessionLocal()
    except Exception as e:  # pragma: no cover - import/wiring failure
        logger.warning("expert chat: could not open a DB session: %s", e)
        yield None
        return
    try:
        yield session
    finally:
        try:
            session.close()
        except Exception:  # pragma: no cover - defensive
            pass


def _session_owner_filter(user_id: Optional[str]):
    """Sessions visible to a user: their own plus unowned (``user_id IS NULL``)."""
    return or_(
        ChatSessionORM.user_id == user_id,
        ChatSessionORM.user_id.is_(None),
    )


def _get_session_or_404(
    product_id: str, session_id: str, db: Session, user_id: Optional[str] = None
) -> ChatSessionORM:
    """Fetch a chat session scoped to the product or raise 404.

    The product scoping is the authorization boundary for transcripts: one
    product can never read another product's conversations even with a valid
    session id. Sessions owned by a DIFFERENT user are invisible (404) —
    only the owner and unowned sessions resolve.
    """
    chat_session = (
        db.query(ChatSessionORM)
        .filter(
            ChatSessionORM.id == session_id,
            ChatSessionORM.product_id == product_id,
            _session_owner_filter(user_id),
        )
        .first()
    )
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return chat_session


def _continue_session(
    product_id: str, requested_id: str, user_id: Optional[str] = None
) -> str:
    """Validate the requested session id belongs to the product, or 404.

    Uses a short-lived local session (see the threading note in the module
docstring). Sessions owned by a different user cannot be continued. Any DB
failure is reported as "not found" — an unresolvable session cannot be
continued.
    """
    with _local_session() as session:
        if session is not None:
            try:
                found = (
                    session.query(ChatSessionORM)
                    .filter(
                        ChatSessionORM.id == requested_id,
                        ChatSessionORM.product_id == product_id,
                        _session_owner_filter(user_id),
                    )
                    .first()
                )
                if found is not None:
                    return found.id
            except Exception as e:
                logger.debug(
                    "chat session lookup failed for %r: %s", requested_id, e
                )
    raise HTTPException(status_code=404, detail="Chat session not found")


def _create_session(
    product_id: str, user: Optional[UserORM], query: str
) -> Optional[str]:
    """Create a new chat session for the product; None when not possible.

    Returns None (and streams statelessly) when the DB is unavailable, the
    product does not exist, or the insert fails — an ephemeral answer is
    better than a 500.
    """
    with _local_session() as session:
        if session is None:
            return None
        try:
            if session.get(ProductORM, product_id) is None:
                return None
            # Only link a user that actually exists in the DB (the
            # AUTH_PROVIDER=none system user and transient Keycloak fallback
            # users are not persisted; a dangling FK would make the INSERT
            # fail on Postgres).
            user_id: Optional[str] = None
            if user is not None and session.get(UserORM, user.id) is not None:
                user_id = user.id
            new_id = _new_id("chat")
            session.add(
                ChatSessionORM(
                    id=new_id,
                    product_id=product_id,
                    user_id=user_id,
                    title=query.strip()[:_SESSION_TITLE_LIMIT],
                )
            )
            session.commit()
            return new_id
        except Exception as e:
            logger.warning(
                "could not create chat session for product %s: %s", product_id, e
            )
            try:
                session.rollback()
            except Exception:  # pragma: no cover - defensive
                pass
            return None


def _append_message_rows(
    session: Session,
    session_id: str,
    rows: List[Dict[str, Any]],
) -> None:
    """Insert transcript rows and refresh the session's ``updated_at``.

    Each row gets an explicit strictly-increasing ``created_at`` (SQLite may
    store the same microsecond for the whole batch, which would make the
    transcript's ``ORDER BY created_at`` ordering unstable) and the parent
    session's ``updated_at`` is touched so the session list stays newest-first.
    """
    if not rows:
        return
    # Base timestamp: now, clamped strictly past the session's newest row so
    # a fast follow-up turn can never interleave with the previous turn's
    # +i-second offsets (SQLite often stores the same microsecond per batch).
    last = (
        session.query(func.max(ChatMessageORM.created_at))
        .filter(ChatMessageORM.session_id == session_id)
        .scalar()
    )
    base = datetime.utcnow()
    if last is not None and last >= base:
        base = last + timedelta(seconds=1)
    for i, row in enumerate(rows):
        session.add(
            ChatMessageORM(
                id=_new_id("msg"),
                session_id=session_id,
                role=row["role"],
                content=row["content"],
                tool_name=row.get("tool_name"),
                created_at=base + timedelta(seconds=i),
            )
        )
    sess = session.get(ChatSessionORM, session_id)
    if sess is not None:
        sess.updated_at = base + timedelta(seconds=len(rows))
    session.commit()


def _tool_event_row(event: ExpertStreamEvent) -> Dict[str, Any]:
    """Extract ``(tool_name, summary)`` from a tool_call/tool_result event."""
    try:
        payload = json.loads(event.content)
    except (ValueError, TypeError):
        payload = {"content": event.content}
    if not isinstance(payload, dict):
        payload = {"content": str(payload)}
    name = str(payload.get("name") or "tool")[:128]
    if event.type == EVENT_TOOL_CALL:
        summary = json.dumps(payload.get("args") or {}, ensure_ascii=False)
    else:
        summary = str(payload.get("content") or "")
    return {"role": "tool", "content": summary[:_TOOL_ROW_LIMIT], "tool_name": name}


def _transcript_rows(query: str, events: List[ExpertStreamEvent]) -> List[Dict[str, Any]]:
    """Build the transcript rows for one streamed turn.

    The user turn first, then one row per tool call/result, then the
    assembled assistant answer.
    """
    # Belt-and-braces cap: pydantic already bounds the query, but the
    # persisted row must never exceed the cap regardless of the input path.
    rows: List[Dict[str, Any]] = [
        {"role": "user", "content": query[:MAX_QUERY_CHARS]}
    ]
    answer_parts: List[str] = []
    for event in events:
        if event.type == EVENT_CONTENT:
            answer_parts.append(event.content)
        elif event.type in (EVENT_TOOL_CALL, EVENT_TOOL_RESULT):
            rows.append(_tool_event_row(event))
    if answer_parts:
        rows.append(
            {"role": "assistant", "content": "".join(answer_parts)[:MAX_QUERY_CHARS]}
        )
    return rows


async def _stream_and_persist(
    product_id: str,
    query: str,
    session_id: Optional[str],
    history: List[Dict[str, str]],
    model: Optional[str],
    seed_history: bool = False,
    persist: bool = True,
    deep_research: bool = False,
) -> AsyncIterator[str]:
    """Run the agent, yield SSE frames, persist the transcript at the end.

    Events are buffered in memory and written in one short DB transaction
    when the stream finishes (a chat turn is small: text deltas + tool
    summaries), so persistence never interleaves with streaming. When
    ``persist`` is False (ephemeral call for an unknown product) the
    transcript is skipped entirely.

    ``deep_research`` selects the planner → researcher → synthesizer runner
    (``run_deep_research_stream``) instead of the regular react agent; the
    SSE/transcript contracts are identical (the deep-research statuses are
    plain status events).
    """
    collected: List[ExpertStreamEvent] = []
    runner = run_deep_research_stream if deep_research else run_agent_chat_stream
    try:
        async for event in runner(
            product_id,
            query,
            session_id=session_id,
            history=history,
            model=model,
            seed_history=seed_history,
        ):
            if event is None:
                continue
            collected.append(event)
            yield _format_sse(event)
    except ValueError as e:
        # Controlled message from the expert layer — safe to surface.
        logger.warning("expert /ask stream failed: %s", e)
        yield _sse({"error": str(e)})
    except Exception as e:  # pragma: no cover - defensive over streaming
        # Unexpected exceptions may embed internal URLs/paths — generic frame,
        # full context in the server log only (review #5).
        logger.error("expert /ask stream failed: %s", e, exc_info=True)
        yield _sse({"error": "Expert agent stream failed; see server logs"})
    finally:
        if persist and session_id:
            try:
                with _local_session() as session:
                    if session is not None:
                        _append_message_rows(
                            session, session_id, _transcript_rows(query, collected)
                        )
            except Exception as e:  # pragma: no cover - best-effort
                logger.warning(
                    "expert chat transcript persistence failed (session %s): %s",
                    session_id,
                    e,
                )


@router.post("/{product_id}/ask")
async def expert_ask(
    product_id: str,
    body: ExpertAskRequest,
    user: UserORM = Depends(get_current_user),
):
    """Stream an expert-chat answer as SSE (LangGraph agent with tools).

    Requires login. When ``body.session_id`` is provided the conversation
    continues that persistent session (404 when it does not belong to the
    product). Otherwise, when the product exists, a new session row is
    created and its id is announced via ``data: {"session_id": ...}`` plus
    the ``X-Session-Id`` header; the transcript (query / tool summaries /
    answer) is persisted best-effort. For an unknown product the request
    still streams (legacy behavior: no 404) but runs statelessly with no
    session and no persistence.
    """
    # P1-17: per-user token bucket (30/min default) on the expert chat —
    # checked before any prompt/cognee work.
    enforce_user_rate_limit(
        user.id,
        setting_key="rate.expert.per_user_min",
        env_name="RATE_EXPERT_PER_USER_MINUTE",
        default_per_minute=30,
    )

    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    session_created = False
    session_id: Optional[str] = None
    persist = False
    if body.session_id:
        session_id = _continue_session(product_id, body.session_id, user.id)
        persist = True
    else:
        created = _create_session(product_id, user, body.query)
        if created is not None:
            session_id = created
            session_created = True
            persist = True

    history = [{"role": m.role, "content": m.content} for m in body.messages]

    async def event_stream():
        if session_created and session_id:
            yield _sse({"session_id": session_id})
        async for frame in _stream_and_persist(
            product_id, body.query, session_id, history, body.model,
            seed_history=session_created, persist=persist,
            deep_research=body.deep_research,
        ):
            yield frame
        yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if session_id:
        headers["X-Session-Id"] = session_id

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=headers,
    )


@router.post("/{product_id}/ask/doc")
async def expert_ask_doc(
    product_id: str,
    body: ExpertAskRequest,
    user: UserORM = Depends(get_current_user),
):
    """Generate a self-contained Markdown document and return it as a file.

    Requires login. Returns ``text/markdown`` with a ``Content-Disposition:
    attachment`` header. ``messages`` is accepted but ignored (doc generation
    is one-shot, not conversational).
    """
    # P1-17: same per-user bucket as /ask (doc generation is the same cost
    # class as a chat turn with deep research).
    enforce_user_rate_limit(
        user.id,
        setting_key="rate.expert.per_user_min",
        env_name="RATE_EXPERT_PER_USER_MINUTE",
        default_per_minute=30,
    )

    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    try:
        md = await run_agent_doc(product_id, body.query, body.model)
    except Exception as e:  # pragma: no cover - defensive over generation
        logger.error("expert /ask/doc failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Expert document generation failed"
        )

    if not md:
        md = (
            f"# Expert document for {product_id}\n\n"
            "_(No content was generated. Ensure the product has indexed knowledge "
            "or generated artifact docs, and that a local LLM is available.)_\n"
        )

    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={_safe_filename(product_id)}"
        },
    )


@router.get("/{product_id}/chat/sessions")
def list_chat_sessions(
    product_id: str,
    user: UserORM = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the product's chat sessions (newest first). Requires login.

    Only sessions owned by the caller plus unowned (``user_id IS NULL``)
    sessions are listed; another user's conversations stay private.
    """
    rows = (
        db.query(ChatSessionORM)
        .filter(
            ChatSessionORM.product_id == product_id,
            _session_owner_filter(user.id),
        )
        .order_by(ChatSessionORM.updated_at.desc(), ChatSessionORM.id)
        .all()
    )
    return [
        {
            "id": s.id,
            "product_id": s.product_id,
            "user_id": s.user_id,
            "title": s.title,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        }
        for s in rows
    ]


@router.get("/{product_id}/chat/sessions/{session_id}/messages")
def list_chat_messages(
    product_id: str,
    session_id: str,
    user: UserORM = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the stored transcript of a chat session. Requires login.

    The session must belong to the given product (404 otherwise) so one
    product can never read another product's conversations; sessions owned
    by a different user are equally invisible.
    """
    chat_session = _get_session_or_404(product_id, session_id, db, user.id)
    rows = (
        db.query(ChatMessageORM)
        .filter(ChatMessageORM.session_id == chat_session.id)
        .order_by(ChatMessageORM.created_at, ChatMessageORM.id)
        .all()
    )
    return [
        {
            "id": m.id,
            "session_id": m.session_id,
            "role": m.role,
            "content": m.content,
            "tool_name": m.tool_name,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in rows
    ]


__all__ = ["router"]
