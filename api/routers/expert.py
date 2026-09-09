"""Expert agent router (Wave B: LangGraph agent + persistent chat sessions).

Endpoints (prefix ``/api/products``, tags ``expert``):

- ``POST /api/products/{product_id}/ask``
    Start a DETACHED expert ask turn (``api.expert.turns.start_turn``) and
    stream it as Server-Sent Events. The generation is a background task
    keyed by ``turn_id`` — a client disconnect (navigation, proxy timeout)
    never cancels it; the answer is persisted to the session transcript
    regardless (issue #9). The first SSE frame announces the turn
    (``data: {"turn_id": "<id>"}``), followed by ``data: {"session_id":
    "<id>"}`` for NEW sessions (also mirrored in the ``X-Session-Id``
    header). A second ask into a session with a running turn → 409 with the
    running ``X-Turn-Id`` response header (the client may re-attach).
- ``GET /api/products/{product_id}/ask/stream/{turn_id}``
    Re-attach to a turn: replay the buffered events, then stream the live
    tail, terminating with ``[DONE]`` — same contract as ``/ask`` minus the
    turn/session announcement frames.
- ``POST /api/products/{product_id}/ask/{turn_id}/cancel``
    Explicitly stop a running turn (the UI Stop button); the partial answer
    is persisted to the transcript.
- ``POST /api/products/{product_id}/ask/doc``
    Generate a self-contained Markdown document and return it as a
    downloadable file (``Content-Disposition: attachment``).
- ``GET /api/products/{product_id}/chat/sessions``
    List the product's chat sessions (newest first).
- ``GET /api/products/{product_id}/chat/sessions/{session_id}/messages``
    Return the stored transcript of a session.
- ``GET /api/products/{product_id}/chat/sessions/{session_id}/active-turn``
    The RUNNING turn of a session (or ``null``) — the re-attach probe.
- ``DELETE /api/products/{product_id}/chat/sessions/{session_id}``
    Delete a session with its transcript; a running turn is cancelled.

SSE event contract (fixed; see ``api.agents.expert`` + ``api.expert.turns``):

    data: {"turn_id": "<id>"}             (first frame, /ask only)
    data: {"session_id": "<id>"}          (second frame for NEW sessions)
    data: {"status": "retrieving"|"thinking"|"answering"}
    data: {"reasoning": "<model thoughts>"}
    data: {"content": "<answer chunk>"}
    data: {"tool_call": {"name": "<tool>", "args": {...}}}
    data: {"tool_result": {"name": "<tool>", "content": "<summary>"}}
    data: {"error": "<message>"}
    data: [DONE]

Idle subscriber waits emit ``: ping`` SSE comments (heartbeats) so proxies
with idle timeouts (e.g. the Next.js rewrite proxy) do not reap the stream.

All endpoints require an authenticated session (``get_current_user``). The
agent machinery lives in ``api.agents.expert``; this router only does request
parsing, turn wiring, session persistence, and file-response packaging.
``run_agent_chat_stream`` / ``run_agent_doc`` are imported as module
attributes so tests can monkeypatch them on this module (the runner is
passed into ``start_turn`` BY VALUE at request time, so the seam holds).

Threading note: the async ``/ask`` handlers never touch the request-scoped
``db`` session (FastAPI runs sync dependencies on a worker thread while the
async handler + SSE generator run on the event loop — a request-scoped
SQLite session would break its same-thread rule the moment it is queried).
All chat-session DB work goes through short-lived sessions opened and closed
inside the handler/generator itself. The sync session-list endpoints use the
request-scoped session normally (same execution context).
"""

from __future__ import annotations

import logging
import re
import secrets
import time
from contextlib import contextmanager
from typing import Iterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from api.agents.expert import run_agent_chat_stream, run_agent_doc
from api.expert.deep_research import run_deep_research_stream
from api.auth.deps import get_current_user
from api.db import get_db
from api.expert.turns import (
    MAX_QUERY_CHARS,
    ActiveTurn,
    TurnBusyError,
    active_for_session,
    cancel_turn,
    cancel_turns_for_session,
    get_turn,
    sse_payload,
    start_turn,
    subscribe,
)
from api.models import ChatMessageORM, ChatSessionORM, ProductORM, UserORM
from api.utils.rate_limit import enforce_user_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/products", tags=["expert"])

#: Max characters of the query used to title a new chat session.
_SESSION_TITLE_LIMIT = 120
#: ``MAX_QUERY_CHARS`` (the query/history/transcript row cap) is re-exported
#: from ``api.expert.turns`` above — pydantic rejects oversized bodies with
#: 422 before any agent or DB work.
#: Max client-provided history messages per request.
MAX_HISTORY_ITEMS = 50


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


def _turn_or_404(turn_id: str, product_id: str, user_id: Optional[str]) -> ActiveTurn:
    """Resolve a turn for the attach/cancel endpoints or raise 404.

    The product scoping + owner check mirror the session endpoints: a turn
    of another product or user is indistinguishable from a nonexistent one.
    """
    turn = get_turn(turn_id)
    if (
        turn is None
        or turn.product_id != product_id
        or turn.user_id != user_id
    ):
        raise HTTPException(status_code=404, detail="Turn not found")
    return turn


@router.post("/{product_id}/ask")
async def expert_ask(
    product_id: str,
    body: ExpertAskRequest,
    user: UserORM = Depends(get_current_user),
):
    """Start a detached expert turn and stream it as SSE (issue #9).

    Requires login. The generation runs as a background task registered in
    ``api.expert.turns`` — a client disconnect (navigation, proxy timeout)
    no longer cancels it; the answer is persisted to the session transcript
    either way. The first SSE frame announces the ``turn_id`` (the re-attach
    handle, also valid for ``/ask/stream/{turn_id}`` and the cancel
    endpoint); a NEW session (no ``body.session_id``) is additionally
    announced via ``data: {"session_id": ...}`` + the ``X-Session-Id``
    header. A second ask into a session with a RUNNING turn → 409 with the
    running turn id in ``X-Turn-Id`` (the client re-attaches instead).
    For an unknown product the request still streams (legacy behavior: no
    404) but runs statelessly with no session and no persistence.
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
    # The runner is captured BY VALUE at request time so tests that
    # monkeypatch the module attribute keep steering the flow.
    runner = run_deep_research_stream if body.deep_research else run_agent_chat_stream

    try:
        turn = start_turn(
            product_id=product_id,
            query=body.query,
            session_id=session_id,
            user_id=user.id,
            history=history,
            model=body.model,
            seed_history=session_created,
            persist=persist,
            runner=runner,
        )
    except TurnBusyError as busy:
        raise HTTPException(
            status_code=409,
            detail="An expert answer is still generating for this chat",
            headers={"X-Turn-Id": busy.turn_id},
        ) from busy

    async def event_stream():
        # Frame contract: turn id first (the re-attach handle), then the
        # new-session id (if any), then the streamed events, then [DONE].
        yield sse_payload({"turn_id": turn.id})
        if session_created and session_id:
            yield sse_payload({"session_id": session_id})
        async for frame in subscribe(turn.id):
            yield frame

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


@router.get("/{product_id}/ask/stream/{turn_id}")
async def expert_ask_stream(
    product_id: str,
    turn_id: str,
    user: UserORM = Depends(get_current_user),
):
    """Re-attach to a turn: SSE replay of the buffered events + live tail.

    Same frame contract as ``/ask`` minus the turn/session announcements;
    terminates with ``data: [DONE]``. Works for a RUNNING turn (live tail)
    and a recently finished one (pure replay within the finished-turn TTL).
    Only the turn's owner may attach.
    """
    _turn_or_404(turn_id, product_id, user.id)

    async def event_stream():
        async for frame in subscribe(turn_id):
            yield frame

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{product_id}/ask/{turn_id}/cancel")
async def expert_ask_cancel(
    product_id: str,
    turn_id: str,
    user: UserORM = Depends(get_current_user),
):
    """Stop a running turn (the UI Stop button); returns the final status.

    The partial answer assembled so far is persisted to the transcript
    (unless the turn already finished — then this is a no-op returning its
    terminal status).
    """
    _turn_or_404(turn_id, product_id, user.id)
    status = await cancel_turn(turn_id)
    return {"turn_id": turn_id, "status": status}


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


@router.get("/{product_id}/chat/sessions/{session_id}/active-turn")
def get_chat_active_turn(
    product_id: str,
    session_id: str,
    user: UserORM = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The RUNNING turn of a chat session, or ``null`` (re-attach probe).

    The UI calls this when mounting / selecting a session: a non-null
    descriptor (with ``turn_id``) means an answer is still generating —
    the client attaches via ``GET /ask/stream/{turn_id}`` to catch the tail.
    """
    chat_session = _get_session_or_404(product_id, session_id, db, user.id)
    turn = active_for_session(chat_session.id)
    return turn.describe() if turn is not None else None


@router.delete("/{product_id}/chat/sessions/{session_id}")
async def delete_chat_session(
    product_id: str,
    session_id: str,
    user: UserORM = Depends(get_current_user),
):
    """Delete a chat session with its whole transcript. Requires login.

    A RUNNING turn of the session is cancelled first (with persistence
    disabled, so it can never write rows for a session that no longer
    exists). Message rows are deleted explicitly — the ORM relationship
    cascade is not trusted here because SQLite builds do not always enable
    foreign-key cascades. The LangGraph checkpoint thread is dropped
    best-effort via ``adelete_thread``.
    """
    # Same-thread rule: async handler → short-lived local DB session.
    with _local_session() as db:
        if db is None:
            raise HTTPException(status_code=503, detail="Chat history unavailable")
        chat_session = (
            db.query(ChatSessionORM)
            .filter(
                ChatSessionORM.id == session_id,
                ChatSessionORM.product_id == product_id,
                _session_owner_filter(user.id),
            )
            .first()
        )
        if chat_session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        # Cancel the running turn BEFORE deleting rows: the turn task would
        # otherwise write its (partial) transcript into a dead session.
        cancelled = cancel_turns_for_session(chat_session.id)
        db.query(ChatMessageORM).filter(
            ChatMessageORM.session_id == chat_session.id
        ).delete(synchronize_session=False)
        db.delete(chat_session)
        db.commit()

    # Best-effort LangGraph checkpoint thread cleanup (async checkpointer
    # API; any failure is harmless — checkpoints are keyed by session id and
    # an orphaned thread is unreachable garbage).
    try:
        from api.agents.runtime import get_checkpointer

        checkpointer = await get_checkpointer()
        deleter = getattr(checkpointer, "adelete_thread", None)
        if deleter is not None:
            await deleter(session_id)
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug("checkpoint thread cleanup failed (%s): %s", session_id, e)

    return {"deleted": session_id, "cancelled_turns": cancelled}


__all__ = ["router"]
