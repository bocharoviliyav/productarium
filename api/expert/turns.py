"""Detached expert ask turns — background generation + SSE re-attach (issue #9).

Why this module exists: the expert ``/ask`` SSE used to run the agent INSIDE
the request lifecycle, so a client disconnect (navigating to another
activity, the 30-second Next.js proxy timeout, an unstable network) cancelled
the whole generation — the LLM server saw the in-flight request as
``cancelled`` and the user lost the answer.

One ask is now a DETACHED background task (``asyncio.create_task``) keyed by
``turn_id``:

- :func:`start_turn` persists the user row immediately (the question is
  visible in history even if the process dies mid-turn), spawns the task
  with a wall-clock budget (the ``expert_stream`` timeout key), and returns
  the turn handle.
- The task streams runner events into an append-only per-turn buffer and
  wakes every subscriber queue. When it finishes (success / error / cancel /
  timeout) it persists the transcript (tool rows + the assembled answer)
  and only THEN flips the status — so a ``completed`` turn always has its
  transcript in the DB (no read-after-completion race for re-attach).
- :func:`subscribe` is a pure view over the buffer: replay from the start,
  then a live tail with ``: ping`` SSE heartbeats (keeps proxies from
  reaping idle streams), then ``data: [DONE]``. Any number of subscribers
  may attach; a disconnecting subscriber only drops its own view — the
  generation itself is unaffected.
- At most ONE running turn per chat session (:class:`TurnBusyError` maps to
  HTTP 409 in the router; the running turn id rides the ``X-Turn-Id``
  header so the client can re-attach instead of failing).

The registry is in-process by design (single uvicorn worker — the same
pattern as the docgen job registry in ``api/docgen/jobs.py``); a process
restart abandons running turns, leaving the persisted question row behind.

Runners are passed in as callables by the router (never imported here) so
the router's monkeypatched test seams keep working.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Set

from sqlalchemy.orm import Session

from api.expert.types import (
    EVENT_CONTENT,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    ExpertStreamEvent,
)

logger = logging.getLogger(__name__)

#: Max characters of one query / transcript row (mirrors the router contract).
MAX_QUERY_CHARS = 32_000

#: Max characters of a tool summary persisted to the transcript.
_TOOL_ROW_LIMIT = 4000

#: SSE comment heartbeat interval while a subscriber waits for events.
HEARTBEAT_SECONDS = 15.0

#: How long finished turns stay addressable (late re-attach / replay).
_FINISHED_TTL_SECONDS = 900.0

#: Hard cap on finished turns kept in memory (oldest dropped first).
_MAX_FINISHED_TURNS = 50

#: Grace period for a cancelled task to run its persistence cleanup.
_CANCEL_WAIT_SECONDS = 10.0

_STATUS_RUNNING = "running"
_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_STATUS_CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# SSE framing helpers (single source; the router imports these)
# ---------------------------------------------------------------------------

def sse_payload(payload: Dict[str, Any]) -> str:
    """Format one SSE ``data:`` frame with a JSON payload."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def format_event_frame(event: ExpertStreamEvent) -> str:
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
        return sse_payload({event.type: payload})
    return sse_payload({event.type: event.content})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TurnBusyError(RuntimeError):
    """A turn is already running for the chat session."""

    def __init__(self, turn_id: str):
        super().__init__(f"turn {turn_id} is still running for this session")
        self.turn_id = turn_id


@dataclass
class ActiveTurn:
    """One detached ask turn: the background task + its event buffer."""

    id: str
    product_id: str
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    #: Query the runner sees (the raw query + inlined attachment blocks);
    #: None -> use ``query``. The transcript keeps the raw ``query``.
    runner_query: Optional[str] = None
    #: Transcript persistence enabled (needs a session row; False for
    #: ephemeral turns or when the session was deleted mid-turn).
    persist: bool = True
    status: str = _STATUS_RUNNING
    #: Terminal status decided by the runner outcome before the final flip.
    final_status: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    #: Append-only event buffer — the replay source for every subscriber.
    events: List[ExpertStreamEvent] = field(default_factory=list)
    #: Subscriber wake queues (items are wake signals, not the events).
    subscribers: Set["asyncio.Queue[None]"] = field(default_factory=set)
    task: Optional["asyncio.Task[None]"] = field(default=None, repr=False)

    @property
    def finished(self) -> bool:
        return self.status != _STATUS_RUNNING

    def append(self, event: ExpertStreamEvent) -> None:
        """Buffer one event and wake every subscriber (append-only)."""
        self.events.append(event)
        self.notify()

    def notify(self) -> None:
        """Wake all subscribers so they re-read the buffer / status."""
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(None)
            except Exception:  # pragma: no cover - defensive (unbounded queue)
                pass

    def describe(self) -> Dict[str, Any]:
        """JSON-safe descriptor for the active-turn / status endpoints."""
        return {
            "turn_id": self.id,
            "session_id": self.session_id,
            "product_id": self.product_id,
            "status": self.status,
            "query": self.query[:200],
            "started_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "finished_at": (
                datetime.fromtimestamp(self.finished_at).isoformat()
                if self.finished_at is not None
                else None
            ),
        }


#: turn_id -> ActiveTurn (running + recently finished).
_TURNS: Dict[str, ActiveTurn] = {}
#: session_id -> turn_id of the RUNNING turn (one per session).
_RUNNING_BY_SESSION: Dict[str, str] = {}


def get_turn(turn_id: str) -> Optional[ActiveTurn]:
    """Fetch a turn by id (running or within the finished TTL)."""
    return _TURNS.get(turn_id)


def active_for_session(session_id: str) -> Optional[ActiveTurn]:
    """The RUNNING turn of a chat session, if any (for re-attach probes)."""
    turn_id = _RUNNING_BY_SESSION.get(session_id)
    if turn_id is None:
        return None
    turn = _TURNS.get(turn_id)
    if turn is None or turn.finished:  # pragma: no cover - defensive
        _RUNNING_BY_SESSION.pop(session_id, None)
        return None
    return turn


def _prune() -> None:
    """Drop expired finished turns; cap the finished-turn footprint."""
    now = time.time()
    expired = [
        tid
        for tid, t in _TURNS.items()
        if t.finished
        and t.finished_at is not None
        and now - t.finished_at > _FINISHED_TTL_SECONDS
    ]
    for tid in expired:
        _TURNS.pop(tid, None)
    finished = sorted(
        (t for t in _TURNS.values() if t.finished),
        key=lambda t: t.finished_at or 0.0,
    )
    for turn in finished[: max(0, len(finished) - _MAX_FINISHED_TURNS)]:
        _TURNS.pop(turn.id, None)


# ---------------------------------------------------------------------------
# Transcript persistence (moved from the router; same semantics)
# ---------------------------------------------------------------------------

def _new_id(prefix: str) -> str:
    """Generate a unique row id (``<prefix>_<hex ts><random>``)."""
    return f"{prefix}_{format(int(time.time()), 'x')}{secrets.token_hex(16)}"


@contextmanager
def _local_session() -> Iterator[Optional[Session]]:
    """Yield a short-lived DB session for background persistence.

    Opened and closed inside the caller's execution context (the turn task
    runs on the event loop), so SQLite's same-thread rule holds. On any
    failure yields None — persistence is best-effort by design.
    """
    try:
        from api.db import SessionLocal

        session = SessionLocal()
    except Exception as e:  # pragma: no cover - import/wiring failure
        logger.warning("expert turns: could not open a DB session: %s", e)
        yield None
        return
    try:
        yield session
    finally:
        try:
            session.close()
        except Exception:  # pragma: no cover - defensive
            pass


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


def _final_transcript_rows(events: List[ExpertStreamEvent]) -> List[Dict[str, Any]]:
    """Build the END-of-turn transcript rows: tool rows + assembled answer.

    The user row is NOT included — it is persisted the moment the turn
    starts (see :func:`start_turn`).
    """
    rows: List[Dict[str, Any]] = []
    answer_parts: List[str] = []
    for event in events:
        if event.type == EVENT_CONTENT:
            answer_parts.append(event.content)
        elif event.type in (EVENT_TOOL_CALL, EVENT_TOOL_RESULT):
            rows.append(_tool_event_row(event))
        # error events are surfaced to subscribers but never persisted
    if answer_parts:
        rows.append(
            {"role": "assistant", "content": "".join(answer_parts)[:MAX_QUERY_CHARS]}
        )
    return rows


def _append_message_rows(
    session: Session, session_id: str, rows: List[Dict[str, Any]]
) -> List[str]:
    """Insert transcript rows and refresh the session's ``updated_at``;
    returns the created row ids in insertion order.

    Each row gets an explicit strictly-increasing ``created_at`` (SQLite may
    store the same microsecond for the whole batch, which would make the
    transcript's ``ORDER BY created_at`` ordering unstable) and the parent
    session's ``updated_at`` is touched so the session list stays newest-first.
    """
    from sqlalchemy import func

    from api.models import ChatMessageORM, ChatSessionORM

    if not rows:
        return []
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
    ids: List[str] = []
    for i, row in enumerate(rows):
        row_id = _new_id("msg")
        ids.append(row_id)
        session.add(
            ChatMessageORM(
                id=row_id,
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
    return ids


def _link_attachments(
    session_id: str, message_id: str, attachment_ids: List[str]
) -> None:
    """Best-effort: point attachment rows at their transcript message."""
    try:
        with _local_session() as session:
            if session is None:
                return
            from api.models import ChatAttachmentORM

            (
                session.query(ChatAttachmentORM)
                .filter(ChatAttachmentORM.id.in_(attachment_ids))
                .update(
                    {"message_id": message_id, "session_id": session_id},
                    synchronize_session=False,
                )
            )
            session.commit()
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug("attachment linking failed for %s: %s", message_id, e)


def _persist_rows(session_id: str, rows: List[Dict[str, Any]]) -> List[str]:
    """Best-effort transcript row insert (never raises); returns row ids."""
    if not rows:
        return []
    try:
        with _local_session() as session:
            if session is not None:
                return _append_message_rows(session, session_id, rows)
    except Exception as e:  # pragma: no cover - best-effort
        logger.warning(
            "expert chat transcript persistence failed (session %s): %s",
            session_id,
            e,
        )
    return []


# ---------------------------------------------------------------------------
# Turn lifecycle
# ---------------------------------------------------------------------------

async def _consume(
    turn: ActiveTurn,
    runner: Callable[..., Any],
    runner_kwargs: Dict[str, Any],
) -> None:
    """Drive the runner, buffering every event for the subscribers."""
    async for event in runner(
        turn.product_id, turn.runner_query or turn.query, **runner_kwargs
    ):
        if event is None:
            continue
        turn.append(event)
        # Give subscribers a chance to flush between events.
        await asyncio.sleep(0)


async def _run_turn_bounded(
    turn: ActiveTurn,
    runner: Callable[..., Any],
    runner_kwargs: Dict[str, Any],
    budget: float,
) -> None:
    """Run one turn with a wall-clock budget; persist + finalize in ``finally``.

    Outcome mapping mirrors the old router stream:

    - ``ValueError`` — controlled expert-layer message, surfaced verbatim;
    - anything else — generic error frame, full context in the server log;
    - budget expiry — timeout error frame, partial answer persisted;
    - cancellation (Stop button / session delete) — no error frame, partial
      answer persisted unless the session was deleted (``persist`` flipped).
    """
    from api.expert.types import EVENT_ERROR

    try:
        await asyncio.wait_for(_consume(turn, runner, runner_kwargs), timeout=budget)
    except asyncio.TimeoutError:
        logger.warning(
            "expert turn %s exceeded the %.0fs budget", turn.id, budget
        )
        turn.append(
            ExpertStreamEvent(
                EVENT_ERROR,
                "Expert answer exceeded the time budget (timeouts.expert_stream); "
                "the partial answer was saved",
            )
        )
        turn.final_status = _STATUS_FAILED
    except asyncio.CancelledError:
        turn.final_status = _STATUS_CANCELLED
    except ValueError as e:
        # Controlled message from the expert layer — safe to surface.
        logger.warning("expert turn %s failed: %s", turn.id, e)
        turn.append(ExpertStreamEvent(EVENT_ERROR, str(e)))
        turn.final_status = _STATUS_FAILED
    except Exception as e:  # pragma: no cover - defensive over streaming
        # Unexpected exceptions may embed internal URLs/paths — generic frame,
        # full context in the server log only.
        logger.error("expert turn %s failed: %s", turn.id, e, exc_info=True)
        turn.append(
            ExpertStreamEvent(
            EVENT_ERROR, "Expert agent stream failed; see server logs"
            )
        )
        turn.final_status = _STATUS_FAILED
    finally:
        if turn.persist and turn.session_id:
            _persist_rows(turn.session_id, _final_transcript_rows(turn.events))
        # Status flips only AFTER the transcript is durable, so a subscriber
        # that sees "completed" can safely read the answer from history.
        turn.status = turn.final_status or _STATUS_COMPLETED
        turn.finished_at = time.time()
        if turn.session_id and _RUNNING_BY_SESSION.get(turn.session_id) == turn.id:
            _RUNNING_BY_SESSION.pop(turn.session_id, None)
        _prune()
        turn.notify()


def start_turn(
    *,
    product_id: str,
    query: str,
    session_id: Optional[str],
    user_id: Optional[str],
    history: List[Dict[str, str]],
    model: Optional[str],
    seed_history: bool = False,
    persist: bool = True,
    runner: Callable[..., Any],
    runner_query: Optional[str] = None,
    attachment_ids: Optional[List[str]] = None,
) -> ActiveTurn:
    """Start a detached ask turn; returns immediately with the handle.

    The user row is persisted right away (best-effort) so the question is in
    the session history even if the process dies mid-turn; when
    ``attachment_ids`` are given, they are linked to that row. Raises
    :class:`TurnBusyError` when the session already has a running turn.
    """
    _prune()
    if session_id:
        running = active_for_session(session_id)
        if running is not None:
            raise TurnBusyError(running.id)

    turn = ActiveTurn(
        id=_new_id("turn"),
        product_id=product_id,
        query=query,
        session_id=session_id,
        user_id=user_id,
        persist=persist,
        runner_query=runner_query,
    )
    if turn.persist and session_id:
        created_ids = _persist_rows(
            session_id, [{"role": "user", "content": query[:MAX_QUERY_CHARS]}]
        )
        if attachment_ids and created_ids:
            _link_attachments(session_id, created_ids[0], attachment_ids)

    runner_kwargs: Dict[str, Any] = {
        "session_id": session_id,
        "history": history,
        "model": model,
        "seed_history": seed_history,
    }
    try:
        from api.config.timeout import resolve_expert_stream_timeout

        budget = resolve_expert_stream_timeout()
    except Exception:  # pragma: no cover - defensive
        budget = 1800.0

    loop = asyncio.get_running_loop()
    turn.task = loop.create_task(
        _run_turn_bounded(turn, runner, runner_kwargs, budget)
    )
    _TURNS[turn.id] = turn
    if session_id:
        _RUNNING_BY_SESSION[session_id] = turn.id
    return turn


async def subscribe(turn_id: str) -> AsyncIterator[str]:
    """Replay + live SSE frames for one turn; terminates with ``[DONE]``.

    Race-free by construction: after every wake the whole append-only buffer
    is re-read from the subscriber's cursor, and the finished check happens
    only after the buffer is drained — so no event can be missed or doubled
    between replay and tail. Idle waits emit ``: ping`` SSE comments
    (ignored by browsers and the frontend parser) to defeat idle-connection
    proxy timeouts.
    """
    turn = _TURNS.get(turn_id)
    if turn is None:
        yield sse_payload({"error": "Turn not found"})
        yield "data: [DONE]\n\n"
        return
    queue: "asyncio.Queue[None]" = asyncio.Queue()
    turn.subscribers.add(queue)
    try:
        cursor = 0
        while True:
            events = turn.events
            while cursor < len(events):
                yield format_event_frame(events[cursor])
                cursor += 1
            if turn.finished:
                break
            try:
                await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
    finally:
        turn.subscribers.discard(queue)
    yield "data: [DONE]\n\n"


async def cancel_turn(turn_id: str) -> str:
    """Cancel a running turn (Stop button); awaits the cleanup, returns status."""
    turn = _TURNS.get(turn_id)
    if turn is None:
        raise KeyError(turn_id)
    if not turn.finished and turn.task is not None:
        turn.task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.shield(turn.task), timeout=_CANCEL_WAIT_SECONDS
            )
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            # The cleanup (partial-answer persistence) is best-effort; a
            # hung task must never block the cancel endpoint.
            pass
    return turn.status


def cancel_turns_for_session(session_id: str) -> List[str]:
    """Cancel + forget every running turn of a session (session delete).

    Persistence is disabled BEFORE cancelling so a finishing turn can never
    write transcript rows for a session that no longer exists.
    """
    cancelled: List[str] = []
    running = active_for_session(session_id)
    if running is None:
        return cancelled
    running.persist = False
    cancelled.append(running.id)
    if running.task is not None:
        running.task.cancel()
    _RUNNING_BY_SESSION.pop(session_id, None)
    _TURNS.pop(running.id, None)
    return cancelled


__all__ = [
    "MAX_QUERY_CHARS",
    "HEARTBEAT_SECONDS",
    "ActiveTurn",
    "TurnBusyError",
    "active_for_session",
    "cancel_turn",
    "cancel_turns_for_session",
    "format_event_frame",
    "get_turn",
    "sse_payload",
    "start_turn",
    "subscribe",
]
