"""Integration tests for the Wave B expert chat router (SSE + sessions).

Covers the fixed SSE contract over the LangGraph agent stream:
- event sequence: the ``turn_id`` announcement FIRST, then for NEW sessions
  the ``session_id`` announcement, then ``status: retrieving`` / ``tool_call``
  / ``tool_result`` frames, ``status: thinking`` + ``reasoning``, ``status:
  answering`` + ``content`` deltas, terminating ``data: [DONE]``;
- session continuation via ``session_id`` (no second announcement);
- transcript persistence across two turns (query / tool rows / answer);
- ``GET /chat/sessions`` and ``GET /chat/sessions/{id}/messages`` (CRUD,
  404 on another product's session, 401 without auth);
- error frame when the agent stream raises; [DONE] always terminates;
- DETACHED TURNS (issue #9): the generation runs in a background task —
  it survives a client disconnect, supports SSE re-attach replay, explicit
  cancel (partial answer persisted), one-running-turn-per-session 409 with
  ``X-Turn-Id``, heartbeat ``: ping`` comments, and session DELETE (cancels
  the running turn, wipes the transcript).

The agent stream is faked via monkeypatching ``run_agent_chat_stream`` on
the router module (the documented test seam), so no LLM is required.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.models import ChatSessionORM, ProductORM, UserORM  # noqa: E402
from tests.conftest import build_test_client  # noqa: E402


def _seed_product(db_mod, product_id: str = "prod_1") -> None:
    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id=product_id, name="Acme"))
        db.commit()


def _frames(resp) -> list:
    """Parse SSE ``data:`` frames from a response body into JSON objects.

    The ``[DONE]`` sentinel is kept as the literal string.
    """
    out = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                out.append("[DONE]")
            else:
                out.append(json.loads(payload))
    return out


def _fake_stream(events):
    """Build a fake ``run_agent_chat_stream`` yielding the given events."""
    from api.expert.types import ExpertStreamEvent

    async def _stream(product_id, query, session_id=None, history=None,
                      model=None, seed_history=False, **kwargs):
        for ev in events:
            yield ev

    return _stream


def _ev(event_type: str, content) -> "ExpertStreamEvent":
    from api.expert.types import ExpertStreamEvent
    return ExpertStreamEvent(event_type, content)


def _turn_of(frames) -> str:
    """The turn id announced by the FIRST frame of an /ask response."""
    assert isinstance(frames[0], dict) and "turn_id" in frames[0], frames[:2]
    return frames[0]["turn_id"]


def _session_of(frames) -> str:
    """The session id announced right after the turn frame (new sessions)."""
    for f in frames:
        if isinstance(f, dict) and "session_id" in f:
            return f["session_id"]
    raise AssertionError(f"no session_id frame in {frames[:3]}")


def _slow_stream(events, delay: float):
    """A fake stream that sleeps ``delay``s before every event (turns take
    wall-clock time — needed for disconnect/cancel/409 race tests)."""
    async def _stream(product_id, query, session_id=None, history=None,
                      model=None, seed_history=False, **kwargs):
        for ev in events:
            await asyncio.sleep(delay)
            yield ev

    return _stream


def _drain_frames(text: str) -> list:
    """Parse SSE ``data:`` frames from a raw (partial) response body chunk."""
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                out.append("[DONE]")
            else:
                try:
                    out.append(json.loads(payload))
                except ValueError:
                    pass
    return out


@pytest.fixture
def client(isolated_db, monkeypatch):
    """TestClient over the expert router with auth disabled."""
    import api.auth.deps as deps
    import api.routers.expert as expert_router

    monkeypatch.setattr(deps, "AUTH_PROVIDER", "none")
    app, client = build_test_client(isolated_db, [expert_router])
    return client


# --- SSE contract ------------------------------------------------------------
class TestSseContract:
    def test_full_event_sequence(self, client, isolated_db, monkeypatch):
        _seed_product(isolated_db)
        import api.routers.expert as expert_router

        events = [
            _ev("status", "retrieving"),
            _ev("tool_call", json.dumps({"name": "knowledge_recall",
                                         "args": {"query": "deploy"}})),
            _ev("tool_result", json.dumps({"name": "knowledge_recall",
                                           "content": "[1] some evidence"})),
            _ev("status", "thinking"),
            _ev("reasoning", "checking the sources"),
            _ev("status", "answering"),
            _ev("content", "Deploy "),
            _ev("content", "takes 5 minutes"),
        ]
        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _fake_stream(events))

        resp = client.post("/api/products/prod_1/ask", json={"query": "how to deploy?"})
        assert resp.status_code == 200
        frames = _frames(resp)
        assert frames[-1] == "[DONE]"

        statuses = [f["status"] for f in frames if isinstance(f, dict) and "status" in f]
        assert statuses == ["retrieving", "thinking", "answering"]

        # The turn announcement is the FIRST frame; a NEW session is
        # announced second (before any statuses).
        turn_id = _turn_of(frames)
        assert turn_id
        session_id = _session_of(frames)
        assert session_id

        tool_calls = [f["tool_call"] for f in frames if isinstance(f, dict) and "tool_call" in f]
        assert tool_calls == [{"name": "knowledge_recall", "args": {"query": "deploy"}}]
        tool_results = [f["tool_result"] for f in frames if isinstance(f, dict) and "tool_result" in f]
        assert tool_results == [{"name": "knowledge_recall",
                                 "content": "[1] some evidence"}]
        reasoning = "".join(f["reasoning"] for f in frames if isinstance(f, dict) and "reasoning" in f)
        assert reasoning == "checking the sources"
        content = "".join(f["content"] for f in frames if isinstance(f, dict) and "content" in f)
        assert content == "Deploy takes 5 minutes"

        # Header mirrors the announced session id.
        assert resp.headers.get("x-session-id") == session_id

    def test_error_frame_then_done(self, client, monkeypatch):
        import api.routers.expert as expert_router

        async def _boom(*args, **kwargs):
            yield _ev("status", "retrieving")
            raise ValueError("LLM unavailable")

        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _boom)
        resp = client.post("/api/products/prod_1/ask", json={"query": "hi"})
        assert resp.status_code == 200
        frames = _frames(resp)
        assert frames[-1] == "[DONE]"
        errors = [f["error"] for f in frames if isinstance(f, dict) and "error" in f]
        # Controlled ValueError messages still surface verbatim.
        assert errors == ["LLM unavailable"]

    def test_unexpected_error_frame_is_generic(self, client, monkeypatch):
        """Non-ValueError exceptions leak nothing (review #5)."""
        import api.routers.expert as expert_router

        async def _boom(*args, **kwargs):
            raise RuntimeError("http://internal-host:8000/v1 exploded")

        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _boom)
        resp = client.post("/api/products/prod_1/ask", json={"query": "hi"})
        assert resp.status_code == 200
        frames = _frames(resp)
        assert frames[-1] == "[DONE]"
        errors = [f["error"] for f in frames if isinstance(f, dict) and "error" in f]
        assert errors == ["Expert agent stream failed; see server logs"]
        assert "internal-host" not in resp.text

    def test_empty_query_400(self, client):
        resp = client.post("/api/products/prod_1/ask", json={"query": "  "})
        assert resp.status_code == 400

    def test_unknown_product_streams_without_session(self, client, monkeypatch):
        """Legacy behavior: unknown product still streams (no 404), no session."""
        import api.routers.expert as expert_router

        events = [_ev("content", "answer")]
        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _fake_stream(events))
        resp = client.post("/api/products/ghost/ask", json={"query": "hi"})
        assert resp.status_code == 200
        frames = _frames(resp)
        assert frames[-1] == "[DONE]"
        assert all("session_id" not in f for f in frames if isinstance(f, dict))
        assert "x-session-id" not in resp.headers


# --- Session lifecycle -------------------------------------------------------
class TestSessionLifecycle:
    def test_new_session_announced_and_persisted(self, client, monkeypatch, isolated_db):
        import api.routers.expert as expert_router

        _seed_product(isolated_db)
        events = [_ev("status", "answering"), _ev("content", "the answer")]
        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _fake_stream(events))

        resp = client.post("/api/products/prod_1/ask", json={"query": "q1"})
        assert resp.status_code == 200
        frames = _frames(resp)
        session_id = _session_of(frames)

        # Session appears in the listing.
        listing = client.get("/api/products/prod_1/chat/sessions").json()
        assert [s["id"] for s in listing] == [session_id]
        assert listing[0]["title"] == "q1"
        assert listing[0]["product_id"] == "prod_1"

        # Transcript: user turn + assistant answer.
        messages = client.get(
            f"/api/products/prod_1/chat/sessions/{session_id}/messages"
        ).json()
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant"]
        assert messages[0]["content"] == "q1"
        assert messages[1]["content"] == "the answer"

    def test_tool_rows_persisted(self, client, monkeypatch, isolated_db):
        import api.routers.expert as expert_router

        _seed_product(isolated_db)
        events = [
            _ev("status", "retrieving"),
            _ev("tool_call", json.dumps({"name": "spec_read", "args": {"name": "API"}})),
            _ev("tool_result", json.dumps({"name": "spec_read", "content": "openapi..."})),
            _ev("content", "based on spec"),
        ]
        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _fake_stream(events))
        resp = client.post("/api/products/prod_1/ask", json={"query": "read spec"})
        session_id = _session_of(_frames(resp))

        messages = client.get(
            f"/api/products/prod_1/chat/sessions/{session_id}/messages"
        ).json()
        roles = [m["role"] for m in messages]
        assert roles == ["user", "tool", "tool", "assistant"]
        tool_rows = [m for m in messages if m["role"] == "tool"]
        assert tool_rows[0]["tool_name"] == "spec_read"
        assert tool_rows[0]["content"] == '{"name": "API"}'
        assert tool_rows[1]["tool_name"] == "spec_read"
        assert tool_rows[1]["content"] == "openapi..."

    def test_continue_session_no_second_announcement(self, client, monkeypatch, isolated_db):
        import api.routers.expert as expert_router

        _seed_product(isolated_db)
        events = [_ev("content", "a1")]
        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _fake_stream(events))
        resp = client.post("/api/products/prod_1/ask", json={"query": "q1"})
        session_id = _session_of(_frames(resp))

        events2 = [_ev("content", "a2")]
        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _fake_stream(events2))
        resp2 = client.post(
            "/api/products/prod_1/ask",
            json={"query": "q2", "session_id": session_id},
        )
        assert resp2.status_code == 200
        frames = _frames(resp2)
        # Continuation: no new announcement frame, header still present.
        assert all("session_id" not in f for f in frames if isinstance(f, dict))
        assert resp2.headers.get("x-session-id") == session_id

        # Both turns are in one transcript, in order.
        messages = client.get(
            f"/api/products/prod_1/chat/sessions/{session_id}/messages"
        ).json()
        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
        assert [m["content"] for m in messages] == ["q1", "a1", "q2", "a2"]

    def test_continue_unknown_session_404(self, client, monkeypatch):
        import api.routers.expert as expert_router

        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _fake_stream([]))
        resp = client.post(
            "/api/products/prod_1/ask",
            json={"query": "q", "session_id": "chat_does_not_exist"},
        )
        assert resp.status_code == 404

    def test_sessions_scoped_per_product(self, client, monkeypatch, isolated_db):
        import api.routers.expert as expert_router

        _seed_product(isolated_db, "prod_1")
        _seed_product(isolated_db, "prod_2")
        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _fake_stream([_ev("content", "x")]))

        r1 = client.post("/api/products/prod_1/ask", json={"query": "q1"})
        s1 = _session_of(_frames(r1))
        client.post("/api/products/prod_2/ask", json={"query": "q2"})

        assert [s["id"] for s in client.get("/api/products/prod_1/chat/sessions").json()] == [s1]
        assert len(client.get("/api/products/prod_2/chat/sessions").json()) == 1
        # prod_2 cannot read prod_1's session.
        resp = client.get(f"/api/products/prod_2/chat/sessions/{s1}/messages")
        assert resp.status_code == 404

    def test_empty_session_messages(self, client, monkeypatch, isolated_db):
        import api.routers.expert as expert_router

        _seed_product(isolated_db)
        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _fake_stream([]))
        resp = client.post("/api/products/prod_1/ask", json={"query": "q"})
        session_id = _session_of(_frames(resp))
        messages = client.get(
            f"/api/products/prod_1/chat/sessions/{session_id}/messages"
        ).json()
        # No content events -> only the user row is persisted.
        assert [m["role"] for m in messages] == ["user"]


# --- Request limits (fix [5]) ------------------------------------------------
class TestRequestLimits:
    """Oversized queries / history are rejected with 422 BEFORE any agent or
    DB work (pydantic Field limits on the request models)."""

    def test_oversize_query_422(self, client):
        from api.routers.expert import MAX_QUERY_CHARS

        resp = client.post(
            "/api/products/prod_1/ask",
            json={"query": "x" * (MAX_QUERY_CHARS + 1)},
        )
        assert resp.status_code == 422

    def test_oversize_history_message_422(self, client):
        from api.routers.expert import MAX_QUERY_CHARS

        resp = client.post(
            "/api/products/prod_1/ask",
            json={
                "query": "hi",
                "messages": [
                    {"role": "user", "content": "x" * (MAX_QUERY_CHARS + 1)}
                ],
            },
        )
        assert resp.status_code == 422

    def test_too_many_history_items_422(self, client):
        from api.routers.expert import MAX_HISTORY_ITEMS

        resp = client.post(
            "/api/products/prod_1/ask",
            json={
                "query": "hi",
                "messages": [
                    {"role": "user", "content": f"m{i}"}
                    for i in range(MAX_HISTORY_ITEMS + 1)
                ],
            },
        )
        assert resp.status_code == 422

    def test_boundary_sized_query_streams(self, client, monkeypatch, isolated_db):
        """Exactly MAX_QUERY_CHARS passes validation (limit is inclusive)."""
        import api.routers.expert as expert_router
        from api.routers.expert import MAX_QUERY_CHARS

        _seed_product(isolated_db)
        monkeypatch.setattr(
            expert_router, "run_agent_chat_stream", _fake_stream([_ev("content", "ok")])
        )
        resp = client.post(
            "/api/products/prod_1/ask", json={"query": "x" * MAX_QUERY_CHARS}
        )
        assert resp.status_code == 200
        assert _frames(resp)[-1] == "[DONE]"


# --- Per-user session isolation (fix [6]) -------------------------------------
class TestSessionIsolation:
    """Sessions owned by ANOTHER user are invisible (404 / hidden listing);
    unowned (``user_id IS NULL``) sessions stay visible to everyone.

    With ``AUTH_PROVIDER=none`` the current user is the transient ``system``
    user (id ``system``), so ``user_other``'s sessions must be filtered out.
    """

    def _seed_sessions(self, db_mod):
        with db_mod.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="Acme"))
            db.add(UserORM(id="user_other", username="other", role="user"))
            db.add(ChatSessionORM(
                id="chat_other", product_id="prod_1",
                user_id="user_other", title="theirs",
            ))
            db.add(ChatSessionORM(
                id="chat_shared", product_id="prod_1",
                user_id=None, title="shared",
            ))
            db.commit()

    def test_other_users_sessions_hidden_in_listing(self, client, isolated_db):
        self._seed_sessions(isolated_db)
        listing = client.get("/api/products/prod_1/chat/sessions").json()
        assert [s["id"] for s in listing] == ["chat_shared"]

    def test_other_users_messages_404(self, client, isolated_db):
        self._seed_sessions(isolated_db)
        resp = client.get(
            "/api/products/prod_1/chat/sessions/chat_other/messages"
        )
        assert resp.status_code == 404

    def test_shared_session_messages_visible(self, client, isolated_db):
        self._seed_sessions(isolated_db)
        resp = client.get(
            "/api/products/prod_1/chat/sessions/chat_shared/messages"
        )
        assert resp.status_code == 200

    def test_continue_other_users_session_404(self, client, isolated_db, monkeypatch):
        import api.routers.expert as expert_router

        self._seed_sessions(isolated_db)
        monkeypatch.setattr(
            expert_router, "run_agent_chat_stream", _fake_stream([_ev("content", "x")])
        )
        resp = client.post(
            "/api/products/prod_1/ask",
            json={"query": "q", "session_id": "chat_other"},
        )
        assert resp.status_code == 404

    def test_continue_shared_session_ok(self, client, isolated_db, monkeypatch):
        import api.routers.expert as expert_router

        self._seed_sessions(isolated_db)
        monkeypatch.setattr(
            expert_router, "run_agent_chat_stream", _fake_stream([_ev("content", "x")])
        )
        resp = client.post(
            "/api/products/prod_1/ask",
            json={"query": "q", "session_id": "chat_shared"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-session-id") == "chat_shared"


# --- Auth --------------------------------------------------------------------
class TestAuth:
    def test_all_endpoints_require_auth(self, isolated_db, monkeypatch):
        import api.auth.deps as deps
        import api.routers.expert as expert_router

        # Auth enabled (local) without a cookie -> 401 everywhere. The real
        # get_current_user must run: no default-admin override here.
        monkeypatch.setattr(deps, "AUTH_PROVIDER", "local")
        app, client = build_test_client(
            isolated_db, [expert_router], default_admin_auth=False
        )
        assert client.post("/api/products/prod_1/ask",
                           json={"query": "hi"}).status_code == 401
        assert client.get("/api/products/prod_1/chat/sessions").status_code == 401
        assert client.get(
            "/api/products/prod_1/chat/sessions/x/messages"
        ).status_code == 401


# --- Detached turns (issue #9) ------------------------------------------------
class _AsgiCall:
    """Hand-driven ASGI request for mid-stream SSE scenarios.

    httpx's ``ASGITransport`` buffers the whole response before returning it,
    so incremental reads / mid-generation disconnects cannot be simulated
    with it. This driver runs the app coroutine directly on the test loop:
    ``receive()`` hands over the request body once and then blocks until
    ``disconnect()`` is called — the ASGI ``http.disconnect`` signal that
    starlette's ``StreamingResponse`` races against, cancelling the SSE
    generator while the DETACHED turn task keeps running. Every ASGI ``send``
    message is appended to ``messages`` so the streamed body can be inspected
    while the request is still in flight.
    """

    def __init__(self, app, method: str, path: str, json_body=None):
        self._app = app
        body = b"" if json_body is None else json.dumps(json_body).encode()
        headers = []
        if json_body is not None:
            headers = [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ]
        self._scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers,
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "root_path": "",
        }
        self._body = body
        self._request_sent = False
        self._disconnected = asyncio.Event()
        self.messages: list = []

    async def receive(self):
        if not self._request_sent:
            self._request_sent = True
            return {"type": "http.request", "body": self._body, "more_body": False}
        # A client that stays connected: block until the test walks away.
        # (The await is cancelled by the app itself once the response ends.)
        await self._disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(self, message):
        self.messages.append(message)

    def disconnect(self) -> None:
        """Simulate the client going away mid-stream."""
        self._disconnected.set()

    async def run(self) -> None:
        await self._app(self._scope, self.receive, self.send)

    @property
    def status_code(self) -> Optional[int]:
        for m in self.messages:
            if m["type"] == "http.response.start":
                return m["status"]
        return None

    def header(self, name: str) -> Optional[str]:
        target = name.lower().encode()
        for m in self.messages:
            if m["type"] == "http.response.start":
                for k, v in m.get("headers", []):
                    if k.lower() == target:
                        return v.decode()
        return None

    def body_text(self) -> str:
        return "".join(
            m.get("body", b"").decode("utf-8", "replace")
            for m in self.messages
            if m["type"] == "http.response.body"
        )

    async def wait_for_body(self, needle: str, timeout: float = 5.0) -> None:
        """Block until ``needle`` appears in the body streamed so far."""
        deadline = time.monotonic() + timeout
        while needle not in self.body_text():
            assert time.monotonic() < deadline, f"{needle!r} never streamed"
            await asyncio.sleep(0.005)


class TestDetachedTurns:
    """issue #9: an ask is a background task that outlives the request.

    The generation survives a client disconnect (starlette cancels the SSE
    generator; the turn task does not), supports re-attach replay, explicit
    cancel with a persisted partial answer, one-running-turn-per-session
    (409 + ``X-Turn-Id``), heartbeat comments, and session DELETE that
    cancels the running turn and wipes the transcript.
    """

    @pytest.fixture(autouse=True)
    def _fresh_rate_limits(self):
        from api.utils.rate_limit import reset_rate_limits

        reset_rate_limits()
        yield
        reset_rate_limits()

    def test_turn_survives_client_disconnect(self, client, monkeypatch, isolated_db):
        import api.expert.turns as turns_mod
        import api.routers.expert as expert_router

        _seed_product(isolated_db)
        events = [
            _ev("status", "retrieving"),
            _ev("content", "part-"),
            _ev("content", "tail"),
        ]
        monkeypatch.setattr(
            expert_router, "run_agent_chat_stream", _slow_stream(events, 0.2)
        )

        async def scenario():
            ask = _AsgiCall(client.app, "POST", "/api/products/prod_1/ask",
                            {"query": "q"})
            ask_task = asyncio.create_task(ask.run())
            # Wait for the FIRST agent event, then walk away mid-generation.
            await ask.wait_for_body('"status"')
            text = ask.body_text()
            assert "tail" not in text  # disconnected before the last event
            turn_id = _turn_of(_drain_frames(text))
            session_id = _session_of(_drain_frames(text))

            ask.disconnect()
            await ask_task  # the SSE generator is cancelled; the turn is not

            turn = turns_mod.get_turn(turn_id)
            assert turn is not None
            deadline = time.monotonic() + 10.0
            while not turn.finished:
                assert time.monotonic() < deadline, "turn never finished"
                await asyncio.sleep(0.01)
            assert turn.status == "completed"
            assert len(turn.events) == len(events)
            return session_id

        session_id = asyncio.run(scenario())

        # "completed" flips only AFTER the transcript is durable.
        messages = client.get(
            f"/api/products/prod_1/chat/sessions/{session_id}/messages"
        ).json()
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["content"] == "part-tail"

    def test_reattach_replays_finished_turn(self, client, monkeypatch, isolated_db):
        import api.routers.expert as expert_router

        _seed_product(isolated_db)
        events = [_ev("status", "answering"), _ev("content", "done answer")]
        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _fake_stream(events))

        resp = client.post("/api/products/prod_1/ask", json={"query": "q"})
        assert resp.status_code == 200
        frames = _frames(resp)
        turn_id = _turn_of(frames)
        session_id = _session_of(frames)

        # Re-attach to the FINISHED turn: pure replay, [DONE]-terminated.
        attach = client.get(f"/api/products/prod_1/ask/stream/{turn_id}")
        assert attach.status_code == 200
        replay = _frames(attach)
        assert replay[-1] == "[DONE]"
        # No announcement frames on re-attach — only the streamed events.
        assert all("turn_id" not in f for f in replay if isinstance(f, dict))
        assert all("session_id" not in f for f in replay if isinstance(f, dict))
        assert [
            f["status"] for f in replay
            if isinstance(f, dict) and "status" in f
        ] == ["answering"]
        content = "".join(
            f["content"] for f in replay
            if isinstance(f, dict) and "content" in f
        )
        assert content == "done answer"

        # Unknown turns are a plain 404 (attach and cancel alike).
        assert client.get(
            "/api/products/prod_1/ask/stream/turn_nope"
        ).status_code == 404
        assert client.post(
            "/api/products/prod_1/ask/turn_nope/cancel"
        ).status_code == 404

        # No RUNNING turn remains for the session.
        probe = client.get(
            f"/api/products/prod_1/chat/sessions/{session_id}/active-turn"
        )
        assert probe.status_code == 200
        assert probe.json() is None

    def test_active_turn_probe_and_409_while_running(
        self, client, monkeypatch, isolated_db
    ):
        import api.routers.expert as expert_router

        _seed_product(isolated_db)
        events = [
            _ev("status", "retrieving"),
            _ev("content", "the "),
            _ev("content", "answer"),
        ]
        monkeypatch.setattr(
            expert_router, "run_agent_chat_stream", _slow_stream(events, 0.15)
        )

        async def scenario():
            ask = _AsgiCall(client.app, "POST", "/api/products/prod_1/ask",
                            {"query": "q"})
            ask_task = asyncio.create_task(ask.run())
            await ask.wait_for_body("session_id")
            session_id = ask.header("x-session-id")
            turn_id = _turn_of(_drain_frames(ask.body_text()))

            # The running turn is discoverable (the re-attach probe).
            probe = _AsgiCall(
                client.app, "GET",
                f"/api/products/prod_1/chat/sessions/{session_id}/active-turn",
            )
            await probe.run()
            assert probe.status_code == 200
            descriptor = json.loads(probe.body_text())
            assert descriptor["turn_id"] == turn_id
            assert descriptor["status"] == "running"

            # A second ask into the same session → 409 + the running id.
            busy = _AsgiCall(client.app, "POST", "/api/products/prod_1/ask",
                             {"query": "again", "session_id": session_id})
            await busy.run()
            assert busy.status_code == 409
            assert busy.header("x-turn-id") == turn_id

            # The rejected ask persisted nothing (busy precedes the user row);
            # the running turn has not persisted its answer yet either.
            msgs = _AsgiCall(
                client.app, "GET",
                f"/api/products/prod_1/chat/sessions/{session_id}/messages",
            )
            await msgs.run()
            assert [m["role"] for m in json.loads(msgs.body_text())] == ["user"]

            # Re-attach while still running: replay + live tail + [DONE].
            attach = _AsgiCall(client.app, "GET",
                               f"/api/products/prod_1/ask/stream/{turn_id}")
            attach_task = asyncio.create_task(attach.run())
            await ask_task
            await attach_task
            replay = _drain_frames(attach.body_text())
            assert replay[-1] == "[DONE]"
            content = "".join(
                f["content"] for f in replay
                if isinstance(f, dict) and "content" in f
            )
            assert content == "the answer"

        asyncio.run(scenario())

    def test_cancel_running_turn_persists_partial_answer(
        self, client, monkeypatch, isolated_db
    ):
        import api.routers.expert as expert_router

        _seed_product(isolated_db)

        async def _stream(product_id, query, **kwargs):
            yield _ev("content", "part1")
            await asyncio.sleep(30.0)
            yield _ev("content", "part2")

        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _stream)

        async def scenario():
            ask = _AsgiCall(client.app, "POST", "/api/products/prod_1/ask",
                            {"query": "q"})
            ask_task = asyncio.create_task(ask.run())
            await ask.wait_for_body("part1")
            turn_id = _turn_of(_drain_frames(ask.body_text()))

            cancel = _AsgiCall(client.app, "POST",
                               f"/api/products/prod_1/ask/{turn_id}/cancel")
            await cancel.run()
            assert cancel.status_code == 200
            assert json.loads(cancel.body_text()) == {
                "turn_id": turn_id,
                "status": "cancelled",
            }

            # The orphaned stream still terminates cleanly with [DONE].
            await ask_task
            assert _drain_frames(ask.body_text())[-1] == "[DONE]"
            return ask.header("x-session-id")

        session_id = asyncio.run(scenario())

        # The partial answer made it into the transcript.
        messages = client.get(
            f"/api/products/prod_1/chat/sessions/{session_id}/messages"
        ).json()
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["content"] == "part1"

    def test_subscribe_emits_heartbeat_comments(
        self, client, monkeypatch, isolated_db
    ):
        import api.expert.turns as turns_mod
        import api.routers.expert as expert_router

        _seed_product(isolated_db)

        async def _stream(product_id, query, **kwargs):
            await asyncio.sleep(0.3)
            yield _ev("content", "late")

        monkeypatch.setattr(expert_router, "run_agent_chat_stream", _stream)
        monkeypatch.setattr(turns_mod, "HEARTBEAT_SECONDS", 0.05)

        async def scenario():
            ask = _AsgiCall(client.app, "POST", "/api/products/prod_1/ask",
                            {"query": "q"})
            await ask.run()
            text = ask.body_text()
            # Idle waits emit SSE comments (proxy keep-alive).
            assert text.count(": ping") >= 2
            assert _drain_frames(text)[-1] == "[DONE]"

        asyncio.run(scenario())

    def test_delete_session_removes_transcript(self, client, monkeypatch, isolated_db):
        import api.routers.expert as expert_router

        _seed_product(isolated_db)
        monkeypatch.setattr(
            expert_router, "run_agent_chat_stream",
            _fake_stream([_ev("content", "answer")]),
        )

        resp = client.post("/api/products/prod_1/ask", json={"query": "hello"})
        session_id = _session_of(_frames(resp))

        deleted = client.delete(f"/api/products/prod_1/chat/sessions/{session_id}")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": session_id, "cancelled_turns": []}

        assert client.get("/api/products/prod_1/chat/sessions").json() == []
        assert client.get(
            f"/api/products/prod_1/chat/sessions/{session_id}/messages"
        ).status_code == 404
        # Idempotency: deleting again is a 404, not an error.
        assert client.delete(
            f"/api/products/prod_1/chat/sessions/{session_id}"
        ).status_code == 404

    def test_delete_session_cancels_running_turn(
        self, client, monkeypatch, isolated_db
    ):
        import api.expert.turns as turns_mod
        import api.routers.expert as expert_router
        from api.models import ChatMessageORM

        _seed_product(isolated_db)
        monkeypatch.setattr(
            expert_router, "run_agent_chat_stream",
            _slow_stream([_ev("content", "never")], 30.0),
        )

        async def scenario():
            ask = _AsgiCall(client.app, "POST", "/api/products/prod_1/ask",
                            {"query": "q"})
            ask_task = asyncio.create_task(ask.run())
            await ask.wait_for_body("turn_id")
            turn_id = _turn_of(_drain_frames(ask.body_text()))
            session_id = ask.header("x-session-id")

            delete = _AsgiCall(client.app, "DELETE",
                               f"/api/products/prod_1/chat/sessions/{session_id}")
            await delete.run()
            assert delete.status_code == 200
            assert json.loads(delete.body_text()) == {
                "deleted": session_id,
                "cancelled_turns": [turn_id],
            }

            # The turn is forgotten entirely — no late writes can follow.
            assert turns_mod.get_turn(turn_id) is None
            assert turns_mod.active_for_session(session_id) is None

            msgs = _AsgiCall(
                client.app, "GET",
                f"/api/products/prod_1/chat/sessions/{session_id}/messages",
            )
            await msgs.run()
            assert msgs.status_code == 404

            # The orphaned stream still terminates — no hang.
            await ask_task
            return session_id

        session_id = asyncio.run(scenario())

        with isolated_db.SessionLocal() as db:
            assert db.query(ChatMessageORM).filter(
                ChatMessageORM.session_id == session_id
            ).count() == 0
