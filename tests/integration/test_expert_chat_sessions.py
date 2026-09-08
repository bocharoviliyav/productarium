"""Integration tests for the Wave B expert chat router (SSE + sessions).

Covers the fixed SSE contract over the LangGraph agent stream:
- event sequence: ``status: retrieving`` first, ``tool_call`` / ``tool_result``
  frames, ``status: thinking`` + ``reasoning``, ``status: answering`` +
  ``content`` deltas, terminating ``data: [DONE]``;
- new-session announcement (``session_id`` frame + ``X-Session-Id`` header);
- session continuation via ``session_id`` (no second announcement);
- transcript persistence across two turns (query / tool rows / answer);
- ``GET /chat/sessions`` and ``GET /chat/sessions/{id}/messages`` (CRUD,
  404 on another product's session, 401 without auth);
- error frame when the agent stream raises; [DONE] always terminates.

The agent stream is faked via monkeypatching ``run_agent_chat_stream`` on
the router module (the documented test seam), so no LLM is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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

        # The session announcement is the FIRST frame (before statuses).
        assert isinstance(frames[0], dict) and "session_id" in frames[0]
        session_id = frames[0]["session_id"]
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
        session_id = frames[0]["session_id"]

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
        session_id = _frames(resp)[0]["session_id"]

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
        session_id = _frames(resp)[0]["session_id"]

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
        s1 = _frames(r1)[0]["session_id"]
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
        session_id = _frames(resp)[0]["session_id"]
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
        assert client.post("/api/products/prod_1/ask/doc",
                           json={"query": "hi"}).status_code == 401
        assert client.get("/api/products/prod_1/chat/sessions").status_code == 401
        assert client.get(
            "/api/products/prod_1/chat/sessions/x/messages"
        ).status_code == 401
