#!/usr/bin/env python3
"""Integration tests for the inbound MCP server at ``/api/mcp`` (contract 3).

Builds a fresh FastMCP streamable-HTTP app per test (no singleton state), runs
it under ``TestClient`` with the session-manager lifespan, and talks real
JSON-RPC (initialize → notifications/initialized → tools/list → tools/call)
exactly like an external MCP client would:

- Bearer auth: missing/invalid token → 401 + WWW-Authenticate; a valid API
  token (sha256) passes and stamps ``last_used_at``.
- ``tools/list`` exposes the four contract tools.
- ``list_products``, ``get_product_knowledge`` (verified-first fallback +
  unknown product error), ``search_knowledge`` (SQLite recent-chunks fallback
  with citations), ``ask_expert`` (monkeypatched expert + timeout bound).
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient  # noqa: E402

_TOKEN = "inbound-test-token"
_HEADERS = {
    "Authorization": f"Bearer {_TOKEN}",
    "Accept": "application/json, text/event-stream",
}


def _make_app():
    """A fresh FastAPI app mounting a fresh inbound MCP server (auth-wrapped)."""
    from fastapi import FastAPI

    import api.mcp.inbound as inbound

    mcp = inbound._build_mcp()
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(lifespan=lifespan)
    app.mount("/api/mcp", inbound._BearerAuthASGI(mcp_app))
    return app


def _seed(db_mod):
    from api.models import (
        ApiTokenORM,
        KnowledgeChunkORM,
        KnowledgeNodeORM,
        ProductORM,
        UserORM,
    )

    with db_mod.SessionLocal() as db:
        db.add(UserORM(id="user_1", username="alice", role="user", provider="local"))
        db.add(
            ApiTokenORM(
                id="tok_1",
                user_id="user_1",
                name="inbound",
                token_hash=hashlib.sha256(_TOKEN.encode()).hexdigest(),
            )
        )
        db.add(ProductORM(id="prod_1", name="Acme", summary="acme summary"))
        db.add(
            KnowledgeNodeORM(
                id="node_1",
                product_id="prod_1",
                title="Guide",
                slug="guide",
                content_md="# On-call guide",
                source="manual",
            )
        )
        db.add(
            KnowledgeChunkORM(
                id="chunk_1",
                product_id="prod_1",
                source_type="knowledge_node",
                source_id="node_1",
                chunk_index=0,
                content="The deploy window is 02:00 UTC",
            )
        )
        db.commit()


def _rpc(client, method, params=None, *, msg_id=1, headers=None):
    payload = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        payload["params"] = params
    return client.post("/api/mcp/", json=payload, headers=headers or _HEADERS)


def _handshake(client):
    resp = _rpc(
        client,
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"},
        },
        msg_id=0,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"]["serverInfo"]["name"] == "productarium"
    client.post(
        "/api/mcp/",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers=_HEADERS,
    )


def _call_tool(client, name, arguments, msg_id=1):
    return _rpc(
        client, "tools/call", {"name": name, "arguments": arguments}, msg_id=msg_id
    )


def _tool_text(resp):
    body = resp.json()
    result = body["result"]
    assert not result.get("isError"), body
    return result["content"][0]["text"]


# --- bearer auth ------------------------------------------------------------------
class TestInboundAuth:
    def test_missing_token_401(self, isolated_db):
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            resp = client.post(
                "/api/mcp/",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"Accept": "application/json, text/event-stream"},
            )
            assert resp.status_code == 401
            assert resp.headers.get("www-authenticate") == "Bearer"
            assert "invalid" in resp.json()["detail"].lower()

    def test_wrong_token_401(self, isolated_db):
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            resp = _rpc(
                client,
                "tools/list",
                headers={
                    "Authorization": "Bearer not-the-token",
                    "Accept": "application/json, text/event-stream",
                },
            )
            assert resp.status_code == 401

    def test_valid_token_passes_and_stamps_last_used(self, isolated_db):
        from api.models import ApiTokenORM

        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _rpc(client, "tools/list", msg_id=2)
            assert resp.status_code == 200
            assert resp.json()["result"]["tools"]

        with isolated_db.SessionLocal() as db:
            tok = db.query(ApiTokenORM).filter_by(id="tok_1").one()
            assert tok.last_used_at is not None

    def test_concurrent_401s_do_not_serialize_the_loop(self, isolated_db, monkeypatch):
        """The sync DB token check must run OFF the event loop.

        Five concurrent bad-token requests each spend ~0.1s in the (sync)
        check; serialized on the loop that is ~0.5s wall time, in threads it
        is ~0.1s.
        """
        import httpx

        import api.mcp.inbound as inbound

        _seed(isolated_db)
        calls = []

        def _slow_check(raw):
            calls.append(raw)
            time.sleep(0.1)
            return False  # force the 401 path

        monkeypatch.setattr(inbound, "_check_api_token", _slow_check)
        app = _make_app()

        async def _run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                tasks = [
                    ac.post(
                        "/api/mcp/",
                        json={"jsonrpc": "2.0", "id": i, "method": "tools/list"},
                        headers={
                            "Authorization": f"Bearer wrong-{i}",
                            "Accept": "application/json, text/event-stream",
                        },
                    )
                    for i in range(5)
                ]
                start = time.monotonic()
                resps = await asyncio.gather(*tasks)
                return time.monotonic() - start, resps

        elapsed, resps = asyncio.run(_run())
        assert all(r.status_code == 401 for r in resps)
        assert len(calls) == 5
        assert elapsed < 0.35  # serialized sync calls would take >= 0.5s


# --- tools/list ---------------------------------------------------------------------
class TestToolsList:
    def test_lists_contract_tools(self, isolated_db):
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _rpc(client, "tools/list", msg_id=2)
            assert resp.status_code == 200
            names = {t["name"] for t in resp.json()["result"]["tools"]}
            assert names == {
                "list_products",
                "get_product_knowledge",
                "search_knowledge",
                "ask_expert",
            }


# --- list_products --------------------------------------------------------------------
class TestListProducts:
    def test_returns_seeded_products(self, isolated_db):
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(client, "list_products", {})
            assert resp.status_code == 200
            body = resp.json()["result"]
            assert not body.get("isError"), body
            # FastMCP structured output: List[Dict] -> {"result": [...]};
            # the unstructured text blocks are one-per-product.
            products = body["structuredContent"]["result"]
            assert products == [
                {"id": "prod_1", "name": "Acme", "summary": "acme summary"}
            ]
            texts = [c["text"] for c in body["content"]]
            assert any("Acme" in t for t in texts)


# --- get_product_knowledge ---------------------------------------------------------------
class TestGetProductKnowledge:
    def test_returns_markdown_with_unverified_fallback(self, isolated_db):
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(
                client, "get_product_knowledge", {"product_id": "prod_1"}
            )
            assert resp.status_code == 200
            text = _tool_text(resp)
            # Nothing is verified -> fallback exports ALL content.
            assert "On-call guide" in text
            assert "# On-call guide" in text

    def test_unknown_product_is_tool_error(self, isolated_db):
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(
                client, "get_product_knowledge", {"product_id": "prod_nope"}
            )
            assert resp.status_code == 200
            result = resp.json()["result"]
            assert result.get("isError") is True


# --- search_knowledge ----------------------------------------------------------------------
class TestSearchKnowledge:
    def test_returns_cited_chunks(self, isolated_db):
        # SQLite -> the pgvector search falls back to recent chunks, so seeded
        # KnowledgeChunkORM rows come back with citation headers.
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(
                client,
                "search_knowledge",
                {"product_id": "prod_1", "query": "deploy window", "top_k": 5},
            )
            assert resp.status_code == 200
            text = _tool_text(resp)
            assert "02:00 UTC" in text
            assert "chunk=chunk_1" in text  # citation header present
            assert "[source: knowledge_node:node_1" in text


# --- ask_expert -----------------------------------------------------------------------------
class TestAskExpert:
    def test_returns_expert_answer(self, isolated_db, monkeypatch):
        import api.expert.chat as chat_mod

        seen = {}

        async def _fake_run_expert_chat(**kwargs):
            seen.update(kwargs)
            return "EXPERT ANSWER"

        monkeypatch.setattr(chat_mod, "run_expert_chat", _fake_run_expert_chat)
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(
                client, "ask_expert", {"product_id": "prod_1", "query": "how to deploy?"}
            )
            assert resp.status_code == 200
            assert _tool_text(resp) == "EXPERT ANSWER"
        assert seen["product_id"] == "prod_1"
        assert seen["query"] == "how to deploy?"

    def test_timeout_is_bounded(self, isolated_db, monkeypatch):
        monkeypatch.setenv("MCP_ASK_TIMEOUT_SECONDS", "1")
        import api.expert.chat as chat_mod

        async def _slow_expert(**kwargs):
            await asyncio.sleep(15)
            return "too late"

        monkeypatch.setattr(chat_mod, "run_expert_chat", _slow_expert)
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(
                client, "ask_expert", {"product_id": "prod_1", "query": "slow?"}
            )
            assert resp.status_code == 200
            result = resp.json()["result"]
            assert result.get("isError") is True
            assert "timed out" in result["content"][0]["text"].lower()
