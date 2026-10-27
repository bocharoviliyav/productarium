#!/usr/bin/env python3
"""Integration tests for the inbound MCP server at ``/api/mcp`` (contract 3).

Builds a fresh FastMCP streamable-HTTP app per test (no singleton state), runs
it under ``TestClient`` with the session-manager lifespan, and talks real
JSON-RPC (initialize → notifications/initialized → tools/list → tools/call)
exactly like an external MCP client would:

- Bearer auth: missing/invalid token → 401 + WWW-Authenticate; a valid API
  token (sha256) passes, stamps ``last_used_at`` and is per-owner rate-limited
  (429 + Retry-After once the shared public bucket is dry).
- ``tools/list`` exposes the five contract tools (incl. SPEC-008
  ``knowledge_search`` / ``knowledge_graph``).
- ``list_products``, ``get_product_knowledge`` (strict verified-only + unknown
  product error), ``knowledge_search`` (SQLite recent-chunks fallback with
  citations), ``knowledge_graph`` (TTL cache + over-cap node shrinking),
  ``ask_expert`` (monkeypatched expert + timeout bound).
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

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_TOKEN = "inbound-test-token"
_HEADERS = {
    "Authorization": f"Bearer {_TOKEN}",
    "Accept": "application/json, text/event-stream",
}


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Every valid request now consumes the shared public per-user bucket."""
    from api.utils.rate_limit import reset_rate_limits

    reset_rate_limits()
    yield
    reset_rate_limits()


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


def _seed_graph(db_mod, product_id: str = "prod_1"):
    """One stored knowledge-graph row (term rms -> page p1) for SPEC-008."""
    from api.memory import knowledge_graph as kg
    from api.models import KnowledgeGraphORM

    graph = kg.new_graph()
    tkey = kg.upsert_term(graph, "rms", description="revenue management system")
    pkey = kg.upsert_page(graph, "codebase", "cb_1", "p1", title="Overview")
    graph["edges"][kg.edge_key(tkey, "mentions", pkey)] = {
        "src": tkey, "dst": pkey, "rel": "mentions",
        "weight": 0.9, "context": "", "updated_at": kg.utc_now_iso(),
    }
    kg.clear_graph_cache(product_id)
    with db_mod.SessionLocal() as db:
        db.add(KnowledgeGraphORM(product_id=product_id, graph=graph, version=1))
        db.commit()
    kg.clear_graph_cache(product_id)


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
                "knowledge_search",
                "knowledge_graph",
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
    def test_strict_verified_only_export(self, isolated_db):
        from api.models import KnowledgeNodeORM

        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(
                client, "get_product_knowledge", {"product_id": "prod_1"}
            )
            assert resp.status_code == 200
            text = _tool_text(resp)
            # P-5: unverified content is NEVER exported (no all-content
            # fallback); only the export header remains.
            assert "On-call guide" not in text
            assert "Acme" in text

            with isolated_db.SessionLocal() as db:
                db.query(KnowledgeNodeORM).filter_by(id="node_1").one().verified = True
                db.commit()

            resp = _call_tool(
                client, "get_product_knowledge", {"product_id": "prod_1"}
            )
            assert "# On-call guide" in _tool_text(resp)

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


# --- knowledge_search / knowledge_graph (SPEC-008) -------------------------------------
class TestKnowledgeGraphTools:
    def test_knowledge_search_cited_chunks(self, isolated_db):
        # SQLite -> the pgvector search falls back to recent chunks, so seeded
        # KnowledgeChunkORM rows come back with citation headers.
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(
                client,
                "knowledge_search",
                {"product_id": "prod_1", "query": "deploy window", "top_k": 5},
            )
            assert resp.status_code == 200
            text = _tool_text(resp)
            assert "02:00 UTC" in text
            assert "chunk=chunk_1" in text  # citation header present
            assert "[source: knowledge_node:node_1" in text

    def test_knowledge_search_missing_product_id_is_error(self, isolated_db):
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(client, "knowledge_search", {"query": "deploy"})
            assert resp.status_code == 200
            result = resp.json()["result"]
            assert result.get("isError") is True  # required-param validation

    def test_knowledge_search_empty_product_id_is_error(self, isolated_db):
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(
                client, "knowledge_search", {"product_id": "  ", "query": "deploy"}
            )
            assert resp.status_code == 200
            assert resp.json()["result"].get("isError") is True

    def test_knowledge_graph_whole(self, isolated_db):
        import json as _json

        _seed(isolated_db)
        _seed_graph(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(client, "knowledge_graph", {"product_id": "prod_1"})
            assert resp.status_code == 200
            graph = _json.loads(_tool_text(resp))
            assert "term:rms" in graph["nodes"]
            assert "page:codebase:cb_1:p1" in graph["nodes"]
            assert len(graph["edges"]) == 1

    def test_knowledge_graph_focus_term_subgraph(self, isolated_db):
        import json as _json

        _seed(isolated_db)
        _seed_graph(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(
                client,
                "knowledge_graph",
                {"product_id": "prod_1", "focus_term": "RMS"},
            )
            assert resp.status_code == 200
            graph = _json.loads(_tool_text(resp))
            assert set(graph["nodes"]) == {"term:rms", "page:codebase:cb_1:p1"}
            # Unknown focus term -> empty graph, not an error.
            resp = _call_tool(
                client,
                "knowledge_graph",
                {"product_id": "prod_1", "focus_term": "nope"},
            )
            assert _json.loads(_tool_text(resp))["nodes"] == {}

    def test_knowledge_graph_missing_product_id_is_error(self, isolated_db):
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(client, "knowledge_graph", {"focus_term": "rms"})
            assert resp.status_code == 200
            result = resp.json()["result"]
            assert result.get("isError") is True

    def test_knowledge_graph_unknown_product_empty(self, isolated_db):
        import json as _json

        from api.memory import knowledge_graph as kg

        _seed(isolated_db)
        kg.clear_graph_cache()
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(
                client, "knowledge_graph", {"product_id": "prod_nope"}
            )
            graph = _json.loads(_tool_text(resp))
            assert graph["nodes"] == {} and graph["edges"] == {}

    def test_knowledge_graph_uses_ttl_cache(self, isolated_db):
        """Q-7: no fresh=True — the second call is served from the TTL cache
        (the DB row disappearing mid-test cannot change the answer)."""
        import json as _json

        from api.memory import knowledge_graph as kg
        from api.models import KnowledgeGraphORM

        _seed(isolated_db)
        _seed_graph(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(client, "knowledge_graph", {"product_id": "prod_1"})
            assert "term:rms" in _json.loads(_tool_text(resp))["nodes"]

            with isolated_db.SessionLocal() as db:
                db.query(KnowledgeGraphORM).filter_by(product_id="prod_1").delete()
                db.commit()

            resp = _call_tool(client, "knowledge_graph", {"product_id": "prod_1"})
            assert "term:rms" in _json.loads(_tool_text(resp))["nodes"]  # cached

    def test_knowledge_graph_over_cap_shrinks_nodes_not_json(self, isolated_db, monkeypatch):
        """Q-7: over-cap graphs shrink max_nodes and re-serialize — the output
        stays VALID JSON under the cap, never a string-cut "(truncated)"."""
        import json as _json

        import api.mcp.inbound as inbound
        from api.memory import knowledge_graph as kg
        from api.models import KnowledgeGraphORM

        _seed(isolated_db)
        nodes = {
            f"term:t{i}": {
                "key": f"term:t{i}", "type": "term", "name": f"t{i}",
                "description": "d" * 200, "aliases": [], "updated_at": "",
            }
            for i in range(200)
        }
        kg.clear_graph_cache("prod_1")
        with isolated_db.SessionLocal() as db:
            db.add(KnowledgeGraphORM(product_id="prod_1", graph={"nodes": nodes, "edges": {}}, version=1))
            db.commit()
        kg.clear_graph_cache("prod_1")

        monkeypatch.setattr(inbound, "_MAX_KNOWLEDGE_CHARS", 20_000)
        with TestClient(_make_app()) as client:
            _handshake(client)
            resp = _call_tool(client, "knowledge_graph", {"product_id": "prod_1"})
            assert resp.status_code == 200
            text = _tool_text(resp)
            assert len(text) <= 20_000
            assert "(truncated)" not in text
            graph = _json.loads(text)  # still valid JSON
            assert 0 < len(graph["nodes"]) < 200  # node budget was shrunk


# --- bearer rate limit (P-4) ----------------------------------------------------------
class TestRateLimit:
    def test_bearer_path_is_rate_limited_429(self, isolated_db, monkeypatch):
        monkeypatch.setenv("RATE_PUBLIC_PER_USER_MINUTE", "1")
        _seed(isolated_db)
        with TestClient(_make_app()) as client:
            _handshake(client)  # initialize consumes the single token
            resp = _rpc(client, "tools/list", msg_id=2)
            assert resp.status_code == 429
            assert resp.headers.get("retry-after")
            assert "Too many requests" in resp.json()["detail"]


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
