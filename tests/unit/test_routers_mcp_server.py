"""Unit tests for api.routers.mcp_server.

Covers:
- POST /api/mcp/message and POST /api/mcp (JSON-RPC entry points)
- initialize, notifications/initialized, tools/list
- tools/call: list_products (empty + with products), get_product_knowledge
  (not found, markdown, json, fallback to unverified), search_product_graph
  (success + exception), ask_expert (sync + async + exception)
- Unknown tool name -> -32601 error
- Unknown method -> -32601 error
- GET /api/mcp/sse (SSE endpoint)
- Direct _handle_mcp_request tests for edge cases
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


# --- Helpers ----------------------------------------------------------------
def _build_client(db_mod, mcp_mod):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(mcp_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[mcp_mod.get_db] = _get_test_db
    return app, TestClient(app)


def _seed_product(db_mod, product_id="prod_1", name="Acme", summary="A summary"):
    from api.models import ProductORM

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id=product_id, name=name, summary=summary))
        db.commit()
    return product_id


def _seed_verified_content(db_mod, product_id="prod_1"):
    from api.models import CodebaseORM, KnowledgeNodeORM

    with db_mod.SessionLocal() as db:
        db.add(CodebaseORM(
            id="cb_v", product_id=product_id, name="svc-v",
            verified=True, generated_docs="Verified docs",
            source="manual", verified_by="admin",
        ))
        db.add(KnowledgeNodeORM(
            id="node_v", product_id=product_id, title="Page V", slug="page-v",
            content_md="Node content", verified=True, source="manual",
        ))
        db.commit()


def _seed_unverified_content(db_mod, product_id="prod_1"):
    from api.models import CodebaseORM, KnowledgeNodeORM

    with db_mod.SessionLocal() as db:
        db.add(CodebaseORM(
            id="cb_u", product_id=product_id, name="svc-u",
            verified=False, generated_docs="Unverified docs",
            source="manual",
        ))
        db.add(KnowledgeNodeORM(
            id="node_u", product_id=product_id, title="Page U", slug="page-u",
            content_md="Unverified node", source="manual",
        ))
        db.commit()


# --- Direct _handle_mcp_request tests ---------------------------------------
class TestHandleMcpRequest:
    def test_initialize(self, isolated_db):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest

        req = McpJsonRpcRequest(id=1, method="initialize")
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        # _handle_mcp_request is async when tools/call is used, but initialize
        # returns a dict directly. However the function is declared async,
        # so we need to await it. Let's use asyncio.
        import asyncio

        res = asyncio.run(result) if hasattr(result, "__await__") else result
        assert res["jsonrpc"] == "2.0"
        assert res["id"] == 1
        assert res["result"]["protocolVersion"] == "2024-11-05"
        assert res["result"]["serverInfo"]["name"] == "productarium"
        assert "tools" in res["result"]["capabilities"]

    def test_notifications_initialized(self, isolated_db):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        req = McpJsonRpcRequest(id=2, method="notifications/initialized")
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result) if hasattr(result, "__await__") else result
        assert res["result"] == {}

    def test_initialized_alias(self, isolated_db):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        req = McpJsonRpcRequest(id=2, method="initialized")
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result) if hasattr(result, "__await__") else result
        assert res["result"] == {}

    def test_tools_list(self, isolated_db):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        req = McpJsonRpcRequest(id=3, method="tools/list")
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result) if hasattr(result, "__await__") else result
        assert "tools" in res["result"]
        tool_names = [t["name"] for t in res["result"]["tools"]]
        assert "list_products" in tool_names
        assert "get_product_knowledge" in tool_names
        assert "search_product_graph" in tool_names
        assert "ask_expert" in tool_names

    def test_unknown_method(self, isolated_db):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        req = McpJsonRpcRequest(id=4, method="something/unknown")
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result) if hasattr(result, "__await__") else result
        assert "error" in res
        assert res["error"]["code"] == -32601

    def test_unknown_tool(self, isolated_db):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        req = McpJsonRpcRequest(
            id=5, method="tools/call",
            params={"name": "nonexistent_tool", "arguments": {}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        assert "error" in res
        assert res["error"]["code"] == -32601
        assert "Unknown tool" in res["error"]["message"]

    def test_list_products_empty(self, isolated_db):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        req = McpJsonRpcRequest(
            id=6, method="tools/call",
            params={"name": "list_products", "arguments": {}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        assert "No products found" in text

    def test_list_products_with_data(self, isolated_db):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        _seed_product(isolated_db, "prod_1", "Acme", "A great product")
        req = McpJsonRpcRequest(
            id=7, method="tools/call",
            params={"name": "list_products", "arguments": {}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        assert "Acme" in text
        assert "A great product" in text
        assert "prod_1" in text

    def test_get_product_knowledge_not_found(self, isolated_db):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        req = McpJsonRpcRequest(
            id=8, method="tools/call",
            params={"name": "get_product_knowledge", "arguments": {"product_id": "ghost"}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        assert "not found" in text

    def test_get_product_knowledge_markdown(self, isolated_db):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        _seed_product(isolated_db)
        _seed_verified_content(isolated_db)
        req = McpJsonRpcRequest(
            id=9, method="tools/call",
            params={"name": "get_product_knowledge", "arguments": {"product_id": "prod_1"}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        assert "Verified Knowledge" in text
        assert "Verified docs" in text

    def test_get_product_knowledge_json(self, isolated_db):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        _seed_product(isolated_db)
        _seed_verified_content(isolated_db)
        req = McpJsonRpcRequest(
            id=10, method="tools/call",
            params={"name": "get_product_knowledge",
                    "arguments": {"product_id": "prod_1", "format": "json"}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        data = json.loads(text)
        assert data["product"]["id"] == "prod_1"

    def test_get_product_knowledge_fallback_unverified(self, isolated_db):
        """When no verified content exists, falls back to ALL content."""
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        _seed_product(isolated_db)
        _seed_unverified_content(isolated_db)
        req = McpJsonRpcRequest(
            id=11, method="tools/call",
            params={"name": "get_product_knowledge", "arguments": {"product_id": "prod_1"}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        # Unverified content should appear in the fallback.
        assert "Unverified docs" in text

    def test_search_product_graph_success(self, isolated_db, monkeypatch):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        async def _fake_query(query, dataset_name=None, top_k=20):
            return "Fact: Productarium is awesome"

        import api.cognee as cognee_mod
        monkeypatch.setattr(cognee_mod, "query_cognee", _fake_query)

        req = McpJsonRpcRequest(
            id=12, method="tools/call",
            params={"name": "search_product_graph",
                    "arguments": {"product_id": "prod_1", "query": "what is it"}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        assert "Knowledge Graph Results" in text
        assert "Fact: Productarium is awesome" in text

    def test_search_product_graph_no_results(self, isolated_db, monkeypatch):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        async def _fake_query(query, dataset_name=None, top_k=20):
            return None

        import api.cognee as cognee_mod
        monkeypatch.setattr(cognee_mod, "query_cognee", _fake_query)

        req = McpJsonRpcRequest(
            id=13, method="tools/call",
            params={"name": "search_product_graph",
                    "arguments": {"product_id": "prod_1", "query": "nothing"}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        assert "No knowledge graph triplets found" in text

    def test_search_product_graph_exception(self, isolated_db, monkeypatch):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        async def _fake_query(query, dataset_name=None, top_k=20):
            raise RuntimeError("cognee unavailable")

        import api.cognee as cognee_mod
        monkeypatch.setattr(cognee_mod, "query_cognee", _fake_query)

        req = McpJsonRpcRequest(
            id=14, method="tools/call",
            params={"name": "search_product_graph",
                    "arguments": {"product_id": "prod_1", "query": "test"}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        assert "Error querying cognee" in text

    def test_ask_expert_sync_result(self, isolated_db, monkeypatch):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        def _fake_expert(product_id, query, model=None, stream=False):
            return "The answer is 42"

        import api.expert.chat as expert_chat_mod
        monkeypatch.setattr(expert_chat_mod, "run_expert_chat", _fake_expert)

        req = McpJsonRpcRequest(
            id=15, method="tools/call",
            params={"name": "ask_expert",
                    "arguments": {"product_id": "prod_1", "query": "what?"}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        assert "42" in text

    def test_ask_expert_async_result(self, isolated_db, monkeypatch):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        async def _fake_expert(product_id, query, model=None, stream=False):
            return "Async answer"

        import api.expert.chat as expert_chat_mod
        monkeypatch.setattr(expert_chat_mod, "run_expert_chat", _fake_expert)

        req = McpJsonRpcRequest(
            id=16, method="tools/call",
            params={"name": "ask_expert",
                    "arguments": {"product_id": "prod_1", "query": "what?"}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        assert "Async answer" in text

    def test_ask_expert_exception(self, isolated_db, monkeypatch):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        def _fake_expert(product_id, query, model=None, stream=False):
            raise RuntimeError("Expert unavailable")

        import api.expert.chat as expert_chat_mod
        monkeypatch.setattr(expert_chat_mod, "run_expert_chat", _fake_expert)

        req = McpJsonRpcRequest(
            id=17, method="tools/call",
            params={"name": "ask_expert",
                    "arguments": {"product_id": "prod_1", "query": "what?"}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        assert "Error asking expert" in text

    def test_ask_expert_empty_result(self, isolated_db, monkeypatch):
        from api.routers.mcp_server import _handle_mcp_request, McpJsonRpcRequest
        import asyncio

        async def _fake_expert(product_id, query, model=None, stream=False):
            return ""

        import api.expert.chat as expert_chat_mod
        monkeypatch.setattr(expert_chat_mod, "run_expert_chat", _fake_expert)

        req = McpJsonRpcRequest(
            id=18, method="tools/call",
            params={"name": "ask_expert",
                    "arguments": {"product_id": "prod_1", "query": "what?"}},
        )
        result = _handle_mcp_request(req, isolated_db.SessionLocal())
        res = asyncio.run(result)
        text = res["result"]["content"][0]["text"]
        assert "empty answer" in text


# --- HTTP endpoint tests ----------------------------------------------------
class TestMcpEndpoints:
    def test_post_message_initialize(self, isolated_db):
        from api.routers import mcp_server as mcp_mod

        app, client = _build_client(isolated_db, mcp_mod)
        resp = client.post("/api/mcp/message", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["protocolVersion"] == "2024-11-05"

    def test_post_root_alias(self, isolated_db):
        """POST /api/mcp (no /message) should also work."""
        from api.routers import mcp_server as mcp_mod

        app, client = _build_client(isolated_db, mcp_mod)
        resp = client.post("/api/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "tools" in body["result"]

    def test_post_tools_call_list_products(self, isolated_db):
        from api.routers import mcp_server as mcp_mod

        _seed_product(isolated_db, "prod_1", "Acme", "Sum")
        app, client = _build_client(isolated_db, mcp_mod)
        resp = client.post("/api/mcp/message", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "list_products", "arguments": {}},
        })
        assert resp.status_code == 200
        text = resp.json()["result"]["content"][0]["text"]
        assert "Acme" in text

    def test_sse_endpoint(self, isolated_db):
        from api.routers import mcp_server as mcp_mod

        app, client = _build_client(isolated_db, mcp_mod)
        resp = client.get("/api/mcp/sse")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "event: endpoint" in body
        assert "/api/mcp/message?session_id=" in body
