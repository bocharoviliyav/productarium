#!/usr/bin/env python3
"""Unit tests for ``api.routers.http_integrations`` + agent tool wiring (issue #3).

Hermetic (isolated SQLite, fake httpx): covers CRUD, secret masking at rest,
placeholder/variable validation, PUT presence semantics, the bounded test
call, the LangChain tool builder (name/params/URL rendering + result cap)
and the expert-agent gathering seam.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _admin_user_orm():
    from api.models import UserORM

    return UserORM(
        id="user_admin1",
        username="admin",
        role="admin",
        provider="local",
        created_at=datetime.utcnow(),
    )


class _FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "hello"):
        self.status_code = status_code
        self.text = text


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient: records the URL, serves canned text."""

    last_url: str | None = None
    last_headers: dict | None = None
    response: _FakeResponse = _FakeResponse()

    def __init__(self, *args, **kwargs):
        self._headers = kwargs.get("headers")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None):
        type(self).last_url = url
        type(self).last_headers = headers
        return type(self).response


def _build_client(db_mod, hi_mod, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.auth import deps as auth_deps

    app = FastAPI()
    app.include_router(hi_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[hi_mod.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.require_admin] = _admin_user_orm
    return app, TestClient(app)


def _mod():
    from api.routers import http_integrations

    return http_integrations


def _stored_row(db_mod, integration_id):
    from api.models import HttpIntegrationORM

    with db_mod.SessionLocal() as db:
        return db.query(HttpIntegrationORM).filter(
            HttpIntegrationORM.id == integration_id
        ).one()


_CREATE_BODY = {
    "name": "jira lookup",
    "description": "Fetch a Jira issue",
    "url_template": "https://jira.example/rest/api/2/issue/{issue_key}",
    "headers": {"Authorization": "Bearer s3cret-token"},
    "variables": [
        {"name": "issue_key", "description": "Issue key", "default": "PROJ-1"}
    ],
    "enabled": True,
}


# --- create ------------------------------------------------------------------
class TestCreateHttpIntegration:
    def test_create_ok(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        resp = client.post("/api/admin/integrations/http", json=_CREATE_BODY)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["id"].startswith("httpint_")
        assert body["name"] == "jira lookup"
        assert body["url_template"] == _CREATE_BODY["url_template"]
        assert body["enabled"] is True
        assert body["variables"] == [
            {
                "name": "issue_key",
                "description": "Issue key",
                "default": "PROJ-1",
            }
        ]

    def test_headers_encrypted_at_rest_and_masked_in_response(
        self, isolated_db, monkeypatch
    ):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        body = client.post("/api/admin/integrations/http", json=_CREATE_BODY).json()
        assert body["headers_masked"] == {"Authorization": "***"}
        assert "s3cret-token" not in body

        row = _stored_row(isolated_db, body["id"])
        assert "s3cret-token" not in str(row.headers)  # encrypted at rest
        from api.mcp.secrets import decrypt_secret_dict

        assert decrypt_secret_dict(row.headers) == {
            "Authorization": "Bearer s3cret-token"
        }

    def test_implicit_product_name_placeholder_accepted(
        self, isolated_db, monkeypatch
    ):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        resp = client.post(
            "/api/admin/integrations/http",
            json={
                "name": "docs",
                "url_template": "https://docs.example/search?q=product:{product_name}",
            },
        )
        assert resp.status_code == 200, resp.text

    def test_list(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        client.post("/api/admin/integrations/http", json=_CREATE_BODY)
        resp = client.get("/api/admin/integrations/http")
        assert resp.status_code == 200
        assert [r["name"] for r in resp.json()] == ["jira lookup"]


# --- validation -----------------------------------------------------------------
class TestCreateValidation:
    def _post(self, client, payload):
        return client.post("/api/admin/integrations/http", json=payload)

    def test_bad_url_scheme(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(client, {"name": "x", "url_template": "ftp://h/api"})
        assert r.status_code == 400

    def test_credentials_in_url_rejected(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client, {"name": "x", "url_template": "http://u:p@h/api"}
        )
        assert r.status_code == 400

    def test_unknown_placeholder_rejected(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client,
            {
                "name": "x",
                "url_template": "https://h/api/{surprise}",
                "variables": [{"name": "issue_key"}],
            },
        )
        assert r.status_code == 400

    def test_bad_variable_name_rejected(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client,
            {
                "name": "x",
                "url_template": "https://h/api/{a b}",
                "variables": [{"name": "a b"}],
            },
        )
        assert r.status_code == 400

    def test_reserved_product_name_variable_rejected(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client,
            {
                "name": "x",
                "url_template": "https://h/api/{product_name}",
                "variables": [{"name": "product_name"}],
            },
        )
        assert r.status_code == 400

    def test_duplicate_variable_name_rejected(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        r = self._post(
            client,
            {
                "name": "x",
                "url_template": "https://h/api/{q}",
                "variables": [{"name": "q"}, {"name": "q"}],
            },
        )
        assert r.status_code == 400

    def test_duplicate_name_409(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        assert client.post("/api/admin/integrations/http", json=_CREATE_BODY).status_code == 200
        r = client.post(
            "/api/admin/integrations/http",
            json={**_CREATE_BODY, "url_template": "https://other.example/{issue_key}"},
        )
        assert r.status_code == 409


# --- update / delete --------------------------------------------------------------
class TestUpdateDelete:
    def test_update_renames_and_replaces_variables(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        created = client.post("/api/admin/integrations/http", json=_CREATE_BODY).json()
        r = client.put(
            f"/api/admin/integrations/http/{created['id']}",
            json={
                "name": "jira v2",
                "url_template": "https://jira.example/browse/{key}",
                "variables": [{"name": "key", "default": "ABC-9"}],
                "enabled": False,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["name"] == "jira v2"
        assert body["url_template"].endswith("/{key}")
        assert body["variables"][0]["name"] == "key"
        assert body["enabled"] is False

    def test_update_keeps_stored_variables_when_absent(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        created = client.post("/api/admin/integrations/http", json=_CREATE_BODY).json()
        r = client.put(
            f"/api/admin/integrations/http/{created['id']}", json={"enabled": False}
        )
        assert r.status_code == 200
        assert r.json()["variables"][0]["name"] == "issue_key"

    def test_update_all_masked_headers_keeps_stored_secrets(
        self, isolated_db, monkeypatch
    ):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        created = client.post("/api/admin/integrations/http", json=_CREATE_BODY).json()
        r = client.put(
            f"/api/admin/integrations/http/{created['id']}",
            json={"headers": {"Authorization": "***"}},
        )
        assert r.status_code == 200
        assert r.json()["headers_masked"] == {"Authorization": "***"}
        row = _stored_row(isolated_db, created["id"])
        from api.mcp.secrets import decrypt_secret_dict

        assert decrypt_secret_dict(row.headers) == {
            "Authorization": "Bearer s3cret-token"
        }

    def test_update_new_placeholder_needs_declared_variable(
        self, isolated_db, monkeypatch
    ):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        created = client.post("/api/admin/integrations/http", json=_CREATE_BODY).json()
        r = client.put(
            f"/api/admin/integrations/http/{created['id']}",
            json={"url_template": "https://h/api/{other}"},
        )
        assert r.status_code == 400

    def test_delete_then_404(self, isolated_db, monkeypatch):
        app, client = _build_client(isolated_db, _mod(), monkeypatch)
        created = client.post("/api/admin/integrations/http", json=_CREATE_BODY).json()
        assert (
            client.delete(f"/api/admin/integrations/http/{created['id']}").status_code
            == 200
        )
        assert (
            client.delete(f"/api/admin/integrations/http/{created['id']}").status_code
            == 404
        )


# --- bounded test call ---------------------------------------------------------------
class TestTestEndpoint:
    def test_call_uses_defaults_and_headers(self, isolated_db, monkeypatch):
        hi_mod = _mod()
        monkeypatch.setattr(hi_mod.httpx, "AsyncClient", _FakeAsyncClient)
        app, client = _build_client(isolated_db, hi_mod, monkeypatch)
        created = client.post("/api/admin/integrations/http", json=_CREATE_BODY).json()

        r = client.post(f"/api/admin/integrations/http/{created['id']}/test")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["status_code"] == 200
        assert body["body_preview"] == "hello"
        # The default variable value was substituted; secret headers attached.
        assert _FakeAsyncClient.last_url == "https://jira.example/rest/api/2/issue/PROJ-1"
        assert _FakeAsyncClient.last_headers == {"Authorization": "Bearer s3cret-token"}

    def test_call_error_returns_sanitized_detail(self, isolated_db, monkeypatch):
        hi_mod = _mod()

        class _BoomClient(_FakeAsyncClient):
            async def get(self, url, headers=None):
                raise RuntimeError("connect failed to https://secret-host/x")

        monkeypatch.setattr(hi_mod.httpx, "AsyncClient", _BoomClient)
        app, client = _build_client(isolated_db, hi_mod, monkeypatch)
        created = client.post("/api/admin/integrations/http", json=_CREATE_BODY).json()

        r = client.post(f"/api/admin/integrations/http/{created['id']}/test")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert "RuntimeError" in body["detail"]


# --- agent tool builder ---------------------------------------------------------------
class TestToolBuilder:
    def _seed_row(self, db_mod):
        from api.models import HttpIntegrationORM

        with db_mod.SessionLocal() as s:
            row = HttpIntegrationORM(
                id="httpint_seed",
                name="Jira Search",
                description="Lookup issues",
                url_template="https://jira.example/browse/{issue_key}",
                headers=None,
                variables=[{"name": "issue_key", "description": "Key", "default": "PROJ-1"}],
                enabled=True,
            )
            s.add(row)
            # A disabled row that must NOT become a tool.
            s.add(HttpIntegrationORM(
                id="httpint_off", name="Off Search",
                url_template="https://off.example/x", enabled=False,
            ))
            s.commit()

    def test_gather_builds_enabled_tools_only(self, isolated_db):
        self._seed_row(isolated_db)
        from api.agents.expert import _gather_http_integration_tools

        tools = _gather_http_integration_tools("prod_1", session_factory=isolated_db.SessionLocal)
        assert [t.name for t in tools] == ["Jira_Search"]
        assert set(tools[0].args.keys()) == {"issue_key"}

    def test_tool_renders_url_and_caps_result(self, isolated_db, monkeypatch):
        import asyncio

        self._seed_row(isolated_db)
        from api.integrations.http_tools import build_http_integration_tools

        _FakeAsyncClient.last_url = None
        _FakeAsyncClient.response = _FakeResponse(200, "x" * 500_000)
        # _call imports the resolver from api.config.timeout at call time.
        monkeypatch.setattr(
            "api.config.timeout.resolve_integration_http_timeout",
            lambda: 5.0,
        )
        # httpx is imported INSIDE _call: patch the shared module object.
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
        monkeypatch.setenv("MCP_TOOL_RESULT_MAX_CHARS", "1000")

        tools = build_http_integration_tools("prod_1", session_factory=isolated_db.SessionLocal)
        result = asyncio.run(tools[0].ainvoke({"issue_key": "ABC-42"}))

        assert _FakeAsyncClient.last_url == "https://jira.example/browse/ABC-42"
        assert result.startswith("HTTP 200")
        assert "truncated" in result
        assert len(result) < 1200

    def test_tool_never_raises_on_network_error(self, isolated_db, monkeypatch):
        import asyncio

        self._seed_row(isolated_db)
        from api.integrations.http_tools import build_http_integration_tools

        class _BoomClient(_FakeAsyncClient):
            async def get(self, url, headers=None):
                raise ConnectionError("boom")

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _BoomClient)
        tools = build_http_integration_tools("prod_1", session_factory=isolated_db.SessionLocal)
        result = asyncio.run(tools[0].ainvoke({}))
        assert "failed" in result


# --- sse MCP transport (issue #3, registry side) ---------------------------------------
class TestSseMcpTransport:
    def _client(self, db_mod, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.auth import deps as auth_deps
        from api.routers import mcp_admin

        app = FastAPI()
        app.include_router(mcp_admin.router)

        def _get_test_db():
            s = db_mod.SessionLocal()
            try:
                yield s
            finally:
                s.close()

        app.dependency_overrides[mcp_admin.get_db] = _get_test_db
        app.dependency_overrides[auth_deps.get_db] = _get_test_db
        app.dependency_overrides[auth_deps.require_admin] = _admin_user_orm
        return app, TestClient(app)

    def test_create_sse_server(self, isolated_db, monkeypatch):
        app, client = self._client(isolated_db, monkeypatch)
        r = client.post(
            "/api/admin/mcp/servers",
            json={
                "name": "legacy-sse",
                "transport": "sse",
                "url": "http://localhost:9000/sse",
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["transport"] == "sse"

    def test_sse_rejects_command_and_args(self, isolated_db, monkeypatch):
        app, client = self._client(isolated_db, monkeypatch)
        r = client.post(
            "/api/admin/mcp/servers",
            json={
                "name": "legacy-sse",
                "transport": "sse",
                "url": "http://localhost:9000/sse",
                "command": "/bin/x",
            },
        )
        assert r.status_code == 400

    def test_build_connection_sse(self, isolated_db):
        from api.mcp.manager import build_connection
        from api.models import McpServerORM

        server = McpServerORM(
            id="mcp_sse", name="s", transport="sse", url="http://h:1/sse"
        )
        conn = build_connection(server)
        assert conn["transport"] == "sse"
        assert conn["url"] == "http://h:1/sse"
        assert conn["timeout"] > 0
