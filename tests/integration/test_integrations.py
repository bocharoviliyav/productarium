#!/usr/bin/env python3
"""Unit tests for the integrations framework (plan section G).

Covers:
- ``api.integrations.registry`` auto-discovery + get_connector + list_connectors
- ``api.integrations.confluence`` connector with a mocked HTTP layer
- ``api.formats.markitdown.convert_to_markdown`` with a mocked markitdown + the
  graceful-placeholder path when markitdown is unavailable
- ``api.integrations._git_base.GitConnector.extract_repo_name``

No live services (no DB / Ollama / network) — HTTP is mocked per-test and the
settings store uses an isolated SQLite env (mirrors test_foundation.py).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest


# --- Shared isolated env (mirrors test_foundation.py) ------------------------
@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DB_PROVIDER", "sqlite")
    monkeypatch.setenv("DB_HOST", str(tmp_path))
    monkeypatch.setenv("DB_NAME", str(db_file))
    monkeypatch.setenv("DB_USERNAME", "")
    monkeypatch.setenv("DB_PASSWORD", "")
    monkeypatch.setenv("AUTH_PROVIDER", "none")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    from cryptography.fernet import Fernet

    monkeypatch.setenv("SETTINGS_SECRET_KEY", Fernet.generate_key().decode())
    yield

# --- Registry ---------------------------------------------------------------
class TestRegistry:
    def test_autodiscovers_all_built_in_connectors(self):
        from api.integrations.registry import reset_registry, list_connectors

        reset_registry()
        names = {c["name"] for c in list_connectors()}
        # The four built-in connectors must be auto-discovered.
        assert {"github", "gitlab", "confluence", "mcp"} <= names

    def test_list_connectors_has_required_fields(self):
        from api.integrations.registry import reset_registry, list_connectors

        reset_registry()
        for c in list_connectors():
            assert {"name", "display_name", "description", "requires_credentials", "configured", "kind"} <= set(c)

    def test_get_connector_returns_none_for_unknown(self):
        from api.integrations.registry import reset_registry, get_connector

        reset_registry()
        assert get_connector("does-not-exist") is None

    def test_get_connector_instantiates_known(self):
        from api.integrations.registry import reset_registry, get_connector
        from api.integrations.confluence import ConfluenceConnector

        reset_registry()
        conn = get_connector("confluence")
        assert isinstance(conn, ConfluenceConnector)

    def test_explicit_register_decorator(self):
        from api.integrations.base import IntegrationConnector
        from api.integrations.registry import register, reset_registry, get_connector

        reset_registry()

        @register
        class DummyConnector(IntegrationConnector):
            name = "dummy-test-connector"
            display_name = "Dummy"

            def test(self):
                return {"success": True, "message": "ok"}

            def list_spaces(self):
                return []

            def pull(self, source_id, opts=None):
                return {"title": source_id, "markdown": "", "attachments": []}

        try:
            assert get_connector("dummy-test-connector") is not None
        finally:
            reset_registry()

    def test_register_rejects_unnamed(self):
        from api.integrations.base import IntegrationConnector
        from api.integrations.registry import register, reset_registry

        reset_registry()

        class NoName(IntegrationConnector):
            def test(self):
                return {}

            def list_spaces(self):
                return []

            def pull(self, source_id, opts=None):
                return {}

        with pytest.raises(ValueError):
            register(NoName)


# --- Git base ---------------------------------------------------------------
class TestGitBase:
    def test_extract_repo_name_github(self):
        from api.integrations._git_base import GitConnector

        assert (
            GitConnector.extract_repo_name("https://github.com/owner/repo.git", "github")
            == "owner_repo"
        )

    def test_extract_repo_name_gitlab_subgroup(self):
        from api.integrations._git_base import GitConnector

        assert (
            GitConnector.extract_repo_name("https://gitlab.com/group/sub/repo", "gitlab")
            == "sub_repo"
        )


# --- Confluence connector (mocked HTTP) -------------------------------------
def _confluence_with_mocks(monkeypatch, *, pages=None, children=None, attachments=None, spaces=None):
    """Build a ConfluenceConnector with its HTTP layer mocked."""
    from api.integrations.confluence import ConfluenceConnector

    conn = ConfluenceConnector(
        config={
            "base_url": "https://example.atlassian.net",
            "token": "fake-token",
            "username": "user@example.com",
        }
    )

    def _page_id_from(path: str, suffix: str) -> str:
        # path like /wiki/api/v2/pages/{id}/children
        return path.split("/pages/")[1].split(suffix)[0]

    def fake_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if path.startswith("/wiki/api/v2/spaces"):
            return {"results": spaces or [], "_links": {}}
        if path.startswith("/wiki/api/v2/pages/") and "/children" in path:
            pid = _page_id_from(path, "/children")
            # `children` is a dict {page_id: [child dicts]}.
            return {"results": (children or {}).get(pid, [])}
        if path.startswith("/wiki/api/v2/pages/") and "/attachments" in path:
            pid = _page_id_from(path, "/attachments")
            # `attachments` is a dict {page_id: [attachment dicts]}.
            return {"results": (attachments or {}).get(pid, [])}
        if path.startswith("/wiki/api/v2/pages/"):
            # Page fetch — return the canned page matching the id in the path.
            page_id = path.rstrip("/").split("/")[-1].split("?")[0]
            page = next((p for p in (pages or []) if str(p.get("id")) == page_id), None)
            if page is None:
                raise ValueError(f"mock: no page {page_id}")
            return page
        raise ValueError(f"mock: unexpected path {path}")

    monkeypatch.setattr(conn, "_get", fake_get)
    monkeypatch.setattr(conn, "_get_bytes", lambda url: b"FAKE-ATTACHMENT-BYTES")
    return conn


class TestConfluenceConnector:
    def test_is_configured_requires_base_and_token(self):
        from api.integrations.confluence import ConfluenceConnector

        assert ConfluenceConnector(config={}).is_configured() is False
        assert ConfluenceConnector(config={"base_url": "x"}).is_configured() is False
        assert (
            ConfluenceConnector(config={"base_url": "x", "token": "t"}).is_configured()
            is True
        )

    def test_auth_headers_basic_when_username(self):
        from api.integrations.confluence import ConfluenceConnector

        conn = ConfluenceConnector(
            config={"base_url": "x", "token": "t", "username": "u@example.com"}
        )
        headers = conn._auth_headers()
        assert headers["Authorization"].startswith("Basic ")
        # Bearer used when no username.
        conn2 = ConfluenceConnector(config={"base_url": "x", "token": "t"})
        assert conn2._auth_headers()["Authorization"] == "Bearer t"

    def test_test_success(self, monkeypatch):
        conn = _confluence_with_mocks(monkeypatch, spaces=[{"id": "1", "name": "Eng", "key": "ENG"}])
        result = conn.test()
        assert result["success"] is True
        assert "Confluence" in result["message"]

    def test_test_not_configured(self):
        from api.integrations.confluence import ConfluenceConnector

        result = ConfluenceConnector(config={}).test()
        assert result["success"] is False

    def test_list_spaces(self, monkeypatch):
        conn = _confluence_with_mocks(
            monkeypatch,
            spaces=[
                {"id": "1", "name": "Engineering", "key": "ENG"},
                {"id": "2", "name": "Product", "key": "PROD"},
            ],
        )
        spaces = conn.list_spaces()
        assert len(spaces) == 2
        assert {s["key"] for s in spaces} == {"ENG", "PROD"}
        assert all(s["type"] == "space" for s in spaces)

    def test_list_spaces_not_configured(self):
        from api.integrations.confluence import ConfluenceConnector

        assert ConfluenceConnector(config={}).list_spaces() == []

    def test_list_spaces_filters_to_configured_space(self, monkeypatch):
        conn = _confluence_with_mocks(
            monkeypatch,
            spaces=[
                {"id": "1", "name": "Engineering", "key": "ENG"},
                {"id": "2", "name": "Product", "key": "PROD"},
            ],
        )
        conn.config["space"] = "PROD"
        spaces = conn.list_spaces()
        assert [s["key"] for s in spaces] == ["PROD"]

    def test_pull_single_page_with_attachments(self, monkeypatch):
        pages = [
            {
                "id": "123",
                "title": "Architecture",
                "body": {"representation": "storage", "value": "<p>Overview</p>"},
                "spaceId": "1",
            }
        ]
        attachments = {
            "123": [
                {
                    "title": "spec.docx",
                    "_links": {"download": "/wiki/api/v2/attachments/9/download"},
                }
            ]
        }
        conn = _confluence_with_mocks(monkeypatch, pages=pages, attachments=attachments)
        # Mock markitdown conversion so we don't need the real package behavior.
        monkeypatch.setattr(
            "api.integrations.confluence.convert_to_markdown",
            lambda raw, filename=None: f"MD({filename})",
        )
        result = conn.pull("123")
        assert result["title"] == "Architecture"
        assert result["page_id"] == "123"
        assert result["page_count"] == 1
        assert "<p>Overview</p>" in result["markdown"]
        assert result["attachments"] == [{"filename": "spec.docx", "markdown": "MD(spec.docx)"}]

    def test_pull_recursive_walks_children(self, monkeypatch):
        pages = [
            {
                "id": "1",
                "title": "Root",
                "body": {"value": "<p>root</p>"},
                "spaceId": "S",
            },
            {
                "id": "2",
                "title": "Child",
                "body": {"value": "<p>child</p>"},
                "spaceId": "S",
            },
        ]
        children = {"1": [{"id": "2", "title": "Child"}]}
        conn = _confluence_with_mocks(
            monkeypatch,
            pages=pages,
            children=children,
        )
        result = conn.pull("1", opts={"recursive": True})
        assert result["page_count"] == 2
        assert "Root" in result["markdown"]
        assert "Child" in result["markdown"]

    def test_pull_not_configured_raises(self):
        from api.integrations.confluence import ConfluenceConnector

        with pytest.raises(ValueError):
            ConfluenceConnector(config={}).pull("123")


# --- markitdown client (mocked) ---------------------------------------------
class TestMarkitdownClient:
    def test_convert_bytes_with_mocked_markitdown(self, monkeypatch):
        import api.formats.markitdown as mc

        class FakeResult:
            text_content = "# Converted markdown"

        class FakeMarkItDown:
            def convert(self, stream):
                # Verify a BytesIO with the filename hint was passed.
                assert getattr(stream, "name", None) == "report.pdf"
                assert stream.getvalue() == b"PDFBYTES"
                return FakeResult()

        # Reset the lazy cache so our fake is picked up.
        monkeypatch.setattr(mc, "_MARKITDOWN_AVAILABLE", None)
        monkeypatch.setattr(mc, "_MARKITDOWN", None)
        monkeypatch.setattr(mc, "_load_markitdown", lambda: FakeMarkItDown)

        out = mc.convert_to_markdown(b"PDFBYTES", filename="report.pdf")
        assert out == "# Converted markdown"

    def test_convert_path_with_mocked_markitdown(self, monkeypatch, tmp_path):
        import api.formats.markitdown as mc

        f = tmp_path / "doc.docx"
        f.write_bytes(b"DOCXBYTES")

        class FakeResult:
            text_content = "docx content"

        class FakeMarkItDown:
            def convert(self, path):
                assert path == str(f)
                return FakeResult()

        monkeypatch.setattr(mc, "_MARKITDOWN_AVAILABLE", None)
        monkeypatch.setattr(mc, "_MARKITDOWN", None)
        monkeypatch.setattr(mc, "_load_markitdown", lambda: FakeMarkItDown)

        assert mc.convert_to_markdown(str(f)) == "docx content"

    def test_placeholder_when_markitdown_missing(self, monkeypatch):
        import api.formats.markitdown as mc

        monkeypatch.setattr(mc, "_MARKITDOWN_AVAILABLE", None)
        monkeypatch.setattr(mc, "_MARKITDOWN", None)
        monkeypatch.setattr(mc, "_load_markitdown", lambda: None)

        out = mc.convert_to_markdown(b"X", filename="foo.pdf")
        assert "markitdown" in out
        assert "not installed" in out
        assert out.startswith("<!--")

    def test_placeholder_on_conversion_error(self, monkeypatch):
        import api.formats.markitdown as mc

        class FakeMarkItDown:
            def convert(self, stream):
                raise RuntimeError("boom")

        monkeypatch.setattr(mc, "_MARKITDOWN_AVAILABLE", None)
        monkeypatch.setattr(mc, "_MARKITDOWN", None)
        monkeypatch.setattr(mc, "_load_markitdown", lambda: FakeMarkItDown)

        out = mc.convert_to_markdown(b"X", filename="foo.docx")
        assert out.startswith("<!--")
        assert "conversion error" in out


# --- MCP connector (config-only, no network) --------------------------------
class TestMcpConnector:
    def test_is_configured_requires_servers(self):
        from api.integrations.mcp import McpConnector

        assert McpConnector(config={}).is_configured() is False
        assert McpConnector(config={"servers": []}).is_configured() is False
        assert (
            McpConnector(
                config={
                    "servers": [
                        {"name": "s1", "transport": "http", "url": "http://localhost:8080/mcp"}
                    ]
                }
            ).is_configured()
            is True
        )
        assert (
            McpConnector(
                config={
                    "servers": [
                        {"name": "s2", "transport": "stdio", "command": ["node", "s.js"]}
                    ]
                }
            ).is_configured()
            is True
        )

    def test_list_spaces_returns_configured_sources(self):
        from api.integrations.mcp import McpConnector

        conn = McpConnector(
            config={
                "servers": [
                    {
                        "name": "srv",
                        "transport": "stdio",
                        "command": ["node", "s.js"],
                        "sources": [
                            {"id": "s1", "title": "Wiki"},
                            {"id": "s2", "title": "Docs"},
                        ],
                    }
                ]
            }
        )
        spaces = conn.list_spaces()
        assert {s["id"] for s in spaces} == {"srv:s1", "srv:s2"}
        assert all(s["server"] == "srv" for s in spaces)

    def test_list_spaces_multi_server(self):
        from api.integrations.mcp import McpConnector

        conn = McpConnector(
            config={
                "servers": [
                    {
                        "name": "a",
                        "transport": "http",
                        "url": "http://x",
                        "sources": [{"id": "a1", "title": "A1"}],
                    },
                    {
                        "name": "b",
                        "transport": "stdio",
                        "command": ["node", "b.js"],
                        "sources": [{"id": "b1", "title": "B1"}],
                    },
                ]
            }
        )
        spaces = conn.list_spaces()
        ids = {s["id"] for s in spaces}
        assert ids == {"a:a1", "b:b1"}

    def test_test_no_servers(self):
        from api.integrations.mcp import McpConnector

        result = McpConnector(config={}).test()
        assert result["success"] is False
        assert "servers" not in result or result.get("servers") is None or result["servers"] == []

    def test_test_http_server_not_reachable(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        conn = McpConnector(
            config={
                "servers": [
                    {"name": "s1", "transport": "http", "url": "http://localhost:1/mcp"}
                ]
            }
        )
        # _http_initialize will fail because nothing is listening on port 1.
        result = conn.test()
        assert result["success"] is False
        assert "servers" in result
        assert len(result["servers"]) == 1
        assert result["servers"][0]["success"] is False

    def test_test_invalid_transport(self):
        from api.integrations.mcp import McpConnector

        conn = McpConnector(
            config={"servers": [{"name": "s1", "transport": "bogus"}]}
        )
        result = conn.test()
        assert result["success"] is False
        assert result["servers"][0]["message"] == "Invalid transport 'bogus'."

    def test_pull_not_configured_raises(self):
        from api.integrations.mcp import McpConnector

        with pytest.raises(ValueError):
            McpConnector(config={}).pull("src1")

    def test_pull_unknown_server_raises(self):
        from api.integrations.mcp import McpConnector

        conn = McpConnector(
            config={"servers": [{"name": "real", "transport": "http", "url": "http://x"}]}
        )
        with pytest.raises(ValueError):
            conn.pull("nonexistent")

    def test_parse_source_id(self):
        from api.integrations.mcp import McpConnector

        assert McpConnector._parse_source_id("srv:tool") == ("srv", "tool")
        assert McpConnector._parse_source_id("srv") == ("srv", None)
        assert McpConnector._parse_source_id("srv:") == ("srv", None)

    def test_pull_mocked_http(self, monkeypatch):
        from api.integrations.mcp import McpConnector

        conn = McpConnector(
            config={
                "servers": [
                    {
                        "name": "srv",
                        "transport": "http",
                        "url": "http://localhost:9999/mcp",
                        "tool": "fetch",
                    }
                ]
            }
        )
        # Mock _server_call to avoid real HTTP.
        monkeypatch.setattr(
            conn,
            "_server_call",
            lambda server, tool, arguments: "# Fetched Content\n\nHello world.",
        )
        result = conn.pull("srv")
        assert result["title"] == "srv"
        assert "Fetched Content" in result["markdown"]
        assert result["server"] == "srv"
        assert result["tool"] == "fetch"
        assert result["transport"] == "http"
        assert result["attachments"] == []


# --- Router end-to-end (TestClient + in-memory DB) --------------------------
class _PullOnlyConnector:
    """A minimal connector used only by the router test (registered explicitly)."""

    name = "pull-test"
    display_name = "Pull Test"
    description = "test connector"
    requires_credentials = False

    def __init__(self, config=None):
        self.config = config or {}

    @classmethod
    def get_config(cls):
        return {}

    def is_configured(self):
        return True

    def test(self):
        return {"success": True, "message": "ok"}

    def list_spaces(self):
        return [{"id": "s1", "title": "Source 1", "type": "repo"}]

    def pull(self, source_id, opts=None):
        return {
            "title": source_id,
            "markdown": f"# {source_id}\n\nbody",
            "attachments": [{"filename": "a.docx", "markdown": "att-md"}],
            "repo_url": "https://example.com/x",
            "repo_type": "github",
        }


class TestIntegrationsRouter:
    def _client_and_db(self, monkeypatch):
        # The FastAPI TestClient runs async endpoints in a worker thread, while
        # the default SQLite engine uses check_same_thread=True. We can't edit
        # api/db.py, so we override the get_db dependency with a thread-safe
        # in-memory engine (StaticPool shares one connection).
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from api.models import Base, UserORM
        from api.db import get_db
        from api.auth.deps import get_current_user
        import api.routers.integrations as router_mod
        from api.integrations.registry import register, reset_registry
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        reset_registry()
        register(_PullOnlyConnector)
        # Avoid real cognee indexing in the background.
        async def _noop_index(text, dataset_name):
            return None
        monkeypatch.setattr(router_mod, "add_and_index_document", _noop_index)

        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
        Base.metadata.create_all(bind=engine)
        TestSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

        def _override_get_db():
            session = TestSessionLocal()
            try:
                yield session
            finally:
                session.close()

        # Override auth so the router tests are independent of AUTH_PROVIDER
        # (which may be frozen at import time when the full suite runs).
        _admin = UserORM(id="test-admin", username="admin", role="admin", provider="local")

        app = FastAPI()
        app.dependency_overrides[get_db] = _override_get_db
        # Also override the get_db the router captured at import time
        # (router_mod.get_db). It is the same object as `get_db` when the router
        # is imported fresh here, but other tests reload api.db (creating a new
        # get_db) after the router was pre-imported via api.api — overriding both
        # keeps this test robust to that, matching test_admin_public's pattern.
        app.dependency_overrides[router_mod.get_db] = _override_get_db
        app.dependency_overrides[get_current_user] = lambda: _admin
        app.include_router(router_mod.router)
        client = TestClient(app)
        return client, TestSessionLocal

    def test_list_connectors_endpoint(self, monkeypatch):
        client, _ = self._client_and_db(monkeypatch)
        resp = client.get("/api/integrations")
        assert resp.status_code == 200
        names = {c["name"] for c in resp.json()}
        assert "pull-test" in names

    def test_spaces_endpoint(self, monkeypatch):
        client, _ = self._client_and_db(monkeypatch)
        resp = client.get("/api/integrations/pull-test/spaces")
        assert resp.status_code == 200
        assert resp.json() == [{"id": "s1", "title": "Source 1", "type": "repo"}]

    def test_test_endpoint_admin_ok_when_auth_none(self, monkeypatch):
        # AUTH_PROVIDER=none -> require_admin returns the system admin.
        client, _ = self._client_and_db(monkeypatch)
        resp = client.post("/api/integrations/pull-test/test")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_unknown_connector_404(self, monkeypatch):
        client, _ = self._client_and_db(monkeypatch)
        assert client.get("/api/integrations/nope/spaces").status_code == 404
        assert client.post("/api/integrations/nope/test").status_code == 404

    def test_from_integration_creates_knowledge_node_default(self, monkeypatch):
        # Non-git pulls now create KnowledgeNodes (the legacy polymorphic
        # ``documentation`` artifact type was removed along with the
        # ``artifacts/from-integration`` endpoint that took an artifact_type).
        client, SessionLocal = self._client_and_db(monkeypatch)
        from api.models import ProductORM, KnowledgeNodeORM

        with SessionLocal() as session:
            session.add(ProductORM(id="prod_1", name="Acme"))
            session.commit()
        resp = client.post(
            "/api/products/prod_1/knowledge/from-integration",
            json={"connector": "pull-test", "source_id": "src-1"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["product_id"] == "prod_1"
        assert body["source"] == "api"
        assert "src-1" in body["title"]
        assert "att-md" in body["content_md"]
        with SessionLocal() as session:
            nodes = session.query(KnowledgeNodeORM).filter_by(product_id="prod_1").all()
            assert len(nodes) == 1

    def test_from_integration_codebase_only_for_git(self, monkeypatch):
        client, SessionLocal = self._client_and_db(monkeypatch)
        from api.models import ProductORM

        with SessionLocal() as session:
            session.add(ProductORM(id="prod_2", name="Acme2"))
            session.commit()
        # The codebase endpoint rejects non-git connectors (github/gitlab only).
        resp = client.post(
            "/api/products/prod_2/codebases/from-integration",
            json={
                "connector": "pull-test",
                "source_id": "src-2",
            },
        )
        assert resp.status_code == 400

    def test_from_integration_creates_knowledge_node(self, monkeypatch):
        client, SessionLocal = self._client_and_db(monkeypatch)
        from api.models import ProductORM, KnowledgeNodeORM

        with SessionLocal() as session:
            session.add(ProductORM(id="prod_3", name="Acme3"))
            session.commit()
        resp = client.post(
            "/api/products/prod_3/knowledge/from-integration",
            json={"connector": "pull-test", "source_id": "My Page"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["product_id"] == "prod_3"
        assert body["node_type"] == "page"
        assert body["source"] == "api"
        assert body["slug"] == "my-page"
        with SessionLocal() as session:
            nodes = session.query(KnowledgeNodeORM).filter_by(product_id="prod_3").all()
            assert len(nodes) == 1

    def test_from_integration_unknown_product_404(self, monkeypatch):
        client, _ = self._client_and_db(monkeypatch)
        resp = client.post(
            "/api/products/no-such-prod/knowledge/from-integration",
            json={"connector": "pull-test", "source_id": "x"},
        )
        assert resp.status_code == 404

    def test_from_integration_unknown_connector_404(self, monkeypatch):
        client, SessionLocal = self._client_and_db(monkeypatch)
        from api.models import ProductORM

        with SessionLocal() as session:
            session.add(ProductORM(id="prod_4", name="Acme4"))
            session.commit()
        resp = client.post(
            "/api/products/prod_4/knowledge/from-integration",
            json={"connector": "nope", "source_id": "x"},
        )
        assert resp.status_code == 404
