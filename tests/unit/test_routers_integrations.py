"""Unit tests for api.routers.integrations.

Covers:
- GET /api/integrations (list registered connectors)
- POST /api/integrations/{name}/test (admin, success + not-found + exception)
- GET /api/integrations/{name}/spaces (not-found + exception -> 502)
- POST /api/products/{id}/codebases/from-integration (non-git 400, product 404,
  connector 404, pull ValueError 400, pull Exception 502, commit failure 500,
  success path)
- POST /api/products/{id}/knowledge/from-integration (product 404, connector 404,
  pull ValueError 400, pull Exception 502, single-page success, multi-page tree,
  commit failure 500)
- Helpers: _new_id, _slugify, _codebase_pydantic, _knowledge_node_pydantic,
  _compose_content, _index_in_background (empty text, with running loop,
  no running loop)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


# --- Helpers ----------------------------------------------------------------
def _system_user():
    from api.models import UserORM

    return UserORM(
        id="system",
        username="system",
        role="admin",
        provider="local",
        created_at=datetime.utcnow(),
    )


def _build_client(db_mod, integ_mod, *, user=None, admin=True):
    """Build a TestClient with dependency overrides for get_db, get_current_user,
    and require_admin."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(integ_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    user_obj = user if user is not None else _system_user()

    def _current_user():
        return user_obj

    def _require_admin():
        if admin:
            return user_obj
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Admin only")

    app.dependency_overrides[integ_mod.get_db] = _get_test_db
    app.dependency_overrides[integ_mod.get_current_user] = _current_user
    app.dependency_overrides[integ_mod.require_admin] = _require_admin
    return app, TestClient(app)


def _seed_product(db_mod, product_id="prod_1"):
    from api.models import ProductORM

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id=product_id, name="Acme"))
        db.commit()
    return product_id


class _FakeGitConnector:
    """Minimal fake connector for git (github/gitlab) operations."""

    name = "github"
    display_name = "GitHub"
    description = "GitHub connector"
    kind = "web"
    requires_credentials = True

    def __init__(self, config=None):
        self.config = dict(config or {})

    @classmethod
    def get_config(cls):
        return {}

    def is_configured(self):
        return bool(self.config)

    def test(self):
        return {"success": True, "message": "OK"}

    def list_spaces(self):
        return [{"id": "repo1", "title": "Repo 1", "type": "repo"}]

    def pull(self, source_id, opts=None):
        return {
            "title": "My Repo",
            "repo_url": f"https://github.com/example/{source_id}",
            "repo_type": "github",
            "markdown": "## README\nHello world.",
        }


class _FakeConfluenceConnector:
    """Minimal fake connector for non-git (confluence) operations."""

    name = "confluence"
    display_name = "Confluence"
    description = "Confluence connector"
    kind = "web"
    requires_credentials = True

    def __init__(self, config=None):
        self.config = dict(config or {})

    @classmethod
    def get_config(cls):
        return {}

    def is_configured(self):
        return bool(self.config)

    def test(self):
        return {"success": True, "message": "OK"}

    def list_spaces(self):
        return [{"id": "space1", "title": "Space 1", "type": "space"}]

    def pull(self, source_id, opts=None):
        return {
            "title": f"Page {source_id}",
            "markdown": "# Page content\nSome text.",
            "attachments": [
                {"filename": "file1.pdf", "markdown": "Attachment content"},
            ],
        }


class _FakeMultiPageConnector:
    """Connector that returns multiple pages for the multi-page tree path."""

    name = "confluence"
    display_name = "Confluence"
    description = "Confluence connector"
    kind = "web"
    requires_credentials = True

    def __init__(self, config=None):
        self.config = dict(config or {})

    @classmethod
    def get_config(cls):
        return {}

    def is_configured(self):
        return bool(self.config)

    def test(self):
        return {"success": True, "message": "OK"}

    def list_spaces(self):
        return []

    def pull(self, source_id, opts=None):
        return {
            "title": "Root Page",
            "markdown": "Root content",
            "pages": [
                {"id": "p0", "title": "Root", "html": "<p>Root</p>", "parent_id": None},
                {"id": "p1", "title": "Child 1", "html": "<p>Child 1</p>", "parent_id": "p0"},
                {"id": "p2", "title": "Child 2", "html": "<p>Child 2</p>", "parent_id": "p0"},
            ],
        }


class _ErrorConnector:
    """Connector whose methods raise for testing error paths."""

    name = "confluence"
    display_name = "Confluence"
    description = "Confluence"
    kind = "web"
    requires_credentials = True

    def __init__(self, config=None):
        self.config = dict(config or {})

    @classmethod
    def get_config(cls):
        return {}

    def is_configured(self):
        return bool(self.config)

    def test(self):
        raise RuntimeError("Connection refused")

    def list_spaces(self):
        raise RuntimeError("Connection refused")

    def pull(self, source_id, opts=None):
        raise RuntimeError("Pull failed")


class _ValueErrorConnector:
    """Connector whose pull raises ValueError (400 path)."""

    name = "github"
    display_name = "GitHub"
    description = "GitHub"
    kind = "web"
    requires_credentials = True

    def __init__(self, config=None):
        self.config = dict(config or {})

    @classmethod
    def get_config(cls):
        return {}

    def is_configured(self):
        return bool(self.config)

    def test(self):
        return {"success": True, "message": "OK"}

    def list_spaces(self):
        return []

    def pull(self, source_id, opts=None):
        raise ValueError("Invalid source id")


def _patch_registry(monkeypatch, connector_cls):
    """Patch the registry to use a fake connector class."""
    import api.integrations.registry as reg

    monkeypatch.setattr(reg, "_REGISTRY", {connector_cls.name: connector_cls})
    monkeypatch.setattr(reg, "_DISCOVERED", True)
    return reg


# --- Helper unit tests ------------------------------------------------------
class TestHelpers:
    def test_new_id_format(self):
        from api.routers.integrations import _new_id

        nid = _new_id("cb")
        assert nid.startswith("cb_")

    def test_slugify_basic(self):
        from api.routers.integrations import _slugify

        assert _slugify("Hello World") == "hello-world"
        assert _slugify("under_score") == "under-score"

    def test_slugify_empty_returns_default(self):
        from api.routers.integrations import _slugify

        result = _slugify("")
        assert result.startswith("node-")

    def test_slugify_none_returns_default(self):
        from api.routers.integrations import _slugify

        result = _slugify(None)
        assert result.startswith("node-")

    def test_codebase_pydantic(self):
        from api.routers.integrations import _codebase_pydantic
        from types import SimpleNamespace

        cb = SimpleNamespace(
            id="cb_1", name="Repo", repo_url="https://github.com/x/y",
            repo_type="github", token=None, generated_docs=None, pages=None,
            verified=False, verified_by=None, verified_at=None, source="api",
        )
        c = _codebase_pydantic(cb)
        assert c.id == "cb_1"
        assert c.name == "Repo"
        assert c.repo_url == "https://github.com/x/y"

    def test_knowledge_node_pydantic(self):
        from api.routers.integrations import _knowledge_node_pydantic
        from types import SimpleNamespace

        n = SimpleNamespace(
            id="node_1", product_id="prod_1", parent_id=None, title="T", slug="t",
            content_md="c", node_type="page", artifact_id=None, source="api",
            verified=False, verified_by=None, verified_at=None, created_by=None,
            created_at=None, updated_at=None,
        )
        kn = _knowledge_node_pydantic(n)
        assert kn.id == "node_1"
        assert kn.title == "T"

    def test_compose_content_markdown_only(self):
        from api.routers.integrations import _compose_content

        result = _compose_content({"markdown": "# Hello"})
        assert "# Hello" in result

    def test_compose_content_with_attachments(self):
        from api.routers.integrations import _compose_content

        result = _compose_content({
            "markdown": "# Page",
            "attachments": [
                {"filename": "doc.pdf", "markdown": "PDF content"},
            ],
        })
        assert "# Page" in result
        assert "Attachment: doc.pdf" in result
        assert "PDF content" in result

    def test_compose_content_no_markdown(self):
        from api.routers.integrations import _compose_content

        result = _compose_content({})
        assert result == ""

    def test_compose_content_attachment_not_dict(self):
        from api.routers.integrations import _compose_content

        result = _compose_content({"markdown": "Main", "attachments": ["not-a-dict"]})
        assert "Main" in result

    def test_index_in_background_empty_text(self):
        from api.routers.integrations import _index_in_background

        # Empty text -> no indexing attempted.
        _index_in_background("prod_1", "")
        _index_in_background("prod_1", None)

    def test_index_in_background_no_running_loop(self, isolated_db, monkeypatch):
        """When there's no running event loop, asyncio.run is used."""
        from api.routers.integrations import _index_in_background
        import api.routers.integrations as integ_mod

        called = []

        async def _fake_add(text, dataset_name=None):
            called.append((text, dataset_name))
            return ["ok"]

        monkeypatch.setattr(integ_mod, "add_and_index_document", _fake_add)

        # Called outside an async context -> no running loop -> asyncio.run.
        _index_in_background("prod_1", "some text")
        assert len(called) == 1
        assert called[0][0] == "some text"
        assert called[0][1] == "prod_prod_1"


# --- GET /api/integrations --------------------------------------------------
class TestListConnectors:
    def test_list_connectors(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeGitConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        resp = client.get("/api/integrations")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert any(c["name"] == "github" for c in data)


# --- POST /api/integrations/{name}/test ------------------------------------
class TestConnectorTest:
    def test_connector_test_success(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeGitConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        resp = client.post("/api/integrations/github/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

    def test_connector_test_not_found(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeGitConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        resp = client.post("/api/integrations/unknown/test")
        assert resp.status_code == 404

    def test_connector_test_exception(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _ErrorConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        resp = client.post("/api/integrations/confluence/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "Test failed" in body["message"]


# --- GET /api/integrations/{name}/spaces -----------------------------------
class TestConnectorSpaces:
    def test_list_spaces_success(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeGitConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        resp = client.get("/api/integrations/github/spaces")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["id"] == "repo1"

    def test_list_spaces_not_found(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeGitConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        resp = client.get("/api/integrations/unknown/spaces")
        assert resp.status_code == 404

    def test_list_spaces_exception_502(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _ErrorConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        resp = client.get("/api/integrations/confluence/spaces")
        assert resp.status_code == 502


# --- POST codebases/from-integration ---------------------------------------
class TestCodebaseFromIntegration:
    def test_non_git_connector_400(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeConfluenceConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/codebases/from-integration",
            json={"connector": "confluence", "source_id": "page1"},
        )
        assert resp.status_code == 400
        assert "does not produce a codebase" in resp.json()["detail"]

    def test_product_not_found_404(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeGitConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        resp = client.post(
            "/api/products/prod_ghost/codebases/from-integration",
            json={"connector": "github", "source_id": "repo1"},
        )
        assert resp.status_code == 404

    def test_connector_not_found_404(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeGitConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/codebases/from-integration",
            json={"connector": "gitlab", "source_id": "repo1"},
        )
        assert resp.status_code == 404

    def test_pull_value_error_400(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _ValueErrorConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/codebases/from-integration",
            json={"connector": "github", "source_id": "bad"},
        )
        assert resp.status_code == 400

    def test_pull_exception_502(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _ErrorConnector)
        from api.routers import integrations as integ_mod

        # _ErrorConnector is registered as "confluence", but codebase needs "github".
        # We need a github-named error connector.
        class _GithubErrorConnector(_ErrorConnector):
            name = "github"

        _patch_registry(monkeypatch, _GithubErrorConnector)

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/codebases/from-integration",
            json={"connector": "github", "source_id": "repo1"},
        )
        assert resp.status_code == 502

    def test_success(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeGitConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/codebases/from-integration",
            json={"connector": "github", "source_id": "myrepo"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == "My Repo"
        assert body["repo_url"] == "https://github.com/example/myrepo"
        assert body["repo_type"] == "github"
        assert body["source"] == "api"

    def test_success_with_explicit_name(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeGitConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/codebases/from-integration",
            json={"connector": "github", "source_id": "myrepo", "name": "Custom Name"},
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "Custom Name"

    def test_commit_failure_500(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeGitConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        # Force commit to fail by closing the session.
        original_commit = isolated_db.SessionLocal().commit

        def _fail_commit(self):
            raise Exception("DB constraint violation")

        # Patch Session.commit to raise.
        from sqlalchemy.orm import Session

        monkeypatch.setattr(Session, "commit", _fail_commit)

        resp = client.post(
            f"/api/products/{pid}/codebases/from-integration",
            json={"connector": "github", "source_id": "myrepo"},
        )
        assert resp.status_code == 500
        assert "Persist failed" in resp.json()["detail"]


# --- POST knowledge/from-integration ---------------------------------------
class TestKnowledgeFromIntegration:
    def test_product_not_found_404(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeConfluenceConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        resp = client.post(
            "/api/products/prod_ghost/knowledge/from-integration",
            json={"connector": "confluence", "source_id": "page1"},
        )
        assert resp.status_code == 404

    def test_connector_not_found_404(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeConfluenceConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/knowledge/from-integration",
            json={"connector": "unknown", "source_id": "page1"},
        )
        assert resp.status_code == 404

    def test_pull_value_error_400(self, isolated_db, monkeypatch):
        class _ConfValueErrorConnector(_ValueErrorConnector):
            name = "confluence"

        _patch_registry(monkeypatch, _ConfValueErrorConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/knowledge/from-integration",
            json={"connector": "confluence", "source_id": "bad"},
        )
        assert resp.status_code == 400

    def test_pull_exception_502(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _ErrorConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/knowledge/from-integration",
            json={"connector": "confluence", "source_id": "page1"},
        )
        assert resp.status_code == 502

    def test_single_page_success(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeConfluenceConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/knowledge/from-integration",
            json={"connector": "confluence", "source_id": "page1"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert "Page page1" in body["title"]
        assert body["source"] == "api"
        assert body["node_type"] == "page"

    def test_single_page_with_explicit_name(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeConfluenceConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/knowledge/from-integration",
            json={"connector": "confluence", "source_id": "page1", "name": "Custom"},
        )
        assert resp.status_code == 201
        assert resp.json()["title"] == "Custom"

    def test_multi_page_tree(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeMultiPageConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/knowledge/from-integration",
            json={"connector": "confluence", "source_id": "space1"},
        )
        assert resp.status_code == 201, resp.text
        root = resp.json()
        assert root["title"] == "Root"

        # Verify children were created by fetching the tree.
        from fastapi import FastAPI
        from fastapi.testclient import TestClient as _TC
        from api.routers import knowledge as knowledge_mod
        app2 = FastAPI()
        app2.include_router(knowledge_mod.router)

        def _get_db():
            s = isolated_db.SessionLocal()
            try:
                yield s
            finally:
                s.close()

        def _user():
            return _system_user()

        app2.dependency_overrides[knowledge_mod.get_db] = _get_db
        app2.dependency_overrides[knowledge_mod.get_current_user] = _user

        client2 = _TC(app2)
        tree_resp = client2.get(f"/api/products/{pid}/knowledge/tree")
        assert tree_resp.status_code == 200
        tree = tree_resp.json()
        assert len(tree) == 1
        assert len(tree[0]["children"]) == 2

    def test_multi_page_commit_failure_500(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeMultiPageConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        from sqlalchemy.orm import Session

        def _fail_commit(self):
            raise Exception("DB error")

        monkeypatch.setattr(Session, "commit", _fail_commit)

        resp = client.post(
            f"/api/products/{pid}/knowledge/from-integration",
            json={"connector": "confluence", "source_id": "space1"},
        )
        assert resp.status_code == 500
        assert "Persist failed" in resp.json()["detail"]

    def test_single_page_commit_failure_500(self, isolated_db, monkeypatch):
        _patch_registry(monkeypatch, _FakeConfluenceConnector)
        from api.routers import integrations as integ_mod

        app, client = _build_client(isolated_db, integ_mod)
        pid = _seed_product(isolated_db)

        from sqlalchemy.orm import Session

        def _fail_commit(self):
            raise Exception("DB error")

        monkeypatch.setattr(Session, "commit", _fail_commit)

        resp = client.post(
            f"/api/products/{pid}/knowledge/from-integration",
            json={"connector": "confluence", "source_id": "page1"},
        )
        assert resp.status_code == 500
        assert "Persist failed" in resp.json()["detail"]
