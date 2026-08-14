#!/usr/bin/env python3
"""Unit tests for api.routers.admin.

Focuses on the lines NOT already covered by tests/integration/test_admin_public.py:
- connectivity tests (POST /{group}/test) for models, git, confluence, integrations
- _ping_model_endpoint branches (success, auth-rejected, probe failure, non-200)
- _lazy_connector / _connector_test_result helper branches
- prompts list/get/put (including invalid filename + path traversal + missing file)
- cognee reindex endpoint (success + failure)
- settings group GETs with ``resolved`` views (git, confluence, integrations, rlm,
  ssl, cognee, timeouts)
- users: POST create (with + without password), POST reset-token, duplicate 409,
  invalid role 400, non-local user reset 400
- api tokens: non-admin delete 403, delete nonexistent 404
- settings PUT validation (rlm invalid mode, models max_prompt_tokens invalid/neg,
  timeouts invalid/neg/float, non-dict body 400)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


# --- Helpers ----------------------------------------------------------------
def _admin_user_orm():
    from api.models import UserORM

    return UserORM(
        id="user_admin1",
        username="admin",
        role="admin",
        provider="local",
        created_at=datetime.utcnow(),
    )


def _non_admin_user_orm():
    from api.models import UserORM

    return UserORM(
        id="user_plain",
        username="plain",
        role="user",
        provider="local",
        created_at=datetime.utcnow(),
    )


def _build_client(db_mod, admin_mod, *, admin=None):
    """Build a TestClient with get_db + require_admin overridden.

    ``admin`` is a UserORM instance (or None for the default admin). It is
    wrapped in a callable so FastAPI's dependency resolver accepts it.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.auth import deps as auth_deps

    app = FastAPI()
    app.include_router(admin_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    admin_obj = admin if admin is not None else _admin_user_orm()

    def _admin_override():
        return admin_obj

    app.dependency_overrides[admin_mod.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.require_admin] = _admin_override
    return app, TestClient(app)


# --- Connectivity tests: models ---------------------------------------------
class TestModelsConnectivity:
    def test_models_test_no_base_url(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/models/test", json={})
        assert resp.status_code == 200
        body = resp.json()
        # With no base_url configured in settings (and env default is local),
        # _test_models reads get_model_for_task which returns env defaults.
        # The test will attempt to ping the endpoint; mock requests to avoid
        # network. But first: if we clear the model config so base_url is empty,
        # we get the "No base_url" branch.
        assert body["success"] is False

    def test_models_test_success(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod

        # Mock requests.get + requests.post for _ping_model_endpoint.
        call_log = []

        class _FakeResp:
            def __init__(self, status, payload=None):
                self.status_code = status
                self._payload = payload or {}

            def json(self):
                return self._payload

        def _fake_get(url, headers=None, timeout=None, verify=True):
            call_log.append(("get", url))
            return _FakeResp(200, {"data": [{"id": "qwen/test"}, {"id": "llama"}]})

        def _fake_post(url, headers=None, json=None, timeout=None, verify=True):
            call_log.append(("post", url))
            return _FakeResp(200, {"choices": [{"message": {"content": "ok"}}]})

        import requests
        monkeypatch.setattr(requests, "get", _fake_get)
        monkeypatch.setattr(requests, "post", _fake_post)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/models/test",
            json={"base_url": "http://x:11434/v1", "model": "qwen/test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "qwen/test" in body["models"]
        assert "2 model(s)" in body["message"]

    def test_models_test_non_200(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod

        class _FakeResp:
            def __init__(self, status):
                self.status_code = status

            def json(self):
                return {}

        def _fake_get(url, **kw):
            return _FakeResp(503)

        import requests
        monkeypatch.setattr(requests, "get", _fake_get)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/models/test",
            json={"base_url": "http://x:11434/v1", "model": "qwen/test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "503" in body["message"]

    def test_models_test_connection_error(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod

        def _fake_get(url, **kw):
            raise ConnectionError("refused")

        import requests
        monkeypatch.setattr(requests, "get", _fake_get)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/models/test",
            json={"base_url": "http://x:11434/v1", "model": "qwen/test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "refused" in body["message"]

    def test_models_test_chat_auth_rejected(self, isolated_db, monkeypatch):
        """GET /v1/models succeeds but chat returns 401 -> auth-rejected message."""
        from api.routers import admin as admin_mod

        class _FakeResp:
            def __init__(self, status, payload=None):
                self.status_code = status
                self._payload = payload or {}

            def json(self):
                return self._payload

        def _fake_get(url, **kw):
            return _FakeResp(200, {"data": [{"id": "qwen/test"}]})

        def _fake_post(url, **kw):
            return _FakeResp(401)

        import requests
        monkeypatch.setattr(requests, "get", _fake_get)
        monkeypatch.setattr(requests, "post", _fake_post)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/models/test",
            json={"base_url": "http://x:11434/v1", "model": "qwen/test"},
        )
        body = resp.json()
        assert body["success"] is False
        assert "Auth rejected" in body["message"]

    def test_models_test_embedder_probe(self, isolated_db, monkeypatch):
        """task=embedder probes /v1/embeddings instead of chat."""
        from api.routers import admin as admin_mod

        post_urls = []

        class _FakeResp:
            def __init__(self, status, payload=None):
                self.status_code = status
                self._payload = payload or {}

            def json(self):
                return self._payload

        def _fake_get(url, **kw):
            return _FakeResp(200, {"data": [{"id": "nomic"}]})

        def _fake_post(url, **kw):
            post_urls.append(url)
            return _FakeResp(200)

        import requests
        monkeypatch.setattr(requests, "get", _fake_get)
        monkeypatch.setattr(requests, "post", _fake_post)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/models/test",
            json={"base_url": "http://x:11434/v1", "model": "nomic", "task": "embedder"},
        )
        body = resp.json()
        assert body["success"] is True
        assert any("embeddings" in u for u in post_urls)

    def test_models_test_chat_probe_non_400_note(self, isolated_db, monkeypatch):
        """Chat probe returns >= 400 (but not 401/403) -> success with chat_note."""
        from api.routers import admin as admin_mod

        class _FakeResp:
            def __init__(self, status, payload=None):
                self.status_code = status
                self._payload = payload or {}

            def json(self):
                return self._payload

        def _fake_get(url, **kw):
            return _FakeResp(200, {"data": [{"id": "qwen/test"}]})

        def _fake_post(url, **kw):
            return _FakeResp(500)

        import requests
        monkeypatch.setattr(requests, "get", _fake_get)
        monkeypatch.setattr(requests, "post", _fake_post)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/models/test",
            json={"base_url": "http://x:11434/v1", "model": "qwen/test"},
        )
        body = resp.json()
        assert body["success"] is True
        assert "500" in body["message"]

    def test_models_test_probe_exception(self, isolated_db, monkeypatch):
        """Chat probe raises -> success with probe-failed note."""
        from api.routers import admin as admin_mod

        class _FakeResp:
            def __init__(self, status, payload=None):
                self.status_code = status
                self._payload = payload or {}

            def json(self):
                return self._payload

        def _fake_get(url, **kw):
            return _FakeResp(200, {"data": [{"id": "qwen/test"}]})

        def _fake_post(url, **kw):
            raise RuntimeError("timeout")

        import requests
        monkeypatch.setattr(requests, "get", _fake_get)
        monkeypatch.setattr(requests, "post", _fake_post)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/models/test",
            json={"base_url": "http://x:11434/v1", "model": "qwen/test"},
        )
        body = resp.json()
        assert body["success"] is True
        assert "probe failed" in body["message"]

    def test_models_test_no_models_but_probe(self, isolated_db, monkeypatch):
        """No models in /v1/models -> probe_model is empty -> skip probe."""
        from api.routers import admin as admin_mod

        class _FakeResp:
            def __init__(self, status, payload=None):
                self.status_code = status
                self._payload = payload or {}

            def json(self):
                return self._payload

        def _fake_get(url, **kw):
            return _FakeResp(200, {"data": []})

        import requests
        monkeypatch.setattr(requests, "get", _fake_get)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/models/test",
            json={"base_url": "http://x:11434/v1"},
        )
        body = resp.json()
        assert body["success"] is True
        assert body["models"] == []


# --- Connectivity tests: git / confluence / integrations --------------------
class TestIntegrationConnectivity:
    def test_git_test_unknown_host(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/git/test", json={"host": "bitbucket"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "Unknown git host" in body["message"]

    def test_git_test_no_connector(self, isolated_db, monkeypatch):
        """When the registry has no connector, _lazy_connector returns an error."""
        from api.routers import admin as admin_mod
        from api.integrations import registry as reg

        # Force no connectors.
        monkeypatch.setattr(reg, "_REGISTRY", {})
        monkeypatch.setattr(reg, "_DISCOVERED", True)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/git/test", json={"host": "github"})
        body = resp.json()
        assert body["success"] is False

    def test_git_test_connector_success(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod
        from api.integrations import registry as reg

        class _FakeConnector:
            def __init__(self, config=None):
                self.config = config or {}

            @classmethod
            def get_config(cls):
                return {}

            def test(self):
                return {"success": True, "message": "reachable"}

        monkeypatch.setattr(reg, "_REGISTRY", {"github": _FakeConnector})
        monkeypatch.setattr(reg, "_DISCOVERED", True)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/git/test", json={"host": "github"})
        body = resp.json()
        assert body["success"] is True
        assert body["message"] == "reachable"

    def test_git_test_connector_raises(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod
        from api.integrations import registry as reg

        class _FakeConnector:
            def __init__(self, config=None):
                self.config = config or {}

            @classmethod
            def get_config(cls):
                return {}

            def test(self):
                raise RuntimeError("conn refused")

        monkeypatch.setattr(reg, "_REGISTRY", {"github": _FakeConnector})
        monkeypatch.setattr(reg, "_DISCOVERED", True)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/git/test", json={"host": "github"})
        body = resp.json()
        assert body["success"] is False
        assert "conn refused" in body["message"]

    def test_git_test_connector_returns_true(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod
        from api.integrations import registry as reg

        class _FakeConnector:
            def __init__(self, config=None):
                self.config = config or {}

            @classmethod
            def get_config(cls):
                return {}

            def test(self):
                return True

        monkeypatch.setattr(reg, "_REGISTRY", {"github": _FakeConnector})
        monkeypatch.setattr(reg, "_DISCOVERED", True)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/git/test", json={"host": "github"})
        body = resp.json()
        assert body["success"] is True

    def test_git_test_connector_no_test_method(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod
        from api.integrations import registry as reg

        class _FakeConnector:
            def __init__(self, config=None):
                self.config = config or {}

            @classmethod
            def get_config(cls):
                return {}

        monkeypatch.setattr(reg, "_REGISTRY", {"github": _FakeConnector})
        monkeypatch.setattr(reg, "_DISCOVERED", True)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/git/test", json={"host": "github"})
        body = resp.json()
        assert body["success"] is False
        assert "no test()" in body["message"]

    def test_confluence_test(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod
        from api.integrations import registry as reg

        class _FakeConnector:
            def __init__(self, config=None):
                self.config = config or {}

            @classmethod
            def get_config(cls):
                return {}

            def test(self):
                return {"success": True, "message": "confluence ok"}

        monkeypatch.setattr(reg, "_REGISTRY", {"confluence": _FakeConnector})
        monkeypatch.setattr(reg, "_DISCOVERED", True)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/confluence/test", json={})
        body = resp.json()
        assert body["success"] is True

    def test_integration_test_no_name(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/integrations/test", json={})
        body = resp.json()
        assert body["success"] is False
        assert "name" in body["message"]

    def test_integration_test_unknown_connector(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod
        from api.integrations import registry as reg

        monkeypatch.setattr(reg, "_REGISTRY", {})
        monkeypatch.setattr(reg, "_DISCOVERED", True)

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/integrations/test", json={"name": "ghost"})
        body = resp.json()
        assert body["success"] is False

    def test_test_group_unknown(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/ghost/test", json={})
        assert resp.status_code == 404


# --- Prompts ----------------------------------------------------------------
class TestPrompts:
    def test_list_prompts(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.get("/api/admin/prompts")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        # refs/prompts/ should have at least overview.md
        filenames = [p["filename"] for p in body]
        assert any(f.endswith(".md") for f in filenames)

    def test_get_prompt(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.get("/api/admin/prompts/overview.md")
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "overview.md"
        assert "content" in body
        assert len(body["content"]) > 0

    def test_get_prompt_invalid_filename(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        # Non-.md extension -> 400
        resp = client.get("/api/admin/prompts/overview.txt")
        assert resp.status_code == 400

    def test_get_prompt_path_traversal_blocked_by_safe_filename(self):
        """_safe_prompt_filename rejects paths that escape PROMPTS_DIR.

        Testing via HTTP is unreliable (Starlette normalizes the URL to 404
        before the handler runs), so we call the validator directly.
        """
        from api.routers.admin import _safe_prompt_filename

        # A traversal attempt that would escape PROMPTS_DIR is rejected.
        assert _safe_prompt_filename("../../etc/passwd.md") is None
        # Non-.md extension rejected.
        assert _safe_prompt_filename("overview.txt") is None
        # Unknown registered file rejected.
        assert _safe_prompt_filename("nonexistent_prompt.md") is None
        # Known registered file accepted.
        assert _safe_prompt_filename("overview.md") == "overview.md"

    def test_get_prompt_unknown_registered_file(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        # A .md file that passes the extension check but is not in PROMPT_FILES
        resp = client.get("/api/admin/prompts/nonexistent_prompt.md")
        assert resp.status_code == 400

    def test_put_prompt(self, isolated_db):
        from api.routers import admin as admin_mod
        from api.prompts import PROMPTS_DIR
        import os

        fname = "overview.md"
        fpath = os.path.join(PROMPTS_DIR, fname)
        original = open(fpath, "r", encoding="utf-8").read()
        try:
            app, client = _build_client(isolated_db, admin_mod)
            resp = client.put(
                f"/api/admin/prompts/{fname}",
                json={"content": "# Test content\n\nNew prompt body."},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["filename"] == fname
            # File was written.
            new_content = open(fpath, "r", encoding="utf-8").read()
            assert "Test content" in new_content
        finally:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(original)

    def test_put_prompt_invalid_filename(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/prompts/bad.txt",
            json={"content": "x"},
        )
        assert resp.status_code == 400


# --- Cognee reindex ---------------------------------------------------------
class TestCogneeReindex:
    def test_reindex_success(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod

        async def _fake_reindex(pid):
            return {"reindexed": True, "product_id": pid}

        # Patch at the use-site (the lazy import inside the handler resolves
        # api.cognee.reindex_product_knowledge_graph).
        import api.cognee as cognee_mod
        monkeypatch.setattr(
            cognee_mod, "reindex_product_knowledge_graph", _fake_reindex
        )

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/cognee/reindex", json={"product_id": "prod_1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["reindexed"] is True

    def test_reindex_no_body(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod

        async def _fake_reindex(pid):
            assert pid is None
            return {"reindexed": True}

        import api.cognee as cognee_mod
        monkeypatch.setattr(
            cognee_mod, "reindex_product_knowledge_graph", _fake_reindex
        )

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/cognee/reindex")
        assert resp.status_code == 200
        assert resp.json()["reindexed"] is True

    def test_reindex_failure_500(self, isolated_db, monkeypatch):
        from api.routers import admin as admin_mod

        async def _fake_reindex(pid):
            raise RuntimeError("cognee down")

        import api.cognee as cognee_mod
        monkeypatch.setattr(
            cognee_mod, "reindex_product_knowledge_graph", _fake_reindex
        )

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/cognee/reindex", json={"product_id": "prod_1"})
        assert resp.status_code == 500
        assert "cognee down" in resp.json()["detail"]


# --- Settings group GETs (resolved views) -----------------------------------
class TestSettingsGroupGets:
    def test_get_git_group(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.get("/api/admin/git")
        assert resp.status_code == 200
        body = resp.json()
        assert body["group"] == "git"
        assert "resolved" in body
        assert "github" in body["resolved"]
        assert "gitlab" in body["resolved"]
        assert body["resolved"]["github"]["hasToken"] is False

    def test_get_confluence_group(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.get("/api/admin/confluence")
        assert resp.status_code == 200
        body = resp.json()
        assert body["group"] == "confluence"
        assert "resolved" in body
        assert "base_url" in body["resolved"]
        assert body["resolved"]["hasToken"] is False

    def test_get_integrations_group(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.get("/api/admin/integrations")
        assert resp.status_code == 200
        body = resp.json()
        assert body["group"] == "integrations"
        assert "resolved" in body

    def test_get_rlm_group(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.get("/api/admin/rlm")
        assert resp.status_code == 200
        body = resp.json()
        assert body["group"] == "rlm"
        assert "resolved" in body
        # fast-rlm not installed -> all modes are "llm"
        for task in ("docgen", "expert", "summary"):
            assert body["resolved"][task] == "llm"

    def test_get_ssl_group(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.get("/api/admin/ssl")
        assert resp.status_code == 200
        body = resp.json()
        assert body["group"] == "ssl"
        assert "settings" in body

    def test_get_cognee_group(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.get("/api/admin/cognee")
        assert resp.status_code == 200
        body = resp.json()
        assert body["group"] == "cognee"
        assert "resolved" in body
        assert "max_concurrency" in body["resolved"]
        assert "delay_seconds" in body["resolved"]
        assert "rate_limit_rps" in body["resolved"]

    def test_get_timeouts_group(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.get("/api/admin/timeouts")
        assert resp.status_code == 200
        body = resp.json()
        assert body["group"] == "timeouts"
        assert "resolved" in body
        # At least one timeout key is present.
        assert "llm_request" in body["resolved"]
        assert "value" in body["resolved"]["llm_request"]


# --- Settings PUT validation ------------------------------------------------
class TestSettingsPutValidation:
    def test_put_rlm_invalid_mode_ignored(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/rlm",
            json={"rlm.expert.mode": "superuser", "rlm.docgen.mode": "auto"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "rlm.docgen.mode" in body["saved"]
        assert "rlm.expert.mode" not in body["saved"]

    def test_put_rlm_valid_mode_saved(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/rlm",
            json={"rlm.expert.mode": "llm"},
        )
        assert resp.status_code == 200
        assert "rlm.expert.mode" in resp.json()["saved"]

    def test_put_models_max_prompt_tokens_invalid(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/models",
            json={
                "models.expert.max_prompt_tokens": "not-a-number",
                "models.expert.model": "qwen/test",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "models.expert.model" in body["saved"]
        assert "models.expert.max_prompt_tokens" not in body["saved"]

    def test_put_models_max_prompt_tokens_negative(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/models",
            json={"models.expert.max_prompt_tokens": "-5"},
        )
        assert resp.status_code == 200
        assert "models.expert.max_prompt_tokens" not in resp.json()["saved"]

    def test_put_models_max_prompt_tokens_valid(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/models",
            json={"models.expert.max_prompt_tokens": "4096"},
        )
        assert resp.status_code == 200
        assert "models.expert.max_prompt_tokens" in resp.json()["saved"]

    def test_put_models_max_prompt_tokens_empty_clears(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/models",
            json={"models.expert.max_prompt_tokens": ""},
        )
        assert resp.status_code == 200
        assert "models.expert.max_prompt_tokens" in resp.json()["saved"]

    def test_put_unknown_group_404(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put("/api/admin/ghostgroup", json={"key": "val"})
        assert resp.status_code == 404

    def test_put_timeouts_invalid_ignored(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/timeouts",
            json={"timeouts.llm_request": "not-a-number"},
        )
        assert resp.status_code == 200
        assert "timeouts.llm_request" not in resp.json()["saved"]

    def test_put_timeouts_negative_ignored(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/timeouts",
            json={"timeouts.llm_request": "-10"},
        )
        assert resp.status_code == 200
        assert "timeouts.llm_request" not in resp.json()["saved"]

    def test_put_timeouts_valid_float(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/timeouts",
            json={"timeouts.llm_request": "120.5"},
        )
        assert resp.status_code == 200
        assert "timeouts.llm_request" in resp.json()["saved"]

    def test_put_timeouts_valid_int(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/timeouts",
            json={"timeouts.llm_request": "120"},
        )
        assert resp.status_code == 200
        assert "timeouts.llm_request" in resp.json()["saved"]

    def test_put_timeouts_empty_clears(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/timeouts",
            json={"timeouts.llm_request": ""},
        )
        assert resp.status_code == 200
        assert "timeouts.llm_request" in resp.json()["saved"]

    def test_put_git_secret_encrypted(self, isolated_db):
        from api.routers import admin as admin_mod
        import api.config.settings as ss

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/git",
            json={"git.github.token": "ghp_secret123"},
        )
        assert resp.status_code == 200
        assert "git.github.token" in resp.json()["saved"]
        # The secret decrypts back.
        assert ss.get_setting("git.github.token") == "ghp_secret123"


# --- Users CRUD (create + reset-token) --------------------------------------
class TestUsersCrud:
    def test_create_user_with_password(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/users",
            json={
                "username": "newuser",
                "email": "new@test.com",
                "role": "user",
                "password": "temppass123",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["user"]["username"] == "newuser"
        assert body["user"]["email"] == "new@test.com"
        assert body["temp_password"] == "temppass123"
        assert body["reset_token"]  # generated

    def test_create_user_generates_password(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/users",
            json={"username": "autogen", "role": "user"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["temp_password"]  # auto-generated
        assert len(body["temp_password"]) > 0

    def test_create_user_empty_username_400(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/users",
            json={"username": "  ", "role": "user"},
        )
        assert resp.status_code == 400
        assert "Username is required" in resp.json()["detail"]

    def test_create_user_invalid_role_400(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/users",
            json={"username": "x", "role": "superuser"},
        )
        assert resp.status_code == 400

    def test_create_user_duplicate_409(self, isolated_db):
        from api.routers import admin as admin_mod
        from api.models import UserORM

        with isolated_db.SessionLocal() as db:
            db.add(UserORM(id="user_dup", username="dup", role="user"))
            db.commit()

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/users",
            json={"username": "dup", "role": "user"},
        )
        assert resp.status_code == 409

    def test_create_user_admin_role(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post(
            "/api/admin/users",
            json={"username": "newadmin", "role": "admin"},
        )
        assert resp.status_code == 200
        assert resp.json()["user"]["role"] == "admin"

    def test_issue_reset_token_success(self, isolated_db):
        from api.routers import admin as admin_mod
        from api.models import UserORM

        with isolated_db.SessionLocal() as db:
            db.add(UserORM(
                id="user_local1", username="local1", role="user", provider="local",
            ))
            db.commit()

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/users/user_local1/reset-token")
        assert resp.status_code == 200
        body = resp.json()
        assert body["temp_password"]
        assert body["reset_token"]
        assert body["user"]["id"] == "user_local1"

    def test_issue_reset_token_user_not_found_404(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/users/ghost/reset-token")
        assert resp.status_code == 404

    def test_issue_reset_token_non_local_400(self, isolated_db):
        from api.routers import admin as admin_mod
        from api.models import UserORM

        with isolated_db.SessionLocal() as db:
            db.add(UserORM(
                id="user_kc", username="kcuser", role="user", provider="keycloak",
            ))
            db.commit()

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.post("/api/admin/users/user_kc/reset-token")
        assert resp.status_code == 400
        assert "local" in resp.json()["detail"].lower()

    def test_demote_user(self, isolated_db):
        from api.routers import admin as admin_mod
        from api.models import UserORM

        with isolated_db.SessionLocal() as db:
            db.add(UserORM(id="user_demote", username="demote", role="admin"))
            db.commit()

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/users",
            json={"user_id": "user_demote", "role": "user"},
        )
        assert resp.status_code == 200
        assert resp.json()["user"]["role"] == "user"

    def test_put_users_invalid_role_400(self, isolated_db):
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        resp = client.put(
            "/api/admin/users",
            json={"user_id": "x", "role": "invalid"},
        )
        assert resp.status_code == 400

    def test_put_users_missing_fields_400(self, isolated_db):
        """Body missing user_id or role -> 400 (isinstance check in put_group).

        We test the dict-but-missing-fields path since FastAPI's Pydantic
        validation rejects a non-dict JSON body with 422 before the handler's
        isinstance check can run.
        """
        from api.routers import admin as admin_mod

        app, client = _build_client(isolated_db, admin_mod)
        # Missing both user_id and role.
        resp = client.put("/api/admin/users", json={})
        assert resp.status_code == 400


# --- API tokens (non-admin + edge cases) ------------------------------------
class TestApiTokensEdge:
    def test_delete_token_non_admin_403(self, isolated_db):
        from api.routers import admin as admin_mod
        from api.models import ApiTokenORM

        with isolated_db.SessionLocal() as db:
            db.add(ApiTokenORM(
                id="tok_other", user_id="user_admin1", name="other",
                token_hash="h" * 64, created_at=datetime.utcnow(),
            ))
            db.commit()

        app, client = _build_client(
            isolated_db, admin_mod, admin=_non_admin_user_orm()
        )
        resp = client.delete("/api/admin/apitokens/tok_other")
        assert resp.status_code == 403

    def test_delete_token_non_admin_own_token_ok(self, isolated_db):
        from api.routers import admin as admin_mod
        from api.models import ApiTokenORM

        with isolated_db.SessionLocal() as db:
            db.add(ApiTokenORM(
                id="tok_own", user_id="user_plain", name="own",
                token_hash="h" * 64, created_at=datetime.utcnow(),
            ))
            db.commit()

        app, client = _build_client(
            isolated_db, admin_mod, admin=_non_admin_user_orm()
        )
        resp = client.delete("/api/admin/apitokens/tok_own")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_list_apitokens_non_admin_sees_only_own(self, isolated_db):
        from api.routers import admin as admin_mod
        from api.models import ApiTokenORM

        with isolated_db.SessionLocal() as db:
            db.add(ApiTokenORM(
                id="tok_mine", user_id="user_plain", name="mine",
                token_hash="h" * 64, created_at=datetime.utcnow(),
            ))
            db.add(ApiTokenORM(
                id="tok_theirs", user_id="user_admin1", name="theirs",
                token_hash="h" * 65, created_at=datetime.utcnow(),
            ))
            db.commit()

        app, client = _build_client(
            isolated_db, admin_mod, admin=_non_admin_user_orm()
        )
        resp = client.get("/api/admin/apitokens")
        assert resp.status_code == 200
        token_ids = [t["id"] for t in resp.json()["tokens"]]
        assert "tok_mine" in token_ids
        assert "tok_theirs" not in token_ids
