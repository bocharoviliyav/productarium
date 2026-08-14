"""Unit tests for api.routers.public — gaps not covered by
tests/integration/test_admin_public.py.

Focuses on:
- ``list_public_products`` (list products)
- ``export_knowledge`` with invalid format (400)
- ``_default_push_target`` (confluence configured, git configured, neither)
- ``ask`` success path (SSE stream with mocked run_expert_chat)
- ``ask`` product-not-found 404
- ``push`` success with a connector that supports push/export
- ``push`` product-not-found 404
- ``push`` connector not registered (501)
- ``push`` connector has no push/export method (501)
- ``push`` push_fn raises (502)
- ``push`` push_fn returns awaitable (async)
- ``_verified_meta`` with kind + verified_by
- ``_knowledge_as_json`` / ``_knowledge_as_markdown`` with specs + links
- ``_load_verified`` helper
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
def _api_token():
    from api.models import ApiTokenORM

    return ApiTokenORM(
        id="tok_fixed",
        user_id="user_admin1",
        token_hash="x" * 64,
        name="fixed",
        created_at=datetime.utcnow(),
    )


def _build_client(db_mod, public_mod, *, with_token=True):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.auth import deps as auth_deps

    app = FastAPI()
    app.include_router(public_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[public_mod.get_db] = _get_test_db
    app.dependency_overrides[auth_deps.get_db] = _get_test_db
    if with_token:
        app.dependency_overrides[auth_deps.require_api_token] = _api_token
    return app, TestClient(app)


def _seed_product_with_verified(db_mod, product_id="prod_1"):
    from api.models import (
        CodebaseORM,
        KnowledgeNodeORM,
        LinksORM,
        ProductORM,
        SpecORM,
    )

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id=product_id, name="Acme", summary="A summary"))
        db.flush()
        db.add(CodebaseORM(
            id="cb_v", product_id=product_id, name="svc-v",
            verified=True, generated_docs="Codebase docs",
            source="manual", verified_by="admin",
        ))
        db.add(SpecORM(
            id="spec_v", product_id=product_id, name="openapi-v",
            kind="openapi", verified=True, content="openapi: 3.0",
            source="manual", verified_by="admin",
        ))
        db.add(LinksORM(
            id="link_v", product_id=product_id, name="links-v",
            verified=True, content='[{"url":"http://x","description":"d"}]',
            source="manual", verified_by="admin",
        ))
        db.add(KnowledgeNodeORM(
            id="node_v", product_id=product_id, title="Page V", slug="page-v",
            content_md="Node content", verified=True, source="manual",
        ))
        db.commit()
    return product_id


def _seed_product(db_mod, product_id="prod_1"):
    from api.models import ProductORM

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id=product_id, name="Acme"))
        db.commit()
    return product_id


# --- Helper unit tests ------------------------------------------------------
class TestHelpers:
    def test_verified_meta_with_kind_and_verified_by(self):
        from api.routers.public import _verified_meta

        e = SimpleNamespace(
            __class__=type("CodebaseORM", (), {}),
            kind="openapi",
            verified_by="admin",
        )
        # type(e).__name__ would be "CodebaseORM" -> lower + strip "ORM"
        # We need to set it up properly.
        class FakeORM:
            pass

        e = FakeORM()
        e.kind = "openapi"
        e.verified_by = "admin"
        meta = _verified_meta(e)
        assert "> fake | kind: openapi | verified_by: admin" == meta

    def test_verified_meta_no_kind_no_verified_by(self):
        from api.routers.public import _verified_meta

        class CodebaseORM:
            pass

        e = CodebaseORM()
        e.kind = None
        e.verified_by = None
        meta = _verified_meta(e)
        assert meta == "> codebase"

    def test_knowledge_as_json_with_all_types(self):
        from api.routers.public import _knowledge_as_json

        product = SimpleNamespace(id="p1", name="Acme", summary="sum")
        cb = SimpleNamespace(
            id="cb1", name="CB", generated_docs="docs",
            verified_by="admin", verified_at=datetime(2024, 1, 1),
            source="manual",
        )
        spec = SimpleNamespace(
            id="s1", name="S", kind="openapi", content="spec",
            verified_by="admin", verified_at=datetime(2024, 1, 1),
            source="manual",
        )
        link = SimpleNamespace(
            id="l1", name="L", content='[{"url":"http://x"}]',
            verified_by="admin", verified_at=datetime(2024, 1, 1),
            source="manual",
        )
        node = SimpleNamespace(
            id="n1", parent_id=None, title="T", slug="t",
            node_type="page", content_md="c",
            verified_by="admin", verified_at=datetime(2024, 1, 1),
            source="manual",
        )
        result = _knowledge_as_json(product, [cb], [spec], [link], [node])
        assert result["product"]["id"] == "p1"
        assert result["verified_only"] is True
        assert len(result["codebases"]) == 1
        assert result["codebases"][0]["generated_docs"] == "docs"
        assert len(result["specs"]) == 1
        assert result["specs"][0]["kind"] == "openapi"
        assert len(result["links"]) == 1
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["slug"] == "t"

    def test_knowledge_as_markdown_with_all_types(self):
        from api.routers.public import _knowledge_as_markdown

        product = SimpleNamespace(id="p1", name="Acme", summary="A summary")

        class CodebaseORM:
            pass

        cb = CodebaseORM()
        cb.id = "cb1"
        cb.name = "CB"
        cb.kind = None
        cb.verified_by = "admin"
        cb.generated_docs = "CB docs"

        class SpecORM:
            pass

        spec = SpecORM()
        spec.id = "s1"
        spec.name = "Spec"
        spec.kind = "openapi"
        spec.verified_by = "admin"
        spec.content = "spec content"

        class LinksORM:
            pass

        link = LinksORM()
        link.id = "l1"
        link.name = "Links"
        link.kind = None
        link.verified_by = None
        link.content = "link content"

        node = SimpleNamespace(
            id="n1", parent_id=None, title="Page V",
            content_md="Node content",
        )

        md = _knowledge_as_markdown(product, [cb], [spec], [link], [node])
        assert "Verified Knowledge" in md
        assert "A summary" in md
        assert "CB docs" in md
        assert "spec content" in md
        assert "link content" in md
        assert "Node content" in md
        assert "Specifications" in md
        assert "Links" in md
        assert "Knowledge Pages" in md

    def test_load_verified(self, isolated_db):
        from api.routers.public import _load_verified

        _seed_product_with_verified(isolated_db)
        cbs, specs, links, nodes = _load_verified("prod_1", isolated_db.SessionLocal())
        assert len(cbs) == 1
        assert len(specs) == 1
        assert len(links) == 1
        assert len(nodes) == 1

    def test_load_verified_empty(self, isolated_db):
        from api.routers.public import _load_verified

        _seed_product(isolated_db)
        cbs, specs, links, nodes = _load_verified("prod_1", isolated_db.SessionLocal())
        assert cbs == []
        assert specs == []
        assert links == []
        assert nodes == []


# --- _default_push_target ---------------------------------------------------
class TestDefaultPushTarget:
    def test_confluence_configured(self, monkeypatch):
        from api.routers import public as public_mod

        monkeypatch.setattr(
            public_mod, "get_confluence_creds",
            lambda: {"base_url": "http://conf", "token": "tok"},
        )
        monkeypatch.setattr(public_mod, "get_git_creds", lambda host: {})
        assert public_mod._default_push_target() == "confluence"

    def test_git_configured(self, monkeypatch):
        from api.routers import public as public_mod

        monkeypatch.setattr(
            public_mod, "get_confluence_creds",
            lambda: {},
        )

        def _git_creds(host):
            if host == "github":
                return {"url": "http://github", "token": "tok"}
            return {}

        monkeypatch.setattr(public_mod, "get_git_creds", _git_creds)
        assert public_mod._default_push_target() == "github"

    def test_neither_configured_defaults_confluence(self, monkeypatch):
        from api.routers import public as public_mod

        monkeypatch.setattr(public_mod, "get_confluence_creds", lambda: {})
        monkeypatch.setattr(public_mod, "get_git_creds", lambda host: {})
        assert public_mod._default_push_target() == "confluence"


# --- GET /api/public/products -----------------------------------------------
class TestListPublicProducts:
    def test_list_products(self, isolated_db):
        from api.routers import public as public_mod

        _seed_product(isolated_db, "prod_1")
        _seed_product(isolated_db, "prod_2")

        app, client = _build_client(isolated_db, public_mod)
        resp = client.get("/api/public/products")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        ids = [p["id"] for p in data]
        assert "prod_1" in ids
        assert "prod_2" in ids

    def test_list_products_empty(self, isolated_db):
        from api.routers import public as public_mod

        app, client = _build_client(isolated_db, public_mod)
        resp = client.get("/api/public/products")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_products_no_token_401(self, isolated_db):
        from api.routers import public as public_mod

        app, client = _build_client(isolated_db, public_mod, with_token=False)
        resp = client.get("/api/public/products")
        assert resp.status_code == 401


# --- GET /api/public/products/{id}/knowledge --------------------------------
class TestExportKnowledge:
    def test_invalid_format_400(self, isolated_db):
        from api.routers import public as public_mod

        _seed_product(isolated_db)
        app, client = _build_client(isolated_db, public_mod)
        resp = client.get("/api/public/products/prod_1/knowledge?format=xml")
        assert resp.status_code == 400
        assert "format must be" in resp.json()["detail"]

    def test_json_export_with_specs_and_links(self, isolated_db):
        from api.routers import public as public_mod

        _seed_product_with_verified(isolated_db)
        app, client = _build_client(isolated_db, public_mod)
        resp = client.get("/api/public/products/prod_1/knowledge?format=json")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["specs"]) == 1
        assert data["specs"][0]["kind"] == "openapi"
        assert len(data["links"]) == 1

    def test_markdown_export_with_specs_and_links(self, isolated_db):
        from api.routers import public as public_mod

        _seed_product_with_verified(isolated_db)
        app, client = _build_client(isolated_db, public_mod)
        resp = client.get("/api/public/products/prod_1/knowledge")
        assert resp.status_code == 200
        md = resp.text
        assert "Specifications" in md
        assert "Links" in md
        assert "openapi: 3.0" in md

    def test_export_product_not_found_404(self, isolated_db):
        from api.routers import public as public_mod

        app, client = _build_client(isolated_db, public_mod)
        resp = client.get("/api/public/products/ghost/knowledge")
        assert resp.status_code == 404


# --- POST /api/public/products/{id}/ask -------------------------------------
class TestAsk:
    def test_ask_product_not_found_404(self, isolated_db):
        from api.routers import public as public_mod

        app, client = _build_client(isolated_db, public_mod)
        resp = client.post(
            "/api/public/products/ghost/ask",
            json={"query": "hello"},
        )
        assert resp.status_code == 404

    def test_ask_success_stream(self, isolated_db, monkeypatch):
        from api.routers import public as public_mod

        _seed_product(isolated_db)

        async def _fake_stream(*args, **kwargs):
            yield "chunk1"
            yield "chunk2"

        # Patch the lazy import target.
        import api.expert.chat as expert_chat_mod
        monkeypatch.setattr(expert_chat_mod, "run_expert_chat", _fake_stream)

        app, client = _build_client(isolated_db, public_mod)
        resp = client.post(
            "/api/public/products/prod_1/ask",
            json={"query": "hello"},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "chunk1" in body
        assert "chunk2" in body
        assert "[DONE]" in body

    def test_ask_non_async_result(self, isolated_db, monkeypatch):
        """When run_expert_chat returns a coroutine (not async iterator)."""
        from api.routers import public as public_mod

        _seed_product(isolated_db)

        async def _fake_coro(*args, **kwargs):
            return "result text"

        import api.expert.chat as expert_chat_mod
        monkeypatch.setattr(expert_chat_mod, "run_expert_chat", _fake_coro)

        app, client = _build_client(isolated_db, public_mod)
        resp = client.post(
            "/api/public/products/prod_1/ask",
            json={"query": "hello"},
        )
        assert resp.status_code == 200
        assert "result text" in resp.text
        assert "[DONE]" in resp.text


# --- POST /api/public/products/{id}/push ------------------------------------
class TestPush:
    def test_push_product_not_found_404(self, isolated_db):
        from api.routers import public as public_mod

        app, client = _build_client(isolated_db, public_mod)
        resp = client.post(
            "/api/public/products/ghost/push",
            json={"target": "confluence"},
        )
        assert resp.status_code == 404

    def test_push_connector_not_registered_501(self, isolated_db, monkeypatch):
        from api.routers import public as public_mod

        _seed_product(isolated_db)

        # Patch registry get_connector to return None.
        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: None)

        app, client = _build_client(isolated_db, public_mod)
        resp = client.post(
            "/api/public/products/prod_1/push",
            json={"target": "unknown"},
        )
        assert resp.status_code == 501
        assert "No connector registered" in resp.json()["detail"]

    def test_push_no_push_or_export_method_501(self, isolated_db, monkeypatch):
        from api.routers import public as public_mod

        _seed_product(isolated_db)

        class _NoPushConnector:
            def __init__(self, config=None):
                pass

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: _NoPushConnector())

        app, client = _build_client(isolated_db, public_mod)
        resp = client.post(
            "/api/public/products/prod_1/push",
            json={"target": "confluence"},
        )
        assert resp.status_code == 501
        assert "does not support push" in resp.json()["detail"]

    def test_push_success_dict_result(self, isolated_db, monkeypatch):
        from api.routers import public as public_mod

        _seed_product(isolated_db)

        class _PushConnector:
            def __init__(self, config=None):
                pass

            def push(self, payload):
                return {"url": "http://conf/page/123", "id": "page-123"}

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: _PushConnector())

        app, client = _build_client(isolated_db, public_mod)
        resp = client.post(
            "/api/public/products/prod_1/push",
            json={"target": "confluence", "space": "SPACE"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["url"] == "http://conf/page/123"

    def test_push_success_non_dict_result(self, isolated_db, monkeypatch):
        from api.routers import public as public_mod

        _seed_product(isolated_db)

        class _ExportConnector:
            def __init__(self, config=None):
                pass

            def export(self, payload):
                return "pushed"

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: _ExportConnector())

        app, client = _build_client(isolated_db, public_mod)
        resp = client.post(
            "/api/public/products/prod_1/push",
            json={"target": "git"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["target"] == "git"

    def test_push_async_result(self, isolated_db, monkeypatch):
        from api.routers import public as public_mod

        _seed_product(isolated_db)

        class _AsyncPushConnector:
            def __init__(self, config=None):
                pass

            async def push(self, payload):
                return {"async_url": "http://conf/async"}

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: _AsyncPushConnector())

        app, client = _build_client(isolated_db, public_mod)
        resp = client.post(
            "/api/public/products/prod_1/push",
            json={"target": "confluence"},
        )
        assert resp.status_code == 200
        assert resp.json()["async_url"] == "http://conf/async"

    def test_push_exception_502(self, isolated_db, monkeypatch):
        from api.routers import public as public_mod

        _seed_product(isolated_db)

        class _ErrorPushConnector:
            def __init__(self, config=None):
                pass

            def push(self, payload):
                raise RuntimeError("Push failed")

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", lambda name: _ErrorPushConnector())

        app, client = _build_client(isolated_db, public_mod)
        resp = client.post(
            "/api/public/products/prod_1/push",
            json={"target": "confluence"},
        )
        assert resp.status_code == 502
        assert "Push to 'confluence' failed" in resp.json()["detail"]

    def test_push_uses_default_target(self, isolated_db, monkeypatch):
        """When body.target is None, _default_push_target is used."""
        from api.routers import public as public_mod

        _seed_product(isolated_db)

        called_target = []

        class _PushConnector:
            def __init__(self, config=None):
                pass

            def push(self, payload):
                return {"ok": True}

        def _get_connector(name):
            called_target.append(name)
            return _PushConnector()

        import api.integrations.registry as reg_mod
        monkeypatch.setattr(reg_mod, "get_connector", _get_connector)
        monkeypatch.setattr(
            public_mod, "get_confluence_creds",
            lambda: {"base_url": "http://conf", "token": "tok"},
        )
        monkeypatch.setattr(public_mod, "get_git_creds", lambda host: {})

        app, client = _build_client(isolated_db, public_mod)
        resp = client.post(
            "/api/public/products/prod_1/push",
            json={},
        )
        assert resp.status_code == 200
        assert called_target[0] == "confluence"
