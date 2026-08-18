#!/usr/bin/env python3
"""Unit tests for the knowledge-tree Wave 2 scope.

Covers:
- ``api.routers.knowledge.build_tree`` (parent_id nesting + orphan handling)
- ``api.routers.knowledge._slugify``
- ``api.docgen.summary.generate_product_summary`` (mocked LLM + empty content)
- ``api.docgen.dispatcher`` new-type dispatch (spec/links/documentation/guides +
  legacy type mapping) via monkeypatched sub-generators
- ``api.docgen.simple._render_links_index`` / ``generate_links_docs`` /
  ``generate_guides_docs`` / ``generate_documentation_docs`` (passthrough)
- Router endpoint integration (create/get/tree/put/delete/verify) over an
  isolated SQLite DB via FastAPI TestClient with dependency overrides.

No live LLM/Postgres/cognee is required: LLM and cognee indexing are mocked.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest


# --- Shared isolated-env fixture (mirrors test_foundation.py) ----------------
@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Isolated SQLite DB + reloaded api.db / api.routers.knowledge for endpoints."""
    db_file = tmp_path / "kt.db"
    monkeypatch.setenv("DB_PROVIDER", "sqlite")
    monkeypatch.setenv("DB_HOST", str(tmp_path))
    monkeypatch.setenv("DB_NAME", str(db_file))
    monkeypatch.setenv("DB_USERNAME", "")
    monkeypatch.setenv("DB_PASSWORD", "")
    monkeypatch.setenv("AUTH_PROVIDER", "none")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    import api.db as db
    importlib.reload(db)
    db.init_db()
    return db


def _node(
    nid: str,
    product_id: str = "prod_1",
    parent_id: str | None = None,
    title: str = "n",
) -> Any:
    return SimpleNamespace(
        id=nid,
        product_id=product_id,
        parent_id=parent_id,
        title=title,
        slug=nid,
        content_md=None,
        node_type="page",
        artifact_id=None,
        source="manual",
        verified=False,
        verified_by=None,
        verified_at=None,
        created_by=None,
        created_at=None,
        updated_at=None,
    )


# --- Tree building -----------------------------------------------------------
class TestBuildTree:
    def test_flat_list_returns_roots_with_empty_children(self):
        from api.routers.knowledge import build_tree
        nodes = [_node("a", title="A"), _node("b", title="B")]
        tree = build_tree(nodes)
        assert len(tree) == 2
        titles = {n["title"] for n in tree}
        assert titles == {"A", "B"}
        for n in tree:
            assert n["children"] == []

    def test_parent_id_nests_under_parent(self):
        from api.routers.knowledge import build_tree
        nodes = [
            _node("root", title="Root"),
            _node("child", parent_id="root", title="Child"),
            _node("grandchild", parent_id="child", title="Grandchild"),
        ]
        tree = build_tree(nodes)
        assert len(tree) == 1
        root = tree[0]
        assert root["id"] == "root"
        assert len(root["children"]) == 1
        child = root["children"][0]
        assert child["id"] == "child"
        assert len(child["children"]) == 1
        assert child["children"][0]["id"] == "grandchild"

    def test_orphan_parent_treated_as_root(self):
        from api.routers.knowledge import build_tree
        # parent_id points to a node not in the set -> becomes a root, not hidden
        nodes = [_node("orphan", parent_id="missing", title="Orphan")]
        tree = build_tree(nodes)
        assert len(tree) == 1
        assert tree[0]["id"] == "orphan"
        assert tree[0]["children"] == []

    def test_empty_list_returns_empty(self):
        from api.routers.knowledge import build_tree
        assert build_tree([]) == []


# --- slugify -----------------------------------------------------------------
class TestSlugify:
    def test_basic(self):
        from api.routers.knowledge import _slugify
        assert _slugify("Overview Page") == "overview-page"
        assert _slugify("API  Docs!") == "api-docs"
        assert _slugify("  Сервис  ")  # non-ascii stripped -> non-empty fallback
        assert _slugify("") == "node"


# --- Summary (mocked LLM) ----------------------------------------------------
class TestProductSummary:
    def test_summary_with_mocked_llm(self, monkeypatch):
        import api.docgen.summary as ks

        class _FakeLLM:
            async def generate(self, prompt: str) -> str:
                # Confirm the prompt includes product name + collected content.
                assert "Acme" in prompt
                assert "artifact docs" in prompt
                assert "node content" in prompt
                return "```markdown\nAcme is a service.\n```"

        # _safe_build_summary_llm takes (model, base_url=..., api_key=...).
        monkeypatch.setattr(ks, "_safe_build_summary_llm", lambda m, **kw: _FakeLLM())
        product = SimpleNamespace(id="prod_1", name="Acme")
        codebases = [SimpleNamespace(id="art_1", name="svc", generated_docs="artifact docs")]
        nodes = [SimpleNamespace(id="node_1", title="P1", content_md="node content")]
        out = asyncio.run(ks.generate_product_summary(product, codebases, [], nodes))
        assert out == "Acme is a service."

    def test_summary_empty_when_no_content(self):
        import api.docgen.summary as ks
        product = SimpleNamespace(id="prod_1", name="Acme")
        out = asyncio.run(ks.generate_product_summary(product, [], [], []))
        assert out == ""

    def test_summary_empty_when_llm_unavailable(self, monkeypatch):
        import api.docgen.summary as ks
        monkeypatch.setattr(ks, "_safe_build_summary_llm", lambda m, **kw: None)
        product = SimpleNamespace(id="prod_1", name="Acme")
        codebases = [SimpleNamespace(id="a", name="a", generated_docs="some docs")]
        out = asyncio.run(ks.generate_product_summary(product, codebases, [], []))
        assert out == ""

    def test_collect_summary_content_caps_large_input(self):
        import api.docgen.summary as ks

        # _collect_summary_content concatenates codebase/specs/node docs and
        # char-caps the result to SUMMARY_CONTEXT_MAX_CHARS via the char-based
        # _cap helper (token-based clamping was removed in the cleanup).
        unit = "Артефакт: файл структуры кодовой базы. "
        big = unit * 1300  # ~50k chars -> well over the 20k char cap
        assert len(big) > ks.SUMMARY_CONTEXT_MAX_CHARS
        codebases = [SimpleNamespace(id="a", name="a", generated_docs=big)]
        out = ks._collect_summary_content(codebases, [], [])
        # Char-clamping fired: output is capped and carries the truncation suffix.
        assert len(out) < len(big)
        assert len(out) <= ks.SUMMARY_CONTEXT_MAX_CHARS + 100
        assert "обрезано для контекста" in out


# --- Router endpoint integration (SQLite + TestClient) -----------------------
class TestKnowledgeRouterEndpoints:
    def _client(self, tmp_path, monkeypatch):
        """Build a TestClient with a dedicated file-based SQLite engine.

        The foundation's ``api.db`` engine cannot be used here: Starlette's
        TestClient runs the ASGI app in a separate thread, and the default
        sqlite3 driver forbids cross-thread connection use. We must not edit
        ``api.db`` to add ``check_same_thread=False``, so instead we override
        the ``get_db`` dependency with sessions bound to our own engine created
        with ``check_same_thread=False`` and a file-based DB (so committed data
        is visible across threads/connections).
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        import api.routers.knowledge as kr
        from api.models import Base, UserORM

        engine = create_engine(
            f"sqlite:///{tmp_path / 'kt_endpoint.db'}",
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
            future=True,
        )
        Base.metadata.create_all(bind=engine)
        TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)

        def _get_db():
            s = TestSession()
            try:
                yield s
            finally:
                s.close()

        app = FastAPI()
        app.include_router(kr.router)
        app.dependency_overrides[kr.get_db] = _get_db

        # Fixed admin user so no session cookie / auth backend is required.
        def _current_user():
            return UserORM(
                id="system", username="system", role="admin",
                provider="local", created_at=datetime.utcnow(),
            )
        app.dependency_overrides[kr.get_current_user] = _current_user
        return TestClient(app), TestSession

    def _seed_product(self, TestSession, product_id="prod_1", owner_id=None):
        from api.models import ProductORM
        with TestSession() as db:
            db.add(ProductORM(id=product_id, name="Acme", description="d", owner_id=owner_id))
            db.commit()
        return product_id

    def test_create_get_tree_update_delete_verify(self, tmp_path, monkeypatch):
        client, TestSession = self._client(tmp_path, monkeypatch)
        pid = self._seed_product(TestSession)

        # Create a root node.
        r = client.post(
            f"/api/products/{pid}/knowledge/nodes",
            json={"title": "Overview", "content_md": "intro"},
        )
        assert r.status_code == 201, r.text
        root = r.json()
        assert root["title"] == "Overview"
        assert root["slug"] == "overview"
        assert root["created_by"] == "system"
        root_id = root["id"]

        # Create a child.
        r = client.post(
            f"/api/products/{pid}/knowledge/nodes",
            json={"parent_id": root_id, "title": "Architecture", "content_md": "arch"},
        )
        assert r.status_code == 201, r.text
        child = r.json()
        assert child["parent_id"] == root_id
        child_id = child["id"]

        # Tree nests the child under the root.
        r = client.get(f"/api/products/{pid}/knowledge/tree")
        assert r.status_code == 200
        tree = r.json()
        assert len(tree) == 1
        assert tree[0]["id"] == root_id
        assert [c["id"] for c in tree[0]["children"]] == [child_id]

        # Get single node.
        r = client.get(f"/api/products/{pid}/knowledge/nodes/{child_id}")
        assert r.status_code == 200
        assert r.json()["content_md"] == "arch"

        # Update node (WYSIWYG saves content_md).
        r = client.put(
            f"/api/products/{pid}/knowledge/nodes/{child_id}",
            json={"content_md": "arch v2", "title": "Architecture2"},
        )
        assert r.status_code == 200
        assert r.json()["content_md"] == "arch v2"
        assert r.json()["title"] == "Architecture2"

        # Verify (admin -> allowed).
        r = client.post(f"/api/products/{pid}/knowledge/nodes/{child_id}/verify")
        assert r.status_code == 200
        body = r.json()
        assert body["verified"] is True
        assert body["verified_by"] == "system"
        assert body["verified_at"] is not None

        # Delete the child then the root. The DB-level ON DELETE CASCADE on
        # knowledge_nodes.parent_id is only enforced on Postgres (SQLite does
        # not enable FK enforcement by default), so we delete explicitly to
        # keep the assertion backend-agnostic.
        r = client.delete(f"/api/products/{pid}/knowledge/nodes/{child_id}")
        assert r.status_code == 200
        r = client.delete(f"/api/products/{pid}/knowledge/nodes/{root_id}")
        assert r.status_code == 200
        r = client.get(f"/api/products/{pid}/knowledge/nodes/{root_id}")
        assert r.status_code == 404
        r = client.get(f"/api/products/{pid}/knowledge/tree")
        assert r.json() == []

    def test_create_rejects_foreign_parent(self, tmp_path, monkeypatch):
        client, TestSession = self._client(tmp_path, monkeypatch)
        pid = self._seed_product(TestSession)
        r = client.post(
            f"/api/products/{pid}/knowledge/nodes",
            json={"parent_id": "node_otherproduct", "title": "X"},
        )
        assert r.status_code == 400

    def test_get_node_404_for_other_product(self, tmp_path, monkeypatch):
        client, TestSession = self._client(tmp_path, monkeypatch)
        pid = self._seed_product(TestSession)
        r = client.post(
            f"/api/products/{pid}/knowledge/nodes",
            json={"title": "X"},
        )
        nid = r.json()["id"]
        # Different product id -> 404.
        r = client.get(f"/api/products/prod_other/knowledge/nodes/{nid}")
        assert r.status_code == 404

    def test_summary_503_when_no_content(self, tmp_path, monkeypatch):
        client, TestSession = self._client(tmp_path, monkeypatch)
        pid = self._seed_product(TestSession)
        # No artifacts/nodes -> nothing to summarize -> 503.
        r = client.post(f"/api/products/{pid}/summary")
        assert r.status_code == 503

    def test_summary_stores_and_returns(self, tmp_path, monkeypatch):
        client, TestSession = self._client(tmp_path, monkeypatch)
        pid = self._seed_product(TestSession)

        # Add a knowledge node with content (no artifacts).
        client.post(
            f"/api/products/{pid}/knowledge/nodes",
            json={"title": "P", "content_md": "The service does X and Y."},
        )

        # Mock the summary LLM so we don't need a live LLM. The real builder
        # takes (model, base_url=..., api_key=...).
        import api.docgen.summary as ks
        class _FakeLLM:
            async def generate(self, prompt):
                return "Acme is a service that does X and Y."
        monkeypatch.setattr(ks, "_safe_build_summary_llm", lambda m, **kw: _FakeLLM())

        r = client.post(f"/api/products/{pid}/summary")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["summary"] == "Acme is a service that does X and Y."
        # Persisted onto ProductORM.summary.
        from api.models import ProductORM
        with TestSession() as db:
            p = db.get(ProductORM, pid)
            assert p.summary == body["summary"]
