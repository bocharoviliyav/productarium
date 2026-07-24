#!/usr/bin/env python3
"""Unit tests for the knowledge-tree Wave 2 scope.

Covers:
- ``api.routers.knowledge.build_tree`` (parent_id nesting + orphan handling)
- ``api.routers.knowledge._slugify``
- ``api.knowledge_summary.generate_product_summary`` (mocked LLM + empty content)
- ``api.artifact_docgen`` new-type dispatch (spec/links/documentation/guides +
  legacy type mapping) via monkeypatched sub-generators
- ``api.artifact_docgen._render_links_index`` / ``generate_links_docs`` /
  ``generate_guides_docs`` / ``generate_documentation_docs`` (passthrough)
- Router endpoint integration (create/get/tree/put/delete/verify) over an
  isolated SQLite DB via FastAPI TestClient with dependency overrides.

No live Ollama/Postgres/cognee is required: LLM and cognee indexing are mocked.
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
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
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
        import api.knowledge_summary as ks

        class _FakeLLM:
            async def generate(self, prompt: str) -> str:
                # Confirm the prompt includes product name + collected content.
                assert "Acme" in prompt
                assert "artifact docs" in prompt
                assert "node content" in prompt
                return "```markdown\nAcme is a service.\n```"

        # _safe_build_summary_llm now takes base_url/api_key kwargs (so the
        # summary LLM can reach a corporate gateway); accept **kwargs here.
        monkeypatch.setattr(ks, "_safe_build_summary_llm", lambda p, m, **kw: _FakeLLM())
        product = SimpleNamespace(id="prod_1", name="Acme")
        artifacts = [SimpleNamespace(id="art_1", name="svc", generated_docs="artifact docs")]
        nodes = [SimpleNamespace(id="node_1", title="P1", content_md="node content")]
        out = asyncio.run(ks.generate_product_summary(product, artifacts, nodes))
        assert out == "Acme is a service."

    def test_summary_empty_when_no_content(self):
        import api.knowledge_summary as ks
        product = SimpleNamespace(id="prod_1", name="Acme")
        out = asyncio.run(ks.generate_product_summary(product, [], []))
        assert out == ""

    def test_summary_empty_when_llm_unavailable(self, monkeypatch):
        import api.knowledge_summary as ks
        monkeypatch.setattr(ks, "_safe_build_summary_llm", lambda p, m, **kw: None)
        product = SimpleNamespace(id="prod_1", name="Acme")
        artifacts = [SimpleNamespace(id="a", name="a", generated_docs="some docs")]
        out = asyncio.run(ks.generate_product_summary(product, artifacts, []))
        assert out == ""

    def test_collect_summary_content_caps_large_input(self):
        import api.knowledge_summary as ks
        big = "x" * (ks.SUMMARY_CONTEXT_MAX_CHARS + 5000)
        artifacts = [SimpleNamespace(id="a", name="a", generated_docs=big)]
        out = ks._collect_summary_content(artifacts, [])
        assert len(out) <= ks.SUMMARY_CONTEXT_MAX_CHARS + 200
        assert out.endswith("обрезано для контекста LLM)\n")


# --- New-type dispatch -------------------------------------------------------
class TestArtifactDocgenDispatch:
    @pytest.fixture
    def patched_generators(self, monkeypatch):
        """Monkeypatch every sub-generator to return a sentinel; no LLM/cognee."""
        import api.artifact_docgen as adg
        sentinels = {}

        def _mk(name):
            async def _f(artifact, product, *a, **k):
                return name
            return _f

        for name in (
            "generate_codebase_docs", "generate_openapi_docs",
            "generate_asyncapi_docs", "generate_testcase_docs",
            "generate_links_docs", "generate_documentation_docs",
            "generate_guides_docs",
        ):
            sentinels[name] = name.upper()
            monkeypatch.setattr(adg, name, _mk(name.upper()))
        return adg

    def _art(self, atype, kind=None):
        return SimpleNamespace(
            id="art_1", name="svc", type=atype, kind=kind,
            repo_url=None, repo_type=None, token=None, content="c",
            allure_url=None, generated_docs=None, pages=None,
        )

    def test_codebase_routes(self, patched_generators):
        adg = patched_generators
        out = asyncio.run(adg.generate_artifact_documentation(self._art("codebase"), SimpleNamespace(id="prod_1")))
        assert out == "GENERATE_CODEBASE_DOCS"

    def test_spec_default_openapi(self, patched_generators):
        adg = patched_generators
        out = asyncio.run(adg.generate_artifact_documentation(self._art("spec"), SimpleNamespace(id="prod_1")))
        assert out == "GENERATE_OPENAPI_DOCS"

    def test_spec_kind_asyncapi(self, patched_generators):
        adg = patched_generators
        out = asyncio.run(adg.generate_artifact_documentation(self._art("spec", kind="asyncapi"), SimpleNamespace(id="prod_1")))
        assert out == "GENERATE_ASYNCAPI_DOCS"

    def test_links_routes(self, patched_generators):
        adg = patched_generators
        out = asyncio.run(adg.generate_artifact_documentation(self._art("links"), SimpleNamespace(id="prod_1")))
        assert out == "GENERATE_LINKS_DOCS"

    def test_guides_routes(self, patched_generators):
        adg = patched_generators
        out = asyncio.run(adg.generate_artifact_documentation(self._art("guides"), SimpleNamespace(id="prod_1")))
        assert out == "GENERATE_GUIDES_DOCS"

    def test_documentation_default_routes(self, patched_generators):
        adg = patched_generators
        out = asyncio.run(adg.generate_artifact_documentation(self._art("documentation"), SimpleNamespace(id="prod_1")))
        assert out == "GENERATE_DOCUMENTATION_DOCS"

    def test_documentation_kind_testcase_routes(self, patched_generators):
        adg = patched_generators
        out = asyncio.run(adg.generate_artifact_documentation(self._art("documentation", kind="testcase"), SimpleNamespace(id="prod_1")))
        assert out == "GENERATE_TESTCASE_DOCS"

    # --- legacy type mapping ---
    def test_legacy_openapi_maps_to_spec_openapi(self, patched_generators):
        adg = patched_generators
        out = asyncio.run(adg.generate_artifact_documentation(self._art("openapi"), SimpleNamespace(id="prod_1")))
        assert out == "GENERATE_OPENAPI_DOCS"

    def test_legacy_asyncapi_maps_to_spec_asyncapi(self, patched_generators):
        adg = patched_generators
        out = asyncio.run(adg.generate_artifact_documentation(self._art("asyncapi"), SimpleNamespace(id="prod_1")))
        assert out == "GENERATE_ASYNCAPI_DOCS"

    def test_legacy_testcase_maps_to_documentation_testcase(self, patched_generators):
        adg = patched_generators
        out = asyncio.run(adg.generate_artifact_documentation(self._art("testcase"), SimpleNamespace(id="prod_1")))
        assert out == "GENERATE_TESTCASE_DOCS"

    def test_unsupported_type_raises(self, patched_generators):
        adg = patched_generators
        with pytest.raises(ValueError):
            asyncio.run(adg.generate_artifact_documentation(self._art("unknown"), SimpleNamespace(id="prod_1")))


# --- Real generators (no LLM path) -------------------------------------------
class TestRealGenerators:
    @pytest.fixture
    def no_index(self, monkeypatch):
        import api.artifact_docgen as adg
        monkeypatch.setattr(adg, "_index_in_background", lambda *a, **k: None)
        return adg

    def test_render_links_index_json_list(self):
        from api.artifact_docgen import _render_links_index
        art = SimpleNamespace(name="Useful links")
        content = '[{"url":"https://a.example","title":"A","description":"aa"},{"url":"https://b.example","title":"B"}]'
        md = _render_links_index(content, art)
        assert "# Links: Useful links" in md
        assert "[A](https://a.example)" in md
        assert "— aa" in md
        assert "[B](https://b.example)" in md

    def test_render_links_index_links_wrapper(self):
        from api.artifact_docgen import _render_links_index
        art = SimpleNamespace(name="L")
        md = _render_links_index('{"links":[{"url":"https://x","title":"X"}]}', art)
        assert "[X](https://x)" in md

    def test_render_links_index_non_json_passthrough(self):
        from api.artifact_docgen import _render_links_index
        art = SimpleNamespace(name="L")
        md = _render_links_index("## My links\n- [x](https://x)", art)
        assert "## My links" in md

    def test_render_links_index_empty(self):
        from api.artifact_docgen import _render_links_index
        art = SimpleNamespace(name="L")
        md = _render_links_index("", art)
        assert "Ссылки не предоставлены" in md

    def test_generate_links_docs_persists_and_pages(self, no_index):
        adg = no_index
        art = SimpleNamespace(
            id="art_1", name="Links", type="links", kind=None,
            content='[{"url":"https://a","title":"A"}]',
            generated_docs=None, pages=None,
        )
        product = SimpleNamespace(id="prod_1")
        md = asyncio.run(adg.generate_links_docs(art, product))
        assert "[A](https://a)" in md
        assert art.generated_docs == md
        assert art.pages is not None and "page_links" in art.pages

    def test_generate_guides_docs_passthrough(self, no_index):
        adg = no_index
        art = SimpleNamespace(
            id="art_1", name="Onboarding", type="guides", kind=None,
            content="Step 1: do thing", generated_docs=None, pages=None,
        )
        md = asyncio.run(adg.generate_guides_docs(art, SimpleNamespace(id="prod_1")))
        assert "# Onboarding" in md
        assert "Step 1: do thing" in md
        assert art.generated_docs == md
        assert "page_guides" in art.pages

    def test_generate_documentation_docs_passthrough_when_llm_empty(self, no_index, monkeypatch):
        adg = no_index
        # LLM enrichment returns "" -> passthrough content is used verbatim.
        async def _empty(*a, **k):
            return ""
        monkeypatch.setattr(adg, "_llm_or_none", _empty)
        art = SimpleNamespace(
            id="art_1", name="Manual doc", type="documentation", kind=None,
            content="## Intro\nSome text", generated_docs=None, pages=None,
        )
        md = asyncio.run(adg.generate_documentation_docs(art, SimpleNamespace(id="prod_1")))
        assert "## Intro" in md
        assert art.generated_docs == md
        assert "page_documentation" in art.pages

    def test_generate_documentation_docs_empty_content_raises(self, no_index):
        adg = no_index
        art = SimpleNamespace(
            id="art_1", name="doc", type="documentation", kind=None,
            content="   ", generated_docs=None, pages=None,
        )
        with pytest.raises(ValueError):
            asyncio.run(adg.generate_documentation_docs(art, SimpleNamespace(id="prod_1")))


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

        # Mock the summary LLM so we don't need live Ollama. The real builder
        # now accepts base_url/api_key kwargs; accept them here too.
        import api.knowledge_summary as ks
        class _FakeLLM:
            async def generate(self, prompt):
                return "Acme is a service that does X and Y."
        monkeypatch.setattr(ks, "_safe_build_summary_llm", lambda p, m, **kw: _FakeLLM())

        r = client.post(f"/api/products/{pid}/summary")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["summary"] == "Acme is a service that does X and Y."
        # Persisted onto ProductORM.summary.
        from api.models import ProductORM
        with TestSession() as db:
            p = db.get(ProductORM, pid)
            assert p.summary == body["summary"]
