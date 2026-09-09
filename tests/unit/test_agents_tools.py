"""Unit tests for ``api.agents.tools`` and ``api.agents.runtime`` (Wave B).

Covers:
- Tool factory: names, EN descriptions, product_id closure binding (the LLM
  cannot influence the product scope).
- ``codebase_file_read``: path traversal (``..``, absolute, symlink escape),
  missing clone (polite error), file-not-found, happy path + truncation.
- ``spec_read`` / ``link_read`` / ``node_read``: exact + prefix matching,
  product isolation (another product's entities are invisible).
- ``knowledge_recall``: pgvector fallback on SQLite (recent chunks) +
  citation formatting.
- ``api.agents.runtime``: Postgres-only checkpointer (memory via explicit
  test-only env override; setup() before pipeline entry; fatal failures).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.models import (  # noqa: E402
    CodebaseORM,
    KnowledgeChunkORM,
    KnowledgeNodeORM,
    LinksORM,
    ProductORM,
    SpecORM,
)


# --- Seed helpers ------------------------------------------------------------
def _seed(db_mod, product_id: str, *, spec: bool = False, links: bool = False,
          node: bool = False, chunks: int = 0):
    """Seed one product with optional entities; returns created ids."""
    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id=product_id, name=f"Product {product_id}"))
        db.commit()
    ids = {}
    if spec:
        with db_mod.SessionLocal() as db:
            db.add(SpecORM(id=f"spec_{product_id}", product_id=product_id,
                           name="Payments API", kind="openapi",
                           content="openapi: 3.0.0\npaths: /pay",
                           source="manual"))
            db.commit()
        ids["spec"] = "Payments API"
    if links:
        with db_mod.SessionLocal() as db:
            db.add(LinksORM(id=f"links_{product_id}", product_id=product_id,
                            name="Runbooks", content='[{"url": "http://x"}]',
                            source="manual"))
            db.commit()
        ids["links"] = "Runbooks"
    if node:
        with db_mod.SessionLocal() as db:
            db.add(KnowledgeNodeORM(id=f"node_{product_id}",
                                    product_id=product_id, title="On-call Guide",
                                    slug="on-call-guide",
                                    content_md="# On-call",
                                    source="manual"))
            db.commit()
        ids["node"] = "On-call Guide"
    if chunks:
        with db_mod.SessionLocal() as db:
            for i in range(chunks):
                db.add(KnowledgeChunkORM(
                    id=f"chunk_{product_id}_{i}", product_id=product_id,
                    source_type="codebase", source_id=f"cb_{product_id}",
                    chunk_index=i, content=f"chunk {i} of {product_id}",
                ))
            db.commit()
    return ids


def _tools_for(db_mod, product_id: str):
    from api.agents.tools import build_expert_tools
    return {t.name: t for t in build_expert_tools(product_id,
                                                  session_factory=db_mod.SessionLocal)}


# --- Tool factory ------------------------------------------------------------
class TestToolFactory:
    def test_returns_expected_tools_with_en_descriptions(self, isolated_db):
        tools = _tools_for(isolated_db, "prod_1")
        assert sorted(tools) == sorted([
            "knowledge_recall", "codebase_file_read", "spec_read",
            "link_read", "node_read",
        ])
        for tool in tools.values():
            assert tool.description, f"{tool.name} has no description"
            # Descriptions must be ASCII English (Wave B requirement).
            assert tool.description.isascii()

    def test_product_id_is_required(self):
        from api.agents.tools import build_expert_tools
        with pytest.raises(ValueError):
            build_expert_tools("")


# --- codebase_file_read ------------------------------------------------------
class TestCodebaseFileRead:
    @pytest.fixture(autouse=True)
    def _allow_local_clones(self, monkeypatch):
        # These tests register arbitrary tmp_path directories as local clones.
        monkeypatch.setenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", "1")

    def _make_clone(self, tmp_path):
        clone = tmp_path / "repo"
        (clone / "src").mkdir(parents=True)
        (clone / "src" / "main.py").write_text("print('hi')")
        (clone / "secret.txt").write_text("top secret")
        # A file OUTSIDE the clone, reachable via a symlink inside it.
        outside = tmp_path / "outside.txt"
        outside.write_text("outside secret")
        os.symlink(outside, clone / "link-outside.txt")
        return clone

    def _add_local_codebase(self, db_mod, product_id, clone_path):
        with db_mod.SessionLocal() as db:
            db.add(CodebaseORM(id=f"cb_{product_id}", product_id=product_id,
                               name=f"repo-{product_id}", repo_url=str(clone_path),
                               repo_type="local", source="manual"))
            db.commit()

    def test_reads_file_inside_clone(self, isolated_db, tmp_path):
        clone = self._make_clone(tmp_path)
        self._add_local_codebase(isolated_db, "prod_1", clone)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["codebase_file_read"].ainvoke({"path": "src/main.py"}))
        assert "print('hi')" in out
        assert "[source: codebase repo-prod_1 file=src/main.py]" in out

    def test_rejects_dotdot_traversal(self, isolated_db, tmp_path):
        clone = self._make_clone(tmp_path)
        self._add_local_codebase(isolated_db, "prod_1", clone)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["codebase_file_read"].ainvoke(
            {"path": "../../outside.txt"}))
        assert "rejected for security reasons" in out
        assert "outside secret" not in out

    def test_rejects_absolute_path(self, isolated_db, tmp_path):
        clone = self._make_clone(tmp_path)
        self._add_local_codebase(isolated_db, "prod_1", clone)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["codebase_file_read"].ainvoke(
            {"path": "/etc/passwd"}))
        assert "absolute paths are not allowed" in out
        assert "passwd" not in out.split("absolute paths are not allowed")[0]

    def test_rejects_symlink_escape(self, isolated_db, tmp_path):
        clone = self._make_clone(tmp_path)
        self._add_local_codebase(isolated_db, "prod_1", clone)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["codebase_file_read"].ainvoke(
            {"path": "link-outside.txt"}))
        assert "outside secret" not in out
        # The symlink target escapes the root: either a traversal rejection
        # or a not-found. Both are safe outcomes; assert no leak.
        assert "outside secret" not in out

    def test_rejects_git_metadata(self, isolated_db, tmp_path):
        clone = self._make_clone(tmp_path)
        git_dir = clone / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text(
            '[remote "origin"]\n'
            "url = https://ghp_SECRET@github.com/victim/private.git\n"
        )
        self._add_local_codebase(isolated_db, "prod_1", clone)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["codebase_file_read"].ainvoke({"path": ".git/config"}))
        assert "not readable by the agent" in out
        assert "SECRET" not in out
        # Case-folded variant must be rejected too (macOS/Windows FS).
        out = asyncio.run(tools["codebase_file_read"].ainvoke({"path": ".GIT/config"}))
        assert "not readable by the agent" in out

    def test_rejects_credential_files(self, isolated_db, tmp_path):
        clone = self._make_clone(tmp_path)
        (clone / ".env").write_text("API_KEY=x")
        (clone / ".env.local").write_text("API_KEY=x")
        (clone / "id_rsa").write_text("-----BEGIN RSA PRIVATE KEY-----")
        (clone / "credentials.json").write_text('{"git":{"token":"t"}}')
        certs = clone / "certs"
        certs.mkdir()
        (certs / "server.pem").write_text("PEM")
        (certs / "server.key").write_text("KEY")
        self._add_local_codebase(isolated_db, "prod_1", clone)
        tools = _tools_for(isolated_db, "prod_1")
        for path in (
            ".env",
            ".env.local",
            "id_rsa",
            "credentials.json",
            "certs/server.pem",
            "certs/server.key",
        ):
            out = asyncio.run(tools["codebase_file_read"].ainvoke({"path": path}))
            assert "not readable by the agent" in out, path
            assert "API_KEY" not in out
        # Ordinary files still read fine.
        out = asyncio.run(tools["codebase_file_read"].ainvoke({"path": "src/main.py"}))
        assert "print('hi')" in out

    def test_polite_error_when_no_clone(self, isolated_db):
        _seed(isolated_db, "prod_1")
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["codebase_file_read"].ainvoke({"path": "x.py"}))
        assert "No local codebase clone is available" in out

    def test_missing_file_not_found_message(self, isolated_db, tmp_path):
        clone = self._make_clone(tmp_path)
        self._add_local_codebase(isolated_db, "prod_1", clone)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["codebase_file_read"].ainvoke(
            {"path": "nope.py"}))
        assert "File not found" in out

    def test_isolated_from_other_product_clone(self, isolated_db, tmp_path):
        clone = self._make_clone(tmp_path)
        _seed(isolated_db, "prod_1")
        self._add_local_codebase(isolated_db, "prod_2", clone)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["codebase_file_read"].ainvoke({"path": "src/main.py"}))
        assert "No local codebase clone is available" in out


# --- local clone-root confinement -------------------------------------------
class TestCloneRootConfinement:
    def test_local_path_outside_managed_root_rejected(self, isolated_db, tmp_path, monkeypatch):
        monkeypatch.delenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", raising=False)
        _seed(isolated_db, "prod_conf")
        clone = tmp_path / "repo"
        (clone / "src").mkdir(parents=True)
        (clone / "src" / "main.py").write_text("print('hi')")
        with isolated_db.SessionLocal() as db:
            db.add(CodebaseORM(id="cb_conf", product_id="prod_conf", name="conf",
                               repo_url=str(clone), repo_type="local", source="manual"))
            db.commit()
        tools = _tools_for(isolated_db, "prod_conf")
        out = asyncio.run(tools["codebase_file_read"].ainvoke({"path": "src/main.py"}))
        assert "No local codebase clone is available" in out
        assert "print('hi')" not in out

    def test_local_path_inside_managed_root_allowed(self, isolated_db, tmp_path, monkeypatch):
        monkeypatch.delenv("PRODUCTARIUM_ALLOW_LOCAL_CLONES", raising=False)
        import api.repositories.documents as docs_mod

        managed = tmp_path / "managed"
        (managed / "repos").mkdir(parents=True)
        clone = managed / "repos" / "repo"
        (clone / "src").mkdir(parents=True)
        (clone / "src" / "main.py").write_text("print('managed')")
        monkeypatch.setattr(docs_mod, "DEFAULT_REPO_ROOT", str(managed))
        _seed(isolated_db, "prod_conf2")
        with isolated_db.SessionLocal() as db:
            db.add(CodebaseORM(id="cb_conf2", product_id="prod_conf2", name="conf2",
                               repo_url=str(clone), repo_type="local", source="manual"))
            db.commit()
        tools = _tools_for(isolated_db, "prod_conf2")
        out = asyncio.run(tools["codebase_file_read"].ainvoke({"path": "src/main.py"}))
        assert "print('managed')" in out


# --- spec/link/node reads ----------------------------------------------------
class TestEntityReads:
    def test_spec_read_exact_and_prefix(self, isolated_db):
        _seed(isolated_db, "prod_1", spec=True)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["spec_read"].ainvoke({"name": "Payments API"}))
        assert "openapi: 3.0.0" in out
        assert "[source: spec Payments API kind=openapi]" in out
        out = asyncio.run(tools["spec_read"].ainvoke({"name": "payments"}))
        assert "openapi: 3.0.0" in out

    def test_spec_read_unknown_name_lists_available(self, isolated_db):
        _seed(isolated_db, "prod_1", spec=True)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["spec_read"].ainvoke({"name": "ghost"}))
        assert "No spec named 'ghost'" in out
        assert "Payments API" in out

    def test_spec_read_isolated_per_product(self, isolated_db):
        _seed(isolated_db, "prod_1", spec=True)
        _seed(isolated_db, "prod_2")
        tools = _tools_for(isolated_db, "prod_2")
        out = asyncio.run(tools["spec_read"].ainvoke({"name": "Payments API"}))
        assert "No specs are attached" in out

    def test_link_read(self, isolated_db):
        _seed(isolated_db, "prod_1", links=True)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["link_read"].ainvoke({"name": "Runbooks"}))
        assert '[source: links Runbooks]' in out
        assert "http://x" in out

    def test_node_read_by_title_and_slug(self, isolated_db):
        _seed(isolated_db, "prod_1", node=True)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["node_read"].ainvoke({"title": "On-call Guide"}))
        assert "# On-call" in out
        out = asyncio.run(tools["node_read"].ainvoke({"slug": "on-call-guide"}))
        assert "# On-call" in out

    def test_node_read_requires_title_or_slug(self, isolated_db):
        _seed(isolated_db, "prod_1", node=True)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["node_read"].ainvoke({}))
        assert "provide 'title' or 'slug'" in out


# --- knowledge_recall --------------------------------------------------------
class TestKnowledgeRecall:
    def test_recall_returns_recent_chunks_with_citations_on_sqlite(self, isolated_db):
        _seed(isolated_db, "prod_1", chunks=3)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["knowledge_recall"].ainvoke({"query": "anything"}))
        assert "[source: codebase:cb_prod_1 chunk=chunk_prod_1_" in out
        assert "chunk 0 of prod_1" in out

    def test_recall_isolated_per_product(self, isolated_db):
        _seed(isolated_db, "prod_1", chunks=2)
        _seed(isolated_db, "prod_2", chunks=2)
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["knowledge_recall"].ainvoke({"query": "x"}))
        assert "of prod_1" in out
        assert "of prod_2" not in out

    def test_recall_empty_when_no_chunks(self, isolated_db):
        _seed(isolated_db, "prod_1")
        tools = _tools_for(isolated_db, "prod_1")
        out = asyncio.run(tools["knowledge_recall"].ainvoke({"query": "x"}))
        assert "No indexed knowledge matched" in out


# --- runtime: checkpointer (Postgres-only, setup before pipeline) -------------
class TestCheckpointer:
    def test_memory_override_env(self, isolated_db, monkeypatch):
        """PRODUCTARIUM_CHECKPOINTER=memory (test-only) -> InMemorySaver."""
        import api.agents.runtime as rt

        rt.reset_checkpointer_cache()
        monkeypatch.setenv("PRODUCTARIUM_CHECKPOINTER", "memory")
        try:
            saver = asyncio.run(rt.get_checkpointer())
            assert type(saver).__name__ == "InMemorySaver"
        finally:
            rt.reset_checkpointer_cache()

    def test_postgres_failure_is_fatal(self, isolated_db, monkeypatch):
        """No memory override + Postgres init failure -> get_checkpointer RAISES.

        The checkpointer is Postgres-only at runtime (no SQLite fallback);
        app startup must fail loudly when Postgres is unavailable.
        """
        import api.agents.runtime as rt

        rt.reset_checkpointer_cache()
        monkeypatch.delenv("PRODUCTARIUM_CHECKPOINTER", raising=False)

        async def _boom() -> None:
            raise RuntimeError("connection refused")

        monkeypatch.setattr(rt, "_build_postgres_saver", _boom)
        try:
            with pytest.raises(RuntimeError, match="connection refused"):
                asyncio.run(rt.get_checkpointer())
        finally:
            rt.reset_checkpointer_cache()

    def test_setup_runs_before_pipeline_entry(self, isolated_db, monkeypatch):
        """setup() must run on the bare connection BEFORE pipeline mode.

        CREATE INDEX CONCURRENTLY (checkpoint migrations) cannot run inside a
        transaction block; pipeline mode implies one — hence the order.
        """
        import api.agents.runtime as rt

        calls: list = []

        class _FakePipeline:
            async def __aenter__(self):
                calls.append("pipeline_enter")
                return self

            async def __aexit__(self, *exc):
                calls.append("pipeline_exit")
                return None

        class _FakeConn:
            def pipeline(self):
                return _FakePipeline()

            async def close(self):
                calls.append("conn_close")

        class _FakeAsyncConnection:
            # langgraph's _ainternal.py subscribes psycopg.AsyncConnection at
            # import time (AsyncConnection[DictRow]); the fake must allow that.
            def __class_getitem__(cls, item):
                return cls

            @staticmethod
            async def connect(*args, **kwargs):
                calls.append("connect")
                return _FakeConn()

        class _FakeSaver:
            def __init__(self, conn=None, **kwargs):
                calls.append("saver_init")
                self.conn = conn
                self.pipe = None

            async def setup(self):
                calls.append("setup")

        monkeypatch.setattr("psycopg.AsyncConnection", _FakeAsyncConnection)
        monkeypatch.setattr(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver", _FakeSaver
        )
        # Hermetic env is not postgres; feed a dummy conn string so the
        # builder proceeds to connect -> setup -> pipeline.
        monkeypatch.setattr(rt, "_postgres_conn_string", lambda: "host=stub")
        monkeypatch.delenv("PRODUCTARIUM_CHECKPOINTER", raising=False)

        rt.reset_checkpointer_cache()
        try:
            saver = asyncio.run(rt.get_checkpointer())
            assert calls[:4] == ["connect", "saver_init", "setup", "pipeline_enter"]
            assert saver.pipe is not None  # pipeline attached after setup
            asyncio.run(rt.close_checkpointer())
            assert calls[-2:] == ["pipeline_exit", "conn_close"]
        finally:
            rt.reset_checkpointer_cache()

    def test_postgres_conn_string_none_on_sqlite_provider(self, isolated_db):
        import api.agents.runtime as rt
        assert rt._postgres_conn_string() is None
