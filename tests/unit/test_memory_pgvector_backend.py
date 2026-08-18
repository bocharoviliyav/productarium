"""Unit tests for ``api.memory.pgvector_backend``.

Covers:
- ``_split_text``: non-empty input returns non-empty chunks; empty/whitespace
  returns []; naive fallback path (adalflow TextSplitter import failure).
- ``_is_pgvector_capable``: False on SQLite (test env) and when pgvector is
  unavailable; True when Postgres + pgvector available.
- ``PgVectorMemoryBackend.index``: empty content → 0; embedder failure → 0;
  happy path mocks ``_embed_batch`` + ``SessionLocal`` and asserts delete-then-
  insert upsert (idempotent per source_id).
- ``query``: returns "" on SQLite / pgvector-absent (no cosine operator); the
  cosine SQL shape is asserted via a mocked engine on the pgvector-capable path.
- ``clear_product``: deletes chunks for the product.
- ``status``: reports backend name + counts (non-fatal on DB down).
- ``reindex_product``: loads products + artifacts and re-indexes them.

The test env is SQLite (conftest autouse ``_isolated_env``), so the cosine
query path is exercised via a mocked engine + ``monkeypatch`` of
``_is_pgvector_capable`` rather than a real pgvector install.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


# --------------------------------------------------------------------------- #
# _split_text
# --------------------------------------------------------------------------- #
class TestSplitText:
    def test_empty_returns_empty(self):
        from api.memory.pgvector_backend import _split_text

        assert _split_text("") == []
        assert _split_text("   \n\t  ") == []

    def test_none_returns_empty(self):
        from api.memory.pgvector_backend import _split_text

        assert _split_text(None) == []  # type: ignore[arg-type]

    def test_non_empty_returns_non_empty(self):
        from api.memory.pgvector_backend import _split_text

        chunks = _split_text("This is a sentence. " * 50)
        assert len(chunks) > 0
        assert all(isinstance(c, str) and c.strip() for c in chunks)

    def test_naive_fallback(self, monkeypatch):
        # Force the TextSplitter import to fail so the naive fallback runs.
        import sys as _sys
        from api.memory import pgvector_backend as pb

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _boom_import(name, *a, **k):
            if name.startswith("adalflow"):
                raise ImportError("adalflow absent")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", _boom_import)

        chunks = pb._split_text("Para one.\n\nPara two is longer than enough to be split into a window " * 20)
        assert len(chunks) > 0
        assert all(isinstance(c, str) and c.strip() for c in chunks)


# --------------------------------------------------------------------------- #
# _is_pgvector_capable
# --------------------------------------------------------------------------- #
class TestIsPgvectorCapable:
    def test_false_on_sqlite(self, isolated_db):
        from api.memory.pgvector_backend import _is_pgvector_capable

        # isolated_db reloads api.db with DB_PROVIDER=sqlite (from autouse env).
        assert _is_pgvector_capable() is False

    def test_false_when_pgvector_unavailable(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb

        # Even if DB_PROVIDER says postgres, missing pgvector lib → False.
        import api.db as db_mod
        import api.models as models_mod

        monkeypatch.setattr(db_mod, "DB_PROVIDER", "postgres")
        monkeypatch.setattr(models_mod, "_PGVECTOR_AVAILABLE", False)

        assert pb._is_pgvector_capable() is False

    def test_true_when_postgres_and_pgvector(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb

        import api.db as db_mod
        import api.models as models_mod

        monkeypatch.setattr(db_mod, "DB_PROVIDER", "postgres")
        monkeypatch.setattr(models_mod, "_PGVECTOR_AVAILABLE", True)

        assert pb._is_pgvector_capable() is True


# --------------------------------------------------------------------------- #
# PgVectorMemoryBackend.index
# --------------------------------------------------------------------------- #
class TestIndex:
    def test_empty_content_returns_zero(self, isolated_db):
        from api.memory.pgvector_backend import PgVectorMemoryBackend

        be = PgVectorMemoryBackend()
        assert asyncio.run(be.index("", "prod_1")) == 0
        assert asyncio.run(be.index("   ", "prod_1")) == 0
        assert asyncio.run(be.index("content", "")) == 0

    def test_embedder_failure_returns_zero(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb

        async def _fail(texts):
            return None

        monkeypatch.setattr(pb, "_embed_batch", _fail)

        be = pb.PgVectorMemoryBackend()
        # Embedder returns None → 0 chunks stored.
        assert asyncio.run(be.index("some content here " * 50, "prod_1")) == 0

    def test_embedder_count_mismatch_returns_zero(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb

        captured: dict = {}

        async def _short(texts):
            captured["n_chunks"] = len(texts)
            # Return fewer vectors than chunks (guaranteed mismatch as long as
            # the splitter produced > 1 chunk).
            return [[0.1, 0.2] for _ in range(len(texts) - 1)]

        monkeypatch.setattr(pb, "_embed_batch", _short)

        be = pb.PgVectorMemoryBackend()
        # A long input guarantees multiple chunks.
        result = asyncio.run(be.index("some content here. " * 500, "prod_1"))
        assert captured["n_chunks"] > 1
        assert result == 0

    def test_happy_path_upserts_chunks(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb
        from api.models import KnowledgeChunkORM, ProductORM

        # Seed a product (FK) + one pre-existing chunk for the same source to
        # verify the delete-before-insert upsert.
        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(KnowledgeChunkORM(
                id="chunk_stale", product_id="prod_1", source_type="codebase",
                source_id="cb_1", chunk_index=0, content="stale",
            ))
            db.commit()
        finally:
            db.close()

        captured_embed: dict = {}

        async def _fake_embed(texts):
            captured_embed["n"] = len(texts)
            return [[float(i), float(i) + 0.1] for i in range(len(texts))]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)

        be = pb.PgVectorMemoryBackend()
        n = asyncio.run(be.index(
            "some content here " * 50, "prod_1",
            source_type="codebase", source_id="cb_1",
        ))
        assert n == captured_embed["n"]
        assert n > 0

        # The stale chunk was deleted; only the new chunks remain for cb_1.
        db = isolated_db.SessionLocal()
        try:
            rows = db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.source_id == "cb_1"
            ).all()
            assert len(rows) == n
            assert all(r.id != "chunk_stale" for r in rows)
            assert all(r.source_type == "codebase" for r in rows)
        finally:
            db.close()

    def test_upsert_without_source_id_inserts_all(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb
        from api.models import KnowledgeChunkORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.commit()
        finally:
            db.close()

        async def _fake_embed(texts):
            return [[float(i)] * 4 for i in range(len(texts))]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)

        be = pb.PgVectorMemoryBackend()
        # source_id=None → no delete filter (insert only).
        n = asyncio.run(be.index("content " * 50, "prod_1", source_id=None))
        assert n > 0

        db = isolated_db.SessionLocal()
        try:
            rows = db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.product_id == "prod_1"
            ).all()
            assert len(rows) == n
        finally:
            db.close()


# --------------------------------------------------------------------------- #
# PgVectorMemoryBackend.query
# --------------------------------------------------------------------------- #
class TestQuery:
    def test_empty_query_returns_empty(self, isolated_db):
        from api.memory.pgvector_backend import PgVectorMemoryBackend

        be = PgVectorMemoryBackend()
        assert asyncio.run(be.query("", "prod_1")) == ""
        assert asyncio.run(be.query("   ", "prod_1")) == ""
        assert asyncio.run(be.query("q", "")) == ""

    def test_returns_empty_on_sqlite(self, isolated_db):
        from api.memory.pgvector_backend import PgVectorMemoryBackend

        be = PgVectorMemoryBackend()
        # SQLite is not pgvector-capable → returns "" without touching the DB.
        assert asyncio.run(be.query("how does auth work", "prod_1")) == ""

    def test_cosine_search_sql_shape(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb

        # Force the pgvector-capable path on.
        monkeypatch.setattr(pb, "_is_pgvector_capable", lambda: True)

        async def _fake_embed_query(q):
            return [0.1, 0.2, 0.3]

        monkeypatch.setattr(pb, "_embed_query", _fake_embed_query)

        executed: dict = {}

        class _FakeResult:
            def fetchall(self):
                return [("chunk one",), ("chunk two",)]

        class _FakeConn:
            def execute(self, sql, params):
                executed["sql"] = str(sql)
                executed["params"] = params
                return _FakeResult()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class _FakeEngine:
            def connect(self):
                return _FakeConn()

        import api.db as db_mod
        monkeypatch.setattr(db_mod, "engine", _FakeEngine())

        be = pb.PgVectorMemoryBackend()
        result = asyncio.run(be.query("how does auth work", "prod_1", top_k=5))

        assert "chunk one" in result
        assert "chunk two" in result
        # The cosine-distance ORDER BY + LIMIT shape is present.
        sql = executed["sql"]
        assert "knowledge_chunks" in sql
        assert "<=>" in sql
        assert "ORDER BY" in sql
        assert "LIMIT" in sql
        assert executed["params"]["pid"] == "prod_1"
        assert executed["params"]["k"] == 5
        # The query vector is passed as a "[...]" string literal for the cast.
        assert executed["params"]["q"].startswith("[")

    def test_query_embedder_failure_returns_empty(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb

        monkeypatch.setattr(pb, "_is_pgvector_capable", lambda: True)

        async def _fail(q):
            return None

        monkeypatch.setattr(pb, "_embed_query", _fail)

        be = pb.PgVectorMemoryBackend()
        assert asyncio.run(be.query("q", "prod_1")) == ""

    def test_query_db_error_returns_empty(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb

        monkeypatch.setattr(pb, "_is_pgvector_capable", lambda: True)

        async def _fake_embed_query(q):
            return [0.1, 0.2]

        monkeypatch.setattr(pb, "_embed_query", _fake_embed_query)

        class _BoomEngine:
            def connect(self):
                raise RuntimeError("db down")

        import api.db as db_mod
        monkeypatch.setattr(db_mod, "engine", _BoomEngine())

        be = pb.PgVectorMemoryBackend()
        assert asyncio.run(be.query("q", "prod_1")) == ""


# --------------------------------------------------------------------------- #
# PgVectorMemoryBackend.clear_product
# --------------------------------------------------------------------------- #
class TestClearProduct:
    def test_clears_chunks_for_product(self, isolated_db):
        from api.models import KnowledgeChunkORM, ProductORM
        from api.memory.pgvector_backend import PgVectorMemoryBackend

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(KnowledgeChunkORM(
                id="c1", product_id="prod_1", source_type="codebase",
                chunk_index=0, content="a",
            ))
            db.add(KnowledgeChunkORM(
                id="c2", product_id="prod_1", source_type="codebase",
                chunk_index=1, content="b",
            ))
            db.add(ProductORM(id="prod_2", name="P2", description=""))
            db.add(KnowledgeChunkORM(
                id="c3", product_id="prod_2", source_type="codebase",
                chunk_index=0, content="c",
            ))
            db.commit()
        finally:
            db.close()

        be = PgVectorMemoryBackend()
        assert asyncio.run(be.clear_product("prod_1")) is True

        db = isolated_db.SessionLocal()
        try:
            assert db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.product_id == "prod_1"
            ).count() == 0
            # prod_2 untouched.
            assert db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.product_id == "prod_2"
            ).count() == 1
        finally:
            db.close()

    def test_empty_product_id_returns_false(self, isolated_db):
        from api.memory.pgvector_backend import PgVectorMemoryBackend

        be = PgVectorMemoryBackend()
        assert asyncio.run(be.clear_product("")) is False


# --------------------------------------------------------------------------- #
# PgVectorMemoryBackend.status
# --------------------------------------------------------------------------- #
class TestStatus:
    def test_status_reports_counts(self, isolated_db):
        from api.models import KnowledgeChunkORM, ProductORM
        from api.memory.pgvector_backend import PgVectorMemoryBackend

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(KnowledgeChunkORM(
                id="c1", product_id="prod_1", chunk_index=0, content="a",
            ))
            db.add(KnowledgeChunkORM(
                id="c2", product_id="prod_1", chunk_index=1, content="b",
            ))
            db.add(ProductORM(id="prod_2", name="P2", description=""))
            db.add(KnowledgeChunkORM(
                id="c3", product_id="prod_2", chunk_index=0, content="c",
            ))
            db.commit()
        finally:
            db.close()

        be = PgVectorMemoryBackend()
        status = be.status()
        assert status["backend"] == "pgvector"
        assert status["chunk_count"] == 3
        assert status["product_count"] == 2

    def test_status_empty_db(self, isolated_db):
        from api.memory.pgvector_backend import PgVectorMemoryBackend

        be = PgVectorMemoryBackend()
        status = be.status()
        assert status["backend"] == "pgvector"
        assert status["chunk_count"] == 0
        assert status["product_count"] == 0


# --------------------------------------------------------------------------- #
# PgVectorMemoryBackend.reindex_product
# --------------------------------------------------------------------------- #
class TestReindexProduct:
    def test_no_products_returns_zero(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb

        # Avoid real embeddings.
        async def _fake_embed(texts):
            return [[float(i)] * 4 for i in range(len(texts))]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)

        be = pb.PgVectorMemoryBackend()
        result = asyncio.run(be.reindex_product())
        assert result["success"] is True
        assert result["reindexed_count"] == 0

    def test_reindexes_one_product(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb
        from api.models import CodebaseORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo",
                generated_docs="generated wiki content " * 10,
            ))
            db.commit()
        finally:
            db.close()

        async def _fake_embed(texts):
            return [[float(i)] * 4 for i in range(len(texts))]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)

        be = pb.PgVectorMemoryBackend()
        result = asyncio.run(be.reindex_product("prod_1"))
        assert result["success"] is True
        assert result["reindexed_count"] == 1

        # Chunks were written.
        from api.models import KnowledgeChunkORM
        db = isolated_db.SessionLocal()
        try:
            assert db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.product_id == "prod_1"
            ).count() > 0
        finally:
            db.close()

    def test_reindex_error_returns_failure(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb

        # Force _load inside reindex to blow up by making SessionLocal raise.
        import api.db as db_mod

        class _BoomSession:
            def __enter__(self):
                raise RuntimeError("db down")

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(db_mod, "SessionLocal", lambda: _BoomSession())

        be = pb.PgVectorMemoryBackend()
        result = asyncio.run(be.reindex_product("prod_1"))
        assert result["success"] is False
        assert result["reindexed_count"] == 0
