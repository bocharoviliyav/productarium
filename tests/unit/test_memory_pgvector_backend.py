"""Unit tests for ``api.memory.pgvector_backend``.

Covers:
- ``_split_text``: non-empty input returns non-empty chunks; empty/whitespace
  returns []; naive fallback path (langchain TextSplitter import failure).
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
        # Force the langchain TextSplitter import to fail so the naive
        # fallback runs.
        from api.memory import pgvector_backend as pb

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _boom_import(name, *a, **k):
            if name.startswith("langchain_text_splitters"):
                raise ImportError("langchain_text_splitters absent")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", _boom_import)

        chunks = pb._split_text("Para one.\n\nPara two is longer than enough to be split into a window " * 20)
        assert len(chunks) > 0
        assert all(isinstance(c, str) and c.strip() for c in chunks)


# --------------------------------------------------------------------------- #
# _embed_batch — langchain ``OpenAIEmbeddings.embed_documents`` returns a
# plain ``List[List[float]]`` (one vector per input text). Failure surfaces
# as a position-aligned list with ``None`` entries (callers drop the failed
# items) rather than an exception. Inputs are sanitized (control chars /
# unpaired surrogates) and a 400-rejected batch is bisected to isolate the
# offending chunk(s).
# --------------------------------------------------------------------------- #
class TestEmbedBatch:
    def _install_embedder(self, monkeypatch, fake_embedder):
        """Patch get_embedder at its source module + stub the rate limiter."""
        import api.tools.embedder as emb_mod
        from api.tools import rate_limiter as rl

        # get_embedder is imported INSIDE _embed_batch from
        # api.tools.embedder, so patch it at its source module (not on pb).
        monkeypatch.setattr(emb_mod, "get_embedder", lambda **kw: fake_embedder)
        # Stub the rate limiter so it just runs the to_thread coroutine.
        monkeypatch.setattr(
            rl._embedder_rate_limiter, "execute",
            lambda func, *a, **kw: func(*a, **kw),
        )

    def test_extracts_vectors_from_embed_documents(self, monkeypatch):
        from api.memory import pgvector_backend as pb

        class _FakeEmbedder:
            def embed_documents(self, texts):
                assert texts == ["a", "b", "c"]
                return [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        self._install_embedder(monkeypatch, _FakeEmbedder())

        result = asyncio.run(pb._embed_batch(["a", "b", "c"]))
        assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    def test_embedder_error_returns_aligned_none(self, monkeypatch):
        from api.memory import pgvector_backend as pb

        class _FakeEmbedder:
            def __init__(self):
                self.calls = 0

            def embed_documents(self, texts):
                self.calls += 1
                raise RuntimeError("embedder boom")

        fake = _FakeEmbedder()
        self._install_embedder(monkeypatch, fake)

        # Systemic failure (no status_code=400): no per-item isolation storm.
        result = asyncio.run(pb._embed_batch(["a", "b"]))
        assert result == [None, None]
        assert fake.calls == 1

    def test_empty_input_returns_empty_list(self):
        from api.memory.pgvector_backend import _embed_batch

        assert asyncio.run(_embed_batch([])) == []

    def test_numpy_array_vector_coerced_to_floats(self, monkeypatch):
        from api.memory import pgvector_backend as pb

        class _Arr:
            def tolist(self):
                return [0.1, 0.2]

        class _FakeEmbedder:
            def embed_documents(self, texts):
                return [_Arr()]

        self._install_embedder(monkeypatch, _FakeEmbedder())

        result = asyncio.run(pb._embed_batch(["a"]))
        assert result == [[0.1, 0.2]]


class TestEmbedSanitize:
    def test_drops_control_chars_and_surrogates(self):
        from api.memory.pgvector_backend import _sanitize_for_embedder

        # NUL / SOH dropped; \r\n\t survive; surrogate dropped.
        assert _sanitize_for_embedder("a\x00b\x01c\rd\ne\tf") == "abc\rd\ne\tf"
        assert _sanitize_for_embedder("bad \ud800 surrogate") == "bad  surrogate"
        assert _sanitize_for_embedder("\ud83d\ude00") == ""  # unpaired pair halves

    def test_keeps_valid_unicode(self):
        from api.memory.pgvector_backend import _sanitize_for_embedder

        assert _sanitize_for_embedder("привет 世界 🎉") == "привет 世界 🎉"
        assert _sanitize_for_embedder("") == ""
        assert _sanitize_for_embedder(None) == ""  # type: ignore[arg-type]

    def test_drops_del_c1_and_invisible_format_chars(self):
        from api.memory.pgvector_backend import _sanitize_for_embedder

        # DEL + C1 controls (another class local tokenizers reject).
        assert _sanitize_for_embedder("a\x7fb") == "ab"
        assert _sanitize_for_embedder("a\x9fb") == "ab"
        # Zero-width, bidi, soft hyphen, BOM.
        assert _sanitize_for_embedder("a\u200bb") == "ab"
        assert _sanitize_for_embedder("a\u202eb") == "ab"
        assert _sanitize_for_embedder("a\xadb") == "ab"
        assert _sanitize_for_embedder("\ufeffhi") == "hi"
        assert _sanitize_for_embedder("a\u2060b") == "ab"
        # Line/paragraph separators become plain newlines.
        assert _sanitize_for_embedder("a\u2028b\u2029c") == "a\nb\nc"
        # Regular punctuation / visible text is untouched.
        assert _sanitize_for_embedder("a—b ‘c’ d") == "a—b ‘c’ d"


class TestEmbedFoldRetry:
    """A single item rejected with 400 gets one bounded ASCII-fold retry
    before being dropped; the drop logs the payload + suspect codepoints."""

    def test_fold_unit(self):
        from api.memory.pgvector_backend import _fold_for_embedder

        assert _fold_for_embedder("привет") == ""  # non-latin folds to nothing
        assert _fold_for_embedder("hello привет") == "hello"
        assert _fold_for_embedder("naïve") == "naive"  # NFKD decomposes
        assert _fold_for_embedder("a\x00b") == "ab"
        assert _fold_for_embedder("a\n\tb") == "a b"  # whitespace collapses
        assert _fold_for_embedder("pure ascii") == "pure ascii"

    def test_fold_retry_rescues_rejected_chunk(self, monkeypatch):
        from api.memory import pgvector_backend as pb

        class _Rejected400(Exception):
            status_code = 400

        calls: list = []

        class _FakeEmbedder:
            def embed_documents(self, texts):
                calls.append(list(texts))
                if any("✗" in t for t in texts):
                    raise _Rejected400(
                        'Error code: 400 - {"message":"Prompt contains invalid tokens"}'
                    )
                return [[0.5, 0.6] for _ in texts]

        TestEmbedBatch()._install_embedder(monkeypatch, _FakeEmbedder())

        result = asyncio.run(pb._embed_batch(["alpha ok", "text ✗ poison", "beta ok"]))
        assert result[0] == [0.5, 0.6]
        assert result[1] == [0.5, 0.6]  # rescued via the folded retry
        assert result[2] == [0.5, 0.6]
        # The last isolated call carried the folded text (poison char gone).
        assert ["text poison"] in calls

    def test_fold_retry_failure_drops_with_diagnostics(self, monkeypatch, caplog):
        import logging

        from api.memory import pgvector_backend as pb

        class _Rejected400(Exception):
            status_code = 400

        class _FakeEmbedder:
            def embed_documents(self, texts):
                raise _Rejected400(
                    'Error code: 400 - {"message":"Prompt contains invalid tokens"}'
                )

        TestEmbedBatch()._install_embedder(monkeypatch, _FakeEmbedder())

        with caplog.at_level(logging.WARNING, logger="api.memory.pgvector_backend"):
            result = asyncio.run(pb._embed_batch(["ascii only", "привет"]))

        assert result == [None, None]
        drops = [r for r in caplog.records if "rejected by the embedder" in r.getMessage()]
        assert len(drops) == 2
        joined = " ".join(r.getMessage() for r in drops)
        assert "ascii only" in joined           # payload snippet visible
        assert "U+043F" in joined                # suspect codepoints visible
        assert "invalid tokens" in joined        # original error kept

    def test_sanitize_before_send(self, monkeypatch):
        from api.memory import pgvector_backend as pb

        received: list = []

        class _FakeEmbedder:
            def embed_documents(self, texts):
                received.extend(texts)
                return [[0.1] for _ in texts]

        self = TestEmbedBatch()
        self._install_embedder(monkeypatch, _FakeEmbedder())

        result = asyncio.run(pb._embed_batch(["a\x00b", "c\ud800d", "e"]))
        assert received == ["ab", "cd", "e"]
        assert result == [[0.1], [0.1], [0.1]]

    def test_empty_after_sanitize_skipped_without_request(self, monkeypatch):
        from api.memory import pgvector_backend as pb

        calls = {"n": 0}

        class _FakeEmbedder:
            def embed_documents(self, texts):
                calls["n"] += 1
                return [[0.1] for _ in texts]

        self = TestEmbedBatch()
        self._install_embedder(monkeypatch, _FakeEmbedder())

        result = asyncio.run(pb._embed_batch(["\x00\x01\x02"]))
        assert result == [None]
        assert calls["n"] == 0

    def test_400_rejection_isolated_via_bisection(self, monkeypatch):
        from api.memory import pgvector_backend as pb

        class _Rejected400(Exception):
            status_code = 400

        class _FakeEmbedder:
            def __init__(self):
                self.calls = 0

            def embed_documents(self, texts):
                self.calls += 1
                if any("POISON" in t for t in texts):
                    raise _Rejected400(
                        'Error code: 400 - {"message":"Prompt contains invalid tokens"}'
                    )
                return [[0.5, 0.6] for _ in texts]

        fake = _FakeEmbedder()
        self = TestEmbedBatch()
        self._install_embedder(monkeypatch, fake)

        result = asyncio.run(pb._embed_batch(["alpha", "POISON", "beta"]))
        assert result[0] == [0.5, 0.6]
        assert result[1] is None
        assert result[2] == [0.5, 0.6]
        # Full batch + bisect down to the poisoned single (+ good halves).
        assert fake.calls >= 3


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

    def test_index_on_legacy_schema_without_citation_columns(self, monkeypatch, isolated_db):
        """Pre-citation-column schema: the INSERT must not reference the
        missing columns. The ORM unit-of-work emits every mapped column
        (NULL for unset) and fails with UndefinedColumn on chunk_id; the
        Core insert with an explicit column list must store the chunks."""
        from api.memory import pgvector_backend as pb
        from api.models import ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.commit()
        finally:
            db.close()

        from api.db import engine as iso_engine
        with iso_engine.begin() as conn:
            conn.exec_driver_sql("DROP TABLE IF EXISTS knowledge_chunks")
            conn.exec_driver_sql(
                "CREATE TABLE knowledge_chunks ("
                " id VARCHAR(64) PRIMARY KEY,"
                " product_id VARCHAR(64) NOT NULL,"
                " source_type VARCHAR(32) NOT NULL,"
                " source_id VARCHAR(64),"
                " chunk_index INTEGER NOT NULL,"
                " content TEXT NOT NULL,"
                " embedding TEXT,"
                " created_at DATETIME NOT NULL)"
            )
        pb.reset_citation_columns_cache()
        assert pb._citation_columns_available() is False

        async def _fake_embed(texts):
            return [[float(i)] * 4 for i in range(len(texts))]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)

        be = pb.PgVectorMemoryBackend()
        n = asyncio.run(be.index(
            "legacy schema content. " * 50, "prod_1",
            source_type="codebase", source_id="cb_old",
        ))
        pb.reset_citation_columns_cache()
        assert n > 0

        # Count via raw SQL: an ORM SELECT would itself reference the
        # missing citation columns on this legacy schema.
        with iso_engine.connect() as conn:
            cnt = conn.exec_driver_sql(
                "SELECT COUNT(*) FROM knowledge_chunks"
            ).scalar()
        assert cnt == n

    def test_index_stores_sanitized_content(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb
        from api.models import KnowledgeChunkORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.commit()
        finally:
            db.close()

        async def _fake_embed(texts):
            return [[0.1, 0.2] for _ in texts]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)

        be = pb.PgVectorMemoryBackend()
        content = "hello\x00 world \ud800 something. " * 30
        n = asyncio.run(be.index(content, "prod_1", source_id="cb_1"))
        assert n > 0

        db = isolated_db.SessionLocal()
        try:
            rows = db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.source_id == "cb_1"
            ).all()
            assert len(rows) == n
            for r in rows:
                assert "\x00" not in r.content
                assert "\ud800" not in r.content
        finally:
            db.close()

    def test_index_drops_chunks_that_failed_embedding(self, monkeypatch, isolated_db, caplog):
        from api.memory import pgvector_backend as pb
        from api.models import KnowledgeChunkORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.commit()
        finally:
            db.close()

        async def _partial(texts):
            # Chunks carrying the marker are "rejected by the embedder".
            return [
                None if "POISON" in t else [0.1, 0.2] for t in texts
            ]

        monkeypatch.setattr(pb, "_embed_batch", _partial)

        be = pb.PgVectorMemoryBackend()
        content = (
            "POISON paragraph with marker words here. " * 30
            + "\n\n"
            + "normal paragraph with clean words here. " * 30
        )
        with caplog.at_level("WARNING"):
            n = asyncio.run(be.index(content, "prod_1", source_id="cb_2"))
        assert n > 0
        assert any("dropped" in rec.message for rec in caplog.records)

        db = isolated_db.SessionLocal()
        try:
            rows = db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.source_id == "cb_2"
            ).all()
            assert len(rows) == n
            assert all("POISON" not in r.content for r in rows)
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

    def test_cosine_rows_use_halfvec_expr_for_over_limit_dim(self, monkeypatch, isolated_db):
        """When the live index is the halfvec expression (dim > 2000), the
        cosine ORDER BY compares through the same halfvec cast so the query
        hits the index instead of erroring on a vector/halfvec mismatch."""
        import api.db as db_mod
        from api.memory import pgvector_backend as pb

        db_mod.reset_hnsw_state()
        db_mod._hnsw_distance_expr = (
            "(embedding::halfvec(2560)) <=> CAST(:q AS halfvec)", "halfvec",
        )
        try:
            monkeypatch.setattr(pb, "_is_pgvector_capable", lambda: True)

            async def _fake_embed_query(q):
                return [0.1, 0.2]

            monkeypatch.setattr(pb, "_embed_query", _fake_embed_query)

            executed: dict = {}

            class _FakeResult:
                def fetchall(self):
                    return [("chunk one",)]

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

            monkeypatch.setattr(db_mod, "engine", _FakeEngine())

            be = pb.PgVectorMemoryBackend()
            result = asyncio.run(be.query("how does auth work", "prod_1", top_k=5))
            assert "chunk one" in result
            assert "halfvec(2560)" in executed["sql"]
            assert "<=>" in executed["sql"]
            assert executed["params"]["q"].startswith("[")
        finally:
            db_mod.reset_hnsw_state()

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

    def test_pages_not_double_indexed_when_docs_present(self, monkeypatch, isolated_db):
        """generated_docs is indexed ONCE under the codebase id; pages are
        section slices of it and must not be queued as well (their upserts
        used to share the same source id and overwrite each other)."""
        from api.memory import pgvector_backend as pb
        from api.models import CodebaseORM, KnowledgeChunkORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo",
                generated_docs="full wiki document content " * 10,
                pages={"overview": {"content": "overview section"}},
            ))
            db.commit()
        finally:
            db.close()

        indexed_sources: list = []

        async def _spy_reindex_one(self, pid, items):
            # P1-20 keyset reindex: per-product swap seam (index() is no
            # longer called item-by-item from reindex_product).
            indexed_sources.extend(source_id for _, _, source_id in items)
            return True

        monkeypatch.setattr(
            pb.PgVectorMemoryBackend, "_reindex_one_product", _spy_reindex_one
        )

        be = pb.PgVectorMemoryBackend()
        result = asyncio.run(be.reindex_product("prod_1"))
        assert result["success"] is True
        assert result["reindexed_count"] == 1
        # Exactly ONE item for the codebase (docs, not docs+pages).
        assert indexed_sources == ["cb_1"]

    def test_legacy_pages_get_distinct_source_ids(self, monkeypatch, isolated_db):
        """A codebase with ONLY pages (no generated_docs) indexes each page
        under `<cb>::page::<id>` so upserts do not overwrite each other."""
        from api.memory import pgvector_backend as pb
        from api.models import CodebaseORM, KnowledgeChunkORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_2", product_id="prod_1", name="repo",
                generated_docs="",
                pages={
                    "overview": {"content": "overview section text"},
                    "api": {"content": "api section text"},
                },
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

        db = isolated_db.SessionLocal()
        try:
            sources = {
                r[0] for r in db.query(KnowledgeChunkORM.source_id).filter(
                    KnowledgeChunkORM.product_id == "prod_1"
                ).all()
            }
            assert sources == {"cb_2::page::overview", "cb_2::page::api"}
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

    # --- P1-20: pagination + atomic per-product swap -------------------------- #
    def test_pagination_reindexes_all_products(self, monkeypatch, isolated_db):
        """Keyset pages of 1 product each still cover every product (no skips,
        no infinite loop) and only ONE load happens per page."""
        from api.memory import pgvector_backend as pb
        from api.models import CodebaseORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            for i in range(3):
                db.add(ProductORM(id=f"prod_{i}", name=f"P{i}", description=""))
                db.add(CodebaseORM(
                    id=f"cb_{i}", product_id=f"prod_{i}", name="repo",
                    generated_docs=f"content for product {i} " * 8,
                ))
            db.commit()
        finally:
            db.close()

        async def _fake_embed(texts):
            return [[float(len(t))] * 4 for t in texts]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)
        monkeypatch.setattr(pb, "_REINDEX_PRODUCT_PAGE_SIZE", 1)

        be = pb.PgVectorMemoryBackend()
        result = asyncio.run(be.reindex_product())
        assert result["success"] is True
        assert result["reindexed_count"] == 3

        from api.models import KnowledgeChunkORM
        db = isolated_db.SessionLocal()
        try:
            for i in range(3):
                assert db.query(KnowledgeChunkORM).filter(
                    KnowledgeChunkORM.product_id == f"prod_{i}"
                ).count() > 0
        finally:
            db.close()

    def test_swap_is_atomic_old_chunks_replaced(self, monkeypatch, isolated_db):
        """A product with stale chunks gets exactly one atomic swap: old ids
        gone, new chunks present in a single pass."""
        from api.memory import pgvector_backend as pb
        from api.models import CodebaseORM, KnowledgeChunkORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo",
                generated_docs="fresh wiki content " * 10,
            ))
            # Stale chunk from a previous index run.
            db.add(KnowledgeChunkORM(
                id="chunk_stale", product_id="prod_1", chunk_index=0,
                content="stale",
            ))
            db.commit()
        finally:
            db.close()

        async def _fake_embed(texts):
            return [[float(len(t))] * 4 for t in texts]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)

        be = pb.PgVectorMemoryBackend()
        result = asyncio.run(be.reindex_product("prod_1"))
        assert result["success"] is True
        assert result["reindexed_count"] == 1

        db = isolated_db.SessionLocal()
        try:
            assert db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.id == "chunk_stale"
            ).count() == 0
            fresh = db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.product_id == "prod_1"
            ).count()
            assert fresh > 0
            # Every fresh row carries its source, so future per-source
            # upserts still work after the swap.
            for row in db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.product_id == "prod_1"
            ).all():
                assert row.source_type == "codebase"
                assert row.source_id == "cb_1"
        finally:
            db.close()

    def test_embedder_failure_keeps_previous_chunks(self, monkeypatch, isolated_db):
        """If nothing can be embedded the swap is skipped entirely — the
        product keeps its previous chunks instead of ending up empty."""
        from api.memory import pgvector_backend as pb
        from api.models import CodebaseORM, KnowledgeChunkORM, ProductORM

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.add(CodebaseORM(
                id="cb_1", product_id="prod_1", name="repo",
                generated_docs="wiki content " * 10,
            ))
            db.add(KnowledgeChunkORM(
                id="chunk_keep", product_id="prod_1", chunk_index=0,
                content="previous index data",
            ))
            db.commit()
        finally:
            db.close()

        async def _broken_embed(texts):
            return None  # total embedder failure

        monkeypatch.setattr(pb, "_embed_batch", _broken_embed)

        be = pb.PgVectorMemoryBackend()
        result = asyncio.run(be.reindex_product("prod_1"))
        assert result["success"] is True  # non-fatal
        assert result["reindexed_count"] == 0  # product not swapped

        db = isolated_db.SessionLocal()
        try:
            assert db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.id == "chunk_keep"
            ).count() == 1
        finally:
            db.close()

# --------------------------------------------------------------------------- #
# Wave D: _compute_char_spans (best-effort chunk offsets within the source)
# --------------------------------------------------------------------------- #
class TestComputeCharSpans:
    def test_located_sequential(self):
        from api.memory.pgvector_backend import _compute_char_spans

        content = "intro alpha beta middle gamma delta end"
        spans = _compute_char_spans(content, ["alpha beta", "gamma delta"])
        assert len(spans) == 2
        for span, chunk in zip(spans, ["alpha beta", "gamma delta"]):
            assert span is not None
            start, end = span
            assert content[start:end] == chunk

    def test_missing_chunk_gets_none(self):
        from api.memory.pgvector_backend import _compute_char_spans

        content = "head tail"
        spans = _compute_char_spans(content, ["not present at all", "tail"])
        assert spans[0] is None
        assert spans[1] == [content.find("tail"), content.find("tail") + len("tail")]

    def test_duplicate_chunk_finds_later_occurrence(self):
        from api.memory.pgvector_backend import _compute_char_spans

        content = "same same"
        spans = _compute_char_spans(content, ["same", "same"])
        assert spans[0] == [0, 4]
        assert spans[1] == [5, 9]

    def test_empty_chunks(self):
        from api.memory.pgvector_backend import _compute_char_spans

        assert _compute_char_spans("content", []) == []

    def test_overlap_chunks_located(self):
        """Overlap splitters emit chunk N+1 that starts INSIDE chunk N: a
        forward miss is retried from the previous chunk's start."""
        from api.memory.pgvector_backend import _compute_char_spans

        content = "alpha beta gamma delta"
        chunk1 = "alpha beta gamma"
        chunk2 = "beta gamma delta"  # begins inside chunk1 (offset 6)
        spans = _compute_char_spans(content, [chunk1, chunk2])
        assert spans[0] == [0, len(chunk1)]
        assert spans[1] is not None
        start, end = spans[1]
        assert content[start:end] == chunk2


# --------------------------------------------------------------------------- #
# Wave D: citation columns probe + graceful degradation
# --------------------------------------------------------------------------- #
class TestCitationColumns:
    def test_available_on_fresh_schema(self, isolated_db):
        """A fresh schema (create_all) carries the Wave D citation columns.

        The probe result is cached in a module-level dict — always reset it so
        a previous test's environment cannot leak into this one.
        """
        from api.memory import pgvector_backend as pb

        pb.reset_citation_columns_cache()
        assert pb._citation_columns_available() is True

    def test_probe_cache_reset(self, isolated_db):
        from api.memory import pgvector_backend as pb

        pb.reset_citation_columns_cache()
        assert pb._citation_columns_cache["available"] is None

    def test_graceful_degradation_when_columns_absent(self, monkeypatch, isolated_db):
        """When the live table lacks the columns (pre-Wave-D install) the
        upsert still stores the chunks — citation metadata is skipped."""
        from api.memory import pgvector_backend as pb
        from api.models import KnowledgeChunkORM, ProductORM

        monkeypatch.setattr(pb, "_citation_columns_available", lambda: False)

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
        n = asyncio.run(be.index(
            "some content here " * 50, "prod_1",
            source_type="codebase", source_id="cb_1", source_path="docs/x.md",
        ))
        assert n > 0

        db = isolated_db.SessionLocal()
        try:
            rows = db.query(KnowledgeChunkORM).filter(
                KnowledgeChunkORM.source_id == "cb_1"
            ).all()
            assert len(rows) == n
            # No citation metadata written (columns treated as absent).
            assert all(r.chunk_id is None for r in rows)
            assert all(r.source_path is None for r in rows)
            assert all(r.char_span is None for r in rows)
        finally:
            db.close()


class TestIndexCitations:
    """The index path stores chunk_id / source_path / char_span when the
    schema supports them (Wave D provenance for recall-time citations)."""

    def test_citation_metadata_stored(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb
        from api.models import KnowledgeChunkORM, ProductORM

        pb.reset_citation_columns_cache()

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.commit()
        finally:
            db.close()

        # Deterministic chunks so the expected spans are exactly computable.
        monkeypatch.setattr(pb, "_split_text", lambda content: ["alpha beta", "gamma delta"])

        async def _fake_embed(texts):
            return [[float(i)] * 4 for i in range(len(texts))]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)

        content = "intro alpha beta middle gamma delta end"
        be = pb.PgVectorMemoryBackend()
        n = asyncio.run(be.index(
            content, "prod_1",
            source_type="codebase", source_id="cb_1", source_path="docs/main.md",
        ))
        assert n == 2

        db = isolated_db.SessionLocal()
        try:
            rows = (
                db.query(KnowledgeChunkORM)
                .filter(KnowledgeChunkORM.source_id == "cb_1")
                .order_by(KnowledgeChunkORM.chunk_index)
                .all()
            )
            assert len(rows) == 2
            for i, row in enumerate(rows):
                assert row.chunk_id == f"c:cb_1:{i}"
                assert row.source_path == "docs/main.md"
                assert isinstance(row.char_span, list) and len(row.char_span) == 2
                start, end = row.char_span
                assert content[start:end] == row.content
            # Exact offsets: sequential find positions.
            first = rows[0].char_span
            second = rows[1].char_span
            assert first == [content.find("alpha beta")] + [content.find("alpha beta") + len("alpha beta")]
            assert second[0] > first[1]  # cursor advanced past the first chunk
        finally:
            db.close()

    def test_unlocatable_chunk_span_none(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb
        from api.models import KnowledgeChunkORM, ProductORM

        pb.reset_citation_columns_cache()

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.commit()
        finally:
            db.close()

        # The splitter is mocked so chunk 0 does NOT appear in the source.
        monkeypatch.setattr(pb, "_split_text", lambda content: ["ghost text", "tail"])

        async def _fake_embed(texts):
            return [[float(i)] * 4 for i in range(len(texts))]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)

        be = pb.PgVectorMemoryBackend()
        n = asyncio.run(be.index(
            "head tail", "prod_1", source_type="spec", source_id="spec_1",
        ))
        assert n == 2

        db = isolated_db.SessionLocal()
        try:
            rows = (
                db.query(KnowledgeChunkORM)
                .filter(KnowledgeChunkORM.source_id == "spec_1")
                .order_by(KnowledgeChunkORM.chunk_index)
                .all()
            )
            assert rows[0].chunk_id == "c:spec_1:0"
            assert rows[0].char_span is None  # unlocatable -> None, never fatal
            assert rows[1].char_span is not None
            assert "head tail"[rows[1].char_span[0]:rows[1].char_span[1]] == "tail"
        finally:
            db.close()


# --------------------------------------------------------------------------- #
# HNSW index: the embedding column is dimensionless by design, so the pin
# (vector(dim)) + index happen on the FIRST index run, once the embedder
# dimension is known (pgvector cannot index a dimensionless column).
# --------------------------------------------------------------------------- #
class TestHnswPin:
    def _install(self, monkeypatch, typmod):
        """Point api.db at a fake Postgres engine whose knowledge_chunks
        embedding column reports the given pg_attribute.typmod."""
        import api.db as db_mod

        statements = []

        class _Result:
            def __init__(self, value):
                self._value = value

            def scalar(self):
                return self._value

        class _FakeConn:
            def execute(self, sql, params=None):
                statements.append(("execute", str(sql)))
                return _Result(typmod)

            def exec_driver_sql(self, sql, params=None):
                statements.append(("driver", sql))

        class _BeginCtx:
            def __enter__(self):
                return _FakeConn()

            def __exit__(self, *exc):
                return False

        class _FakeEngine:
            def begin(self):
                return _BeginCtx()

        monkeypatch.setattr(db_mod, "DB_PROVIDER", "postgres")
        monkeypatch.setattr(db_mod, "engine", _FakeEngine())
        db_mod.reset_hnsw_state()
        return db_mod, statements

    def teardown_method(self):
        import api.db as db_mod

        db_mod.reset_hnsw_state()

    def _ddl(self, statements):
        return [s for kind, s in statements if kind == "driver"]

    def test_dimensionless_column_pinned_and_indexed(self, monkeypatch):
        db_mod, statements = self._install(monkeypatch, typmod=-1)
        assert db_mod.ensure_embedding_dimension_and_hnsw(768) is True
        ddl = self._ddl(statements)
        assert ddl[0].startswith("ALTER TABLE knowledge_chunks")
        assert "vector(768)" in ddl[0]
        assert ddl[1].startswith("CREATE INDEX IF NOT EXISTS")
        assert "hnsw" in ddl[1]
        # Cached success: a second call issues no statements.
        before = len(statements)
        assert db_mod.ensure_embedding_dimension_and_hnsw(768) is True
        assert len(statements) == before

    def test_already_pinned_same_dim_only_creates_index(self, monkeypatch):
        db_mod, statements = self._install(monkeypatch, typmod=768 + 4)
        assert db_mod.ensure_embedding_dimension_and_hnsw(768) is True
        ddl = self._ddl(statements)
        assert len(ddl) == 1
        assert ddl[0].startswith("CREATE INDEX")

    def test_already_pinned_bare_typmod_convention(self, monkeypatch):
        """Current pgvector builds store the bare dim as typmod (768, not
        dim+4): a correct pin must not be misreported as a mismatch."""
        db_mod, statements = self._install(monkeypatch, typmod=768)
        assert db_mod.ensure_embedding_dimension_and_hnsw(768) is True
        ddl = self._ddl(statements)
        assert len(ddl) == 1
        assert ddl[0].startswith("CREATE INDEX")

    def test_dimension_mismatch_is_cached_failure(self, monkeypatch, caplog):
        db_mod, statements = self._install(monkeypatch, typmod=768 + 4)
        with caplog.at_level("WARNING"):
            assert db_mod.ensure_embedding_dimension_and_hnsw(384) is False
        assert self._ddl(statements) == []  # no ALTER, no index
        assert "reindex" in caplog.text
        # Cached failure: a retry issues no statements and no new warning.
        before = len(statements)
        assert db_mod.ensure_embedding_dimension_and_hnsw(384) is False
        assert len(statements) == before

    def test_dim_over_halfvec_limit_rejected_without_engine(self, monkeypatch, caplog):
        """Above halfvec's 4000-dim HNSW limit there is nothing to index:
        rejected before any engine round-trip."""
        db_mod, statements = self._install(monkeypatch, typmod=-1)
        with caplog.at_level("WARNING"):
            assert db_mod.ensure_embedding_dimension_and_hnsw(4001) is False
        assert statements == []
        assert "1-4000" in caplog.text

    def test_dim_over_vector_limit_pins_halfvec_index(self, monkeypatch):
        """2560-dim embedders (Qwen3-Embedding-4B): the pin stays vector(2560)
        and the HNSW index switches to the halfvec cast expression."""
        db_mod, statements = self._install(monkeypatch, typmod=-1)
        assert db_mod.ensure_embedding_dimension_and_hnsw(2560) is True
        ddl = self._ddl(statements)
        assert "vector(2560)" in ddl[0]
        assert "halfvec(2560)" in ddl[1]
        assert "halfvec_cosine_ops" in ddl[1]

    def test_already_pinned_over_limit_only_creates_halfvec_index(self, monkeypatch):
        db_mod, statements = self._install(monkeypatch, typmod=2560)
        assert db_mod.ensure_embedding_dimension_and_hnsw(2560) is True
        ddl = self._ddl(statements)
        assert len(ddl) == 1
        assert "halfvec(2560)" in ddl[0]

    def test_startup_halfvec_when_pinned_over_limit(self, monkeypatch):
        # typmod 2564 = 2560 + 4 (varlena convention): startup derives 2560.
        db_mod, statements = self._install(monkeypatch, typmod=2564)
        db_mod._ensure_hnsw_index()
        ddl = self._ddl(statements)
        assert len(ddl) == 1
        assert "halfvec(2560)" in ddl[0]

    def test_sqlite_provider_is_noop(self, monkeypatch):
        import api.db as db_mod

        db_mod.reset_hnsw_state()
        monkeypatch.setattr(db_mod, "DB_PROVIDER", "sqlite")
        assert db_mod.ensure_embedding_dimension_and_hnsw(768) is False

    def test_startup_defers_when_dimensionless(self, monkeypatch, caplog):
        db_mod, statements = self._install(monkeypatch, typmod=-1)
        with caplog.at_level("INFO"):
            db_mod._ensure_hnsw_index()
        # No CREATE INDEX attempt on a dimensionless column (the old code
        # warned "column does not have dimensions" on every startup).
        assert self._ddl(statements) == []
        assert "deferred" in caplog.text.lower()

    def test_startup_indexes_when_already_pinned(self, monkeypatch):
        db_mod, statements = self._install(monkeypatch, typmod=768 + 4)
        db_mod._ensure_hnsw_index()
        ddl = self._ddl(statements)
        assert len(ddl) == 1 and ddl[0].startswith("CREATE INDEX")

    def test_index_run_pins_the_dimension(self, monkeypatch, isolated_db):
        """The backend's index() feeds the first real embedding batch's
        dimension into the pin helper."""
        import api.db as db_mod
        from api.memory import pgvector_backend as pb
        from api.models import ProductORM

        calls = []

        def _record(dim):
            calls.append(dim)
            return True

        monkeypatch.setattr(db_mod, "ensure_embedding_dimension_and_hnsw", _record)

        db = isolated_db.SessionLocal()
        try:
            db.add(ProductORM(id="prod_1", name="P1", description=""))
            db.commit()
        finally:
            db.close()

        async def _fake_embed(texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        monkeypatch.setattr(pb, "_embed_batch", _fake_embed)

        be = pb.PgVectorMemoryBackend()
        n = asyncio.run(be.index("some content here " * 50, "prod_1"))
        assert n > 0
        assert calls == [4]


# --------------------------------------------------------------------------- #
# embedding_distance_expr: the cosine ORDER BY must match the live index kind
# (plain vector, or the halfvec cast pair when the column was pinned beyond
# pgvector's 2000-dim vector-index limit).
# --------------------------------------------------------------------------- #
class TestEmbeddingDistanceExpr:
    def teardown_method(self):
        import api.db as db_mod

        db_mod.reset_hnsw_state()

    def _install(self, monkeypatch, scalars, provider="postgres"):
        """Fake connect()-engine whose execute().scalar() pops queued values."""
        import api.db as db_mod

        executed = []

        class _Result:
            def __init__(self, value):
                self._value = value

            def scalar(self):
                return self._value

        class _FakeConn:
            def execute(self, sql, params=None):
                executed.append(str(sql))
                return _Result(scalars.pop(0) if scalars else None)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class _FakeEngine:
            def connect(self):
                return _FakeConn()

        monkeypatch.setattr(db_mod, "DB_PROVIDER", provider)
        monkeypatch.setattr(db_mod, "engine", _FakeEngine())
        db_mod.reset_hnsw_state()
        return db_mod, executed

    def test_halfvec_pair_when_index_is_halfvec(self, monkeypatch):
        indexdef = (
            "CREATE INDEX ix_knowledge_chunks_embedding_hnsw "
            "ON public.knowledge_chunks USING hnsw "
            "((embedding::halfvec(2560)) halfvec_cosine_ops)"
        )
        db_mod, executed = self._install(monkeypatch, [2564, indexdef])
        expr, cast = db_mod.embedding_distance_expr()
        assert expr == "(embedding::halfvec(2560)) <=> CAST(:q AS halfvec)"
        assert cast == "halfvec"
        # Cached: a second call issues no further queries.
        n = len(executed)
        assert db_mod.embedding_distance_expr() == (expr, cast)
        assert len(executed) == n

    def test_vector_pair_within_vector_limit(self, monkeypatch):
        db_mod, executed = self._install(monkeypatch, [768])
        expr, cast = db_mod.embedding_distance_expr()
        assert expr == "embedding <=> CAST(:q AS vector)"
        assert cast == "vector"
        # typmod within the vector limit: pg_indexes is never consulted.
        assert len(executed) == 1

    def test_vector_pair_when_indexdef_not_halfvec(self, monkeypatch):
        db_mod, _ = self._install(
            monkeypatch, [2564, "CREATE INDEX ... (embedding vector_cosine_ops)"],
        )
        expr, cast = db_mod.embedding_distance_expr()
        assert expr == "embedding <=> CAST(:q AS vector)"
        assert cast == "vector"

    def test_vector_pair_on_sqlite_without_engine(self, monkeypatch):
        db_mod, executed = self._install(monkeypatch, [], provider="sqlite")
        assert db_mod.embedding_distance_expr() == (
            "embedding <=> CAST(:q AS vector)", "vector",
        )
        assert executed == []

    def test_introspection_failure_falls_back_to_vector(self, monkeypatch):
        import api.db as db_mod

        class _Boom:
            def connect(self):
                raise RuntimeError("down")

        monkeypatch.setattr(db_mod, "DB_PROVIDER", "postgres")
        monkeypatch.setattr(db_mod, "engine", _Boom())
        db_mod.reset_hnsw_state()
        assert db_mod.embedding_distance_expr() == (
            "embedding <=> CAST(:q AS vector)", "vector",
        )

    def test_hnsw_index_sql_switches_to_halfvec(self):
        import api.db as db_mod

        assert "halfvec(2560)" in db_mod._hnsw_index_sql(2560)
        assert "halfvec_cosine_ops" in db_mod._hnsw_index_sql(2560)
        assert "halfvec" not in db_mod._hnsw_index_sql(2000)
        assert "vector_cosine_ops" in db_mod._hnsw_index_sql(2000)
