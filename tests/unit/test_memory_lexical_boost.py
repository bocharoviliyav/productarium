"""Lexical boost for expert recall (fork port 3.2).

Covers the pure helpers (flag, token extraction, ILIKE escaping, lexical
re-ranking, reciprocal rank fusion) and the ``PgVectorMemoryBackend.query``
integration: SQL shapes of both legs, the RRF merge, the flag-off fallback to
pure cosine, the embedder-down fallback to lexical hits, and the degradation
when the lexical DB stage fails.
"""
import asyncio

from api.memory.lexical_boost import (
    escape_ilike,
    extract_query_tokens,
    lexical_boost_enabled,
    rank_lexical_rows,
    reciprocal_rank_fusion,
)


# --------------------------------------------------------------------------- #
# Flag
# --------------------------------------------------------------------------- #
class TestLexicalBoostEnabled:
    def test_default_on(self, monkeypatch):
        monkeypatch.delenv("EXPERT_LEXICAL_BOOST", raising=False)
        assert lexical_boost_enabled() is True

    def test_explicit_on_variants(self, monkeypatch):
        for raw in ("1", "true", "TRUE", "Yes", " on "):
            monkeypatch.setenv("EXPERT_LEXICAL_BOOST", raw)
            assert lexical_boost_enabled() is True, raw

    def test_explicit_off_variants(self, monkeypatch):
        for raw in ("0", "false", "False", "off", "no", ""):
            monkeypatch.setenv("EXPERT_LEXICAL_BOOST", raw)
            assert lexical_boost_enabled() is False, raw

    def test_garbage_is_off(self, monkeypatch):
        monkeypatch.setenv("EXPERT_LEXICAL_BOOST", "maybe")
        assert lexical_boost_enabled() is False


# --------------------------------------------------------------------------- #
# Token extraction
# --------------------------------------------------------------------------- #
class TestExtractQueryTokens:
    def test_backticked_single_word_table_name(self):
        # Bare single words never pass the compound filter — the backtick is
        # what marks `orders` as an exact name worth an ILIKE scan.
        assert extract_query_tokens("как устроена таблица `orders`?") == ["orders"]

    def test_backticked_dotted_name(self):
        assert extract_query_tokens("что лежит в `public.users`?") == ["public.users"]

    def test_backticked_path(self):
        assert extract_query_tokens("смотри `src/main.py`") == ["src/main.py"]

    def test_bare_camel_identifier(self):
        assert extract_query_tokens("как работает sendSmsCode") == ["sendSmsCode"]

    def test_bare_snake_identifier(self):
        assert extract_query_tokens("где используется user_session_log") == [
            "user_session_log"
        ]

    def test_russian_prose_yields_no_tokens(self):
        assert extract_query_tokens("как работает авторизация в этой системе") == []

    def test_english_prose_yields_no_tokens(self):
        # Single bare words are prose, not identifiers.
        assert extract_query_tokens("how does auth work here") == []

    def test_dedup_case_insensitive(self):
        toks = extract_query_tokens("sendSmsCode и ещё раз SendSmsCode")
        assert toks == ["sendSmsCode"]

    def test_limit_caps_tokens(self):
        query = " ".join(f"token_{i}_name" for i in range(10))
        assert len(extract_query_tokens(query, limit=3)) == 3

    def test_short_backticked_span_dropped(self):
        assert extract_query_tokens("что такое `ab`?") == []

    def test_empty_query(self):
        assert extract_query_tokens("") == []
        assert extract_query_tokens("   ") == []


# --------------------------------------------------------------------------- #
# ILIKE escaping
# --------------------------------------------------------------------------- #
class TestEscapeIlike:
    def test_underscores_escaped(self):
        assert escape_ilike("user_session_log") == r"user\_session\_log"

    def test_percent_escaped(self):
        assert escape_ilike("50%") == r"50\%"

    def test_backslash_escaped_first(self):
        assert escape_ilike("a\\b") == "a\\\\b"

    def test_plain_token_unchanged(self):
        assert escape_ilike("orders") == "orders"

    def test_empty(self):
        assert escape_ilike("") == ""


# --------------------------------------------------------------------------- #
# Lexical re-ranking
# --------------------------------------------------------------------------- #
class TestRankLexicalRows:
    def test_orders_by_occurrence_count(self):
        rows = [
            "mentions audit_log once here",
            "no match in this row",
            "audit_log audit_log twice here",
        ]
        assert rank_lexical_rows(rows, ["audit_log"], top_k=3) == [
            "audit_log audit_log twice here",
            "mentions audit_log once here",
            "no match in this row",
        ]

    def test_top_k_cap(self):
        rows = ["a token here", "b token here", "c token here"]
        assert rank_lexical_rows(rows, ["token"], top_k=2) == [
            "a token here",
            "b token here",
        ]

    def test_ties_keep_input_order(self):
        rows = ["r1 x", "r2 x", "r3 x"]
        assert rank_lexical_rows(rows, ["x"], top_k=3) == rows

    def test_no_tokens_returns_empty(self):
        assert rank_lexical_rows(["row"], [], top_k=3) == []

    def test_no_rows_returns_empty(self):
        assert rank_lexical_rows([], ["tok"], top_k=3) == []


# --------------------------------------------------------------------------- #
# Reciprocal rank fusion
# --------------------------------------------------------------------------- #
class TestReciprocalRankFusion:
    def test_overlap_wins(self):
        merged = reciprocal_rank_fusion(["a", "b"], ["c", "a"])
        assert merged[0] == "a"  # present in both legs
        assert set(merged) == {"a", "b", "c"}

    def test_non_overlapping_order(self):
        # a: 1/61; b: 1/62; c: 1/61 — ties broken alphabetically.
        assert reciprocal_rank_fusion(["a", "b"], ["c"]) == ["a", "c", "b"]

    def test_dedup_across_lists(self):
        assert reciprocal_rank_fusion(["x"], ["x"]) == ["x"]

    def test_limit(self):
        assert reciprocal_rank_fusion(["a", "b", "c"], limit=2) == ["a", "b"]

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_skips_empty_items(self):
        assert reciprocal_rank_fusion(["", "a"], [""]) == ["a"]

    def test_tie_break_is_deterministic(self):
        assert reciprocal_rank_fusion(["b"], ["a"]) == ["a", "b"]


# --------------------------------------------------------------------------- #
# PgVectorMemoryBackend.query integration (fake engine, SQL-shape asserts)
# --------------------------------------------------------------------------- #
class _DispatchEngine:
    """Fake engine dispatching by SQL shape: ILIKE → lexical rows, else cosine."""

    def __init__(self, cosine_rows, lexical_rows, executed=None, lexical_error=None):
        self._cosine_rows = cosine_rows
        self._lexical_rows = lexical_rows
        self._executed = executed if executed is not None else []
        self._lexical_error = lexical_error

    def connect(self):
        engine = self

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params):
                sql_str = str(sql)
                engine._executed.append({"sql": sql_str, "params": params})
                if "ILIKE" in sql_str:
                    if engine._lexical_error is not None:
                        raise engine._lexical_error
                    rows = [(r,) for r in engine._lexical_rows]
                else:
                    rows = [(r,) for r in engine._cosine_rows]

                class _Result:
                    def fetchall(self):
                        return rows

                return _Result()

        return _Conn()


class TestQueryLexicalIntegration:
    def _setup(self, monkeypatch, engine):
        from api.memory import pgvector_backend as pb
        import api.db as db_mod

        monkeypatch.setattr(pb, "_is_pgvector_capable", lambda: True)

        async def _fake_embed_query(q):
            return [0.1, 0.2, 0.3]

        monkeypatch.setattr(pb, "_embed_query", _fake_embed_query)
        monkeypatch.setattr(db_mod, "engine", engine)
        monkeypatch.setenv("EXPERT_LEXICAL_BOOST", "true")
        return pb.PgVectorMemoryBackend()

    def test_merge_rrf_overlap_first(self, monkeypatch, isolated_db):
        executed = []
        engine = _DispatchEngine(
            cosine_rows=["both-chunk", "cos-only"],
            lexical_rows=["both-chunk", "lex-only"],
            executed=executed,
        )
        be = self._setup(monkeypatch, engine)

        result = asyncio.run(be.query("как устроена таблица `orders`?", "prod_1", top_k=5))

        parts = result.split("\n\n")
        # Agreement wins; the rest follow by RRF score with deterministic ties.
        assert parts[0] == "both-chunk"
        assert set(parts) == {"both-chunk", "cos-only", "lex-only"}
        assert len(parts) == 3
        # Both legs executed.
        sqls = [e["sql"] for e in executed]
        assert any("<=>" in s for s in sqls)
        assert any("ILIKE" in s for s in sqls)

    def test_lexical_sql_shape(self, monkeypatch, isolated_db):
        executed = []
        engine = _DispatchEngine(
            cosine_rows=["cos"], lexical_rows=["lex"], executed=executed
        )
        be = self._setup(monkeypatch, engine)

        asyncio.run(be.query("где `user_session_log`?", "prod_1", top_k=5))

        lex = next(e for e in executed if "ILIKE" in e["sql"])
        assert "knowledge_chunks" in lex["sql"]
        assert "ILIKE ANY (:toks)" in lex["sql"]
        assert "ORDER BY id" in lex["sql"]
        assert "LIMIT :cap" in lex["sql"]
        assert lex["params"]["pid"] == "prod_1"
        assert lex["params"]["toks"] == [r"user\_session\_log"]
        assert lex["params"]["cap"] == 200

    def test_flag_off_is_pure_cosine(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb
        import api.db as db_mod

        monkeypatch.setattr(pb, "_is_pgvector_capable", lambda: True)

        async def _fake_embed_query(q):
            return [0.1]

        monkeypatch.setattr(pb, "_embed_query", _fake_embed_query)
        executed = []
        monkeypatch.setattr(
            db_mod, "engine", _DispatchEngine(["cos one", "cos two"], ["lex"], executed)
        )
        monkeypatch.setenv("EXPERT_LEXICAL_BOOST", "false")

        be = pb.PgVectorMemoryBackend()
        result = asyncio.run(be.query("как устроена таблица `orders`?", "prod_1"))

        # Only the cosine leg ran; the result is the pure cosine ranking.
        assert all("ILIKE" not in e["sql"] for e in executed)
        assert result == "cos one\n\ncos two"

    def test_embedder_down_serves_lexical_hits(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb
        import api.db as db_mod

        monkeypatch.setattr(pb, "_is_pgvector_capable", lambda: True)

        async def _no_vec(q):
            return None

        monkeypatch.setattr(pb, "_embed_query", _no_vec)
        monkeypatch.setattr(
            db_mod, "engine", _DispatchEngine(["cos"], ["lex one", "lex two"])
        )
        monkeypatch.setenv("EXPERT_LEXICAL_BOOST", "true")

        be = pb.PgVectorMemoryBackend()
        result = asyncio.run(be.query("как устроена таблица `orders`?", "prod_1"))

        assert result == "lex one\n\nlex two"

    def test_embedder_down_no_tokens_returns_empty(self, monkeypatch, isolated_db):
        from api.memory import pgvector_backend as pb
        import api.db as db_mod

        monkeypatch.setattr(pb, "_is_pgvector_capable", lambda: True)

        async def _no_vec(q):
            return None

        monkeypatch.setattr(pb, "_embed_query", _no_vec)
        monkeypatch.setattr(db_mod, "engine", _DispatchEngine(["cos"], ["lex"]))
        monkeypatch.setenv("EXPERT_LEXICAL_BOOST", "true")

        be = pb.PgVectorMemoryBackend()
        assert asyncio.run(be.query("q", "prod_1")) == ""

    def test_lexical_db_error_degrades_to_cosine(self, monkeypatch, isolated_db, caplog):
        executed = []
        engine = _DispatchEngine(
            cosine_rows=["cos one", "cos two"],
            lexical_rows=["lex"],
            executed=executed,
            lexical_error=RuntimeError("lexical stage down"),
        )
        be = self._setup(monkeypatch, engine)

        with caplog.at_level("WARNING"):
            result = asyncio.run(
                be.query("как устроена таблица `orders`?", "prod_1", top_k=5)
            )

        # The lexical leg failed but returned [] — the cosine ranking still
        # serves the query (merged = cosine-only RRF).
        assert result == "cos one\n\ncos two"
        assert any("lexical boost failed" in rec.message for rec in caplog.records)

    def test_prose_query_skips_lexical_leg(self, monkeypatch, isolated_db):
        executed = []
        engine = _DispatchEngine(
            cosine_rows=["cos one", "cos two"], lexical_rows=["lex"], executed=executed
        )
        be = self._setup(monkeypatch, engine)

        result = asyncio.run(be.query("how does auth work", "prod_1", top_k=5))

        # No code-like tokens → no ILIKE stage; pure cosine behavior.
        assert all("ILIKE" not in e["sql"] for e in executed)
        assert result == "cos one\n\ncos two"
