#!/usr/bin/env python3
"""Unit tests for api.memory.knowledge_graph (SPEC-008).

Pure graph ops (upsert/merge/prune/budget caps), term resolution + source
boosts, subgraph views, tolerant JSON parsing, batched extraction with a mock
LLM, SQLite storage roundtrips, incremental merges via
``update_graph_from_pages``, the graph-boost recall leg and the RMS
disambiguation acceptance scenario (same term, two products, strict scope).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from api.memory import knowledge_graph as kg


@pytest.fixture(autouse=True)
def _clean_graph_state():
    """No graph cache / lock leakage between tests."""
    kg.clear_graph_cache()
    kg._graph_lock = None
    yield
    kg.clear_graph_cache()
    kg._graph_lock = None


def _seed_product(db_mod, product_id: str, name: str = "P"):
    from api.models import ProductORM

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id=product_id, name=name, summary=""))
        db.commit()


def _graph_with_rms(artifact_id: str, description: str) -> dict:
    graph = kg.new_graph()
    tkey = kg.upsert_term(graph, "rms", description=description)
    pkey = kg.upsert_page(graph, "codebase", artifact_id, "p1", title="Overview")
    graph["edges"][kg.edge_key(tkey, "mentions", pkey)] = {
        "src": tkey, "dst": pkey, "rel": "mentions",
        "weight": 0.9, "context": "", "updated_at": kg.utc_now_iso(),
    }
    return graph


# --- pure graph ops -----------------------------------------------------------
class TestPureGraphOps:
    def test_upsert_term_normalizes_and_caps_aliases(self):
        graph = kg.new_graph()
        key = kg.upsert_term(graph, "  Root   Mean Square ", aliases=["RMSE", "rmse", "Mean Sq"])
        assert key == "term:root mean square"
        node = graph["nodes"][key]
        assert node["name"] == "root mean square"
        assert node["description"] == ""
        # Duplicate alias (case-insensitive) and the name itself are dropped.
        assert node["aliases"] == ["rmse", "mean sq"]
        assert kg.upsert_term(graph, "  ") == ""

    def test_set_page_terms_replaces_mentions(self):
        graph = kg.new_graph()
        pkey = kg.upsert_page(graph, "codebase", "cb1", "p1")
        kg.set_page_terms(graph, pkey, [
            {"term": "alpha", "weight": 0.9},
            {"term": "beta", "weight": 4.0, "context": "x" * 500},
        ])
        # beta's out-of-range weight clamps to 1.0 and its context is capped.
        beta = graph["edges"][kg.edge_key("term:beta", "mentions", pkey)]
        assert beta["weight"] == 1.0 and len(beta["context"]) == 200
        # Re-set with only alpha: beta's mention edge dies; the now-orphan
        # term node is pruned by the caller (prune_orphan_terms).
        kg.set_page_terms(graph, pkey, [{"term": "alpha", "weight": 0.5}])
        mentions = [e for e in graph["edges"].values() if e["rel"] == "mentions"]
        assert len(mentions) == 1 and mentions[0]["src"] == "term:alpha"
        assert kg.prune_orphan_terms(graph) == 1
        assert "term:beta" not in graph["nodes"]

    def test_add_term_relations_validates(self):
        graph = kg.new_graph()
        added = kg.add_term_relations(graph, [
            {"from": "billing", "to": "invoicing", "kind": "synonym"},
            {"from": "rms", "to": "rms", "kind": "related"},       # self
            {"from": "a", "to": "b", "kind": "nonsense"},           # bad kind
            {"from": "auth", "to": "iam", "kind": "Part_Of"},       # normalized
        ])
        assert added == 2
        assert kg.edge_key("term:auth", "part_of", "term:iam") in graph["edges"]
        # Unknown endpoints became bare term nodes.
        assert graph["nodes"]["term:billing"]["description"] == ""

    def test_prune_artifact_pages_drops_dead_pages_and_terms(self):
        graph = kg.new_graph()
        p1 = kg.upsert_page(graph, "codebase", "cb1", "p1")
        p2 = kg.upsert_page(graph, "codebase", "cb1", "p2")
        kg.set_page_terms(graph, p1, [{"term": "keepme", "weight": 0.5}])
        kg.set_page_terms(graph, p2, [{"term": "deadterm", "weight": 0.5}])
        removed = kg.prune_artifact_pages(graph, "codebase", "cb1", {"p1"})
        assert removed == 1
        assert p2 not in graph["nodes"]
        assert "term:deadterm" not in graph["nodes"]  # orphan pruned
        assert "term:keepme" in graph["nodes"]

    def test_enforce_budget_trims_weakest_edges(self, monkeypatch):
        graph = kg.new_graph()
        monkeypatch.setattr(kg, "MAX_EDGES", 3)
        pkey = kg.upsert_page(graph, "codebase", "cb1", "p1")
        for i in range(6):
            tkey = kg.upsert_term(graph, f"term{i}")
            graph["edges"][kg.edge_key(tkey, "mentions", pkey)] = {
                "src": tkey, "dst": pkey, "rel": "mentions",
                "weight": i / 10, "context": "", "updated_at": kg.utc_now_iso(),
            }
        kg.enforce_budget(graph)
        assert len(graph["edges"]) == 3
        weights = sorted(e["weight"] for e in graph["edges"].values())
        assert weights == [0.3, 0.4, 0.5]  # weakest LRU-trimmed
        # Terms that lost their only edge are pruned.
        assert len([n for n in graph["nodes"].values() if n["type"] == "term"]) == 3


# --- retrieval helpers ---------------------------------------------------------
class TestResolveAndBoosts:
    def _graph(self):
        graph = _graph_with_rms("cb1", "revenue management system")
        tkey = kg.upsert_term(graph, "sso", aliases=["single sign-on"])
        p2 = kg.upsert_page(graph, "codebase", "cb1", "p2")
        graph["edges"][kg.edge_key(tkey, "mentions", p2)] = {
            "src": tkey, "dst": p2, "rel": "mentions",
            "weight": 0.4, "context": "", "updated_at": kg.utc_now_iso(),
        }
        return graph

    def test_resolve_exact_name_and_alias(self):
        keys = kg.resolve_query_terms("How does RMS work?", self._graph())
        assert "term:rms" in keys

        keys = kg.resolve_query_terms("configure SSO please", self._graph())
        assert "term:sso" in keys

    def test_resolve_no_graph_or_no_tokens(self):
        assert kg.resolve_query_terms("rms", None) == []
        assert kg.resolve_query_terms("", self._graph()) == []

    def test_related_source_boosts_covers_page_and_legacy_sources(self):
        boosts = kg.related_source_boosts(self._graph(), ["term:rms"])
        assert boosts == {"cb1::page::p1": 0.9, "cb1": 0.9}
        assert kg.related_source_boosts(self._graph(), []) == {}
        assert kg.related_source_boosts(None, ["term:rms"]) == {}

    def test_subgraph_focus_and_capped_whole(self):
        graph = self._graph()
        focused = kg.subgraph(graph, focus_term="RMS")
        assert set(focused["nodes"]) == {"term:rms", "page:codebase:cb1:p1"}
        assert set(focused["edges"]) == {kg.edge_key("term:rms", "mentions", "page:codebase:cb1:p1")}
        # Unknown focus term -> empty graph; alias hit works.
        assert kg.subgraph(graph, focus_term="nope")["nodes"] == {}
        assert "term:sso" in kg.subgraph(graph, focus_term="single sign-on")["nodes"]
        # Whole graph capped by max_nodes keeps the best-connected nodes.
        capped = kg.subgraph(graph, max_nodes=1)
        assert len(capped["nodes"]) == 1
        assert kg.subgraph(None)["nodes"] == {}


# --- LLM extraction -------------------------------------------------------------
class TestExtraction:
    def test_parse_extraction_tolerates_prose(self):
        raw = 'Sure! Here is the JSON:\n```json\n{"pages": {"p1": []}, "relations": []}\n```\nDone.'
        assert kg._parse_extraction(raw)["pages"] == {"p1": []}
        assert kg._parse_extraction("no json at all") == {}
        assert kg._parse_extraction("") == {}
        # Non-object JSON answers are dropped by the object wrapper.
        assert kg._parse_extraction('["a", "b"]') == {}

    def test_extract_json_objects_and_arrays(self):
        from api.docgen._common import extract_json

        assert extract_json(
            'Sure! Here is the JSON:\n```json\n{"pages": {}}\n```\nDone.'
        ) == {"pages": {}}
        assert extract_json('noise ["a", "b"] tail') == ["a", "b"]
        assert extract_json("[] pure") == []
        assert extract_json("no json at all") is None
        assert extract_json(None) is None
        assert extract_json("") is None

    def test_extract_pages_terms_batches_and_filters(self):
        pages = {
            "p1": {"title": "A", "content": "alpha " * 10},
            "p2": {"title": "B", "content": "beta " * 10},
            "empty": {"title": "E", "content": "   "},
        }
        raw_p1 = (
            '{"pages": {"p1": [{"term": "alpha", "weight": 0.7}],'
            ' "ghost": [{"term": "nope"}]},'
            ' "relations": [{"from": "alpha", "to": "beta", "kind": "related"},'
            ' {"from": "x", "to": "y", "kind": "bogus"}]}'
        )
        raw_p2 = '{"pages": {"p2": [{"term": "beta", "weight": 0.2}]}, "relations": []}'

        class _LLM:
            def __init__(self):
                self.calls = 0

            async def generate(self, prompt: str) -> str:
                self.calls += 1
                return raw_p1 if "### p1 " in prompt else raw_p2

        llm = _LLM()
        page_terms, relations = asyncio.run(
            kg.extract_pages_terms(pages, llm, batch_size=1, concurrency=2)
        )
        assert llm.calls == 2  # empty page never enters a batch
        assert page_terms == {
            "p1": [{"term": "alpha", "weight": 0.7}],
            "p2": [{"term": "beta", "weight": 0.2}],
        }  # unknown page id "ghost" filtered out
        # Relations pass through unvalidated here; the merge filters kinds.
        assert relations == [
            {"from": "alpha", "to": "beta", "kind": "related"},
            {"from": "x", "to": "y", "kind": "bogus"},
        ]
        graph = kg.new_graph()
        assert kg.add_term_relations(graph, relations) == 1

    def test_extract_failed_batch_is_non_fatal(self):
        class _LLM:
            async def generate(self, prompt: str) -> str:
                raise RuntimeError("llm down")

        page_terms, relations = asyncio.run(
            kg.extract_pages_terms({"p1": {"content": "x"}}, _LLM(), batch_size=1, concurrency=1)
        )
        assert page_terms == {} and relations == []


# --- storage + incremental merge -------------------------------------------------
class TestStorageAndMerge:
    def test_mutate_and_load_roundtrip_with_stats(self, isolated_db):
        _seed_product(isolated_db, "prod_1")
        def _mutate(g: dict) -> bool:
            kg.set_page_terms(
                g, kg.upsert_page(g, "codebase", "cb1", "p1"),
                [{"term": "rms", "weight": 0.8}],
            )
            return True

        saved = asyncio.run(kg.mutate_product_graph("prod_1", _mutate))
        assert saved is True
        graph = asyncio.run(kg.load_product_graph("prod_1", fresh=True))
        assert "term:rms" in graph["nodes"]
        stats = kg.graph_stats()
        assert stats == [{
            "product_id": "prod_1", "nodes": 2, "edges": 1, "version": 1,
            "updated_at": stats[0]["updated_at"],
        }]
        assert asyncio.run(kg.load_product_graph("prod_nope", fresh=True)) is None

    def test_update_graph_from_pages_incremental(self, isolated_db):
        _seed_product(isolated_db, "prod_1")
        pages_v1 = {
            "p1": {"title": "Overview", "content": "RMS billing overview"},
            "p2": {"title": "API", "content": "API reference"},
        }
        raw = (
            '{"pages": {"p1": [{"term": "rms", "description": "revenue", "weight": 0.9}],'
            ' "p2": [{"term": "api", "weight": 0.5}]}, "relations": []}'
        )

        class _LLM:
            async def generate(self, prompt: str) -> str:
                return raw

        # First pass: only p1 changed.
        assert asyncio.run(
            kg.update_graph_from_pages(
                "prod_1", "codebase", "cb1", pages_v1, only={"p1"}, llm=_LLM()
            )
        )
        graph = asyncio.run(kg.load_product_graph("prod_1", fresh=True))
        assert "page:codebase:cb1:p1" in graph["nodes"]
        assert "page:codebase:cb1:p2" in graph["nodes"]  # node upserted for all pages
        assert kg.edge_key("term:rms", "mentions", "page:codebase:cb1:p1") in graph["edges"]
        assert "term:api" not in graph["nodes"]  # p2 not extracted yet

        # Second pass: full artifact — p2 finally gets extracted too.
        pages_v2 = {
            "p1": {"title": "Overview", "content": "generic text without domain words"},
            "p2": {"title": "API", "content": "API reference"},
        }
        assert asyncio.run(
            kg.update_graph_from_pages("prod_1", "codebase", "cb1", pages_v2, llm=_LLM())
        )
        graph = asyncio.run(kg.load_product_graph("prod_1", fresh=True))
        assert "term:api" in graph["nodes"]
        assert kg.edge_key("term:api", "mentions", "page:codebase:cb1:p2") in graph["edges"]

        # Page removed entirely -> node + mentions pruned.
        assert asyncio.run(
            kg.update_graph_from_pages(
                "prod_1", "codebase", "cb1", {"p2": pages_v2["p2"]}, llm=_LLM()
            )
        )
        graph = asyncio.run(kg.load_product_graph("prod_1", fresh=True))
        assert "page:codebase:cb1:p1" not in graph["nodes"]

    def test_update_graph_without_llm_is_noop(self, isolated_db, monkeypatch):
        _seed_product(isolated_db, "prod_1")
        monkeypatch.setattr(kg, "_build_llm", lambda: None)
        assert not asyncio.run(
            kg.update_graph_from_pages(
                "prod_1", "codebase", "cb1", {"p1": {"content": "x"}}
            )
        )
        assert asyncio.run(kg.load_product_graph("prod_1", fresh=True)) is None

    def test_rebuild_disabled_gate(self, isolated_db, monkeypatch):
        import api.config.timeout as timeout_mod

        monkeypatch.setattr(timeout_mod, "resolve_timeout_bool", lambda key: False)
        result = asyncio.run(kg.rebuild_product_graph("prod_1"))
        assert result["success"] is False
        assert "disabled" in result["message"].lower()

    def test_rebuild_collects_codebase_and_database_pages(self, isolated_db):
        from api.models import CodebaseORM, DatabaseORM, ProductORM

        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="P", summary=""))
            db.add(CodebaseORM(id="cb_1", product_id="prod_1", name="cb",
                               pages={"p1": {"title": "T", "content": "text"}}))
            db.add(DatabaseORM(id="db_1", product_id="prod_1", name="db",
                               pages={"q1": {"title": "D", "content": "data"}}))
            db.commit()

        raw = (
            '{"pages": {"p1": [{"term": "alpha", "weight": 0.7}],'
            ' "q1": [{"term": "beta", "weight": 0.6}]},'
            ' "relations": [{"from": "alpha", "to": "beta", "kind": "related"}]}'
        )

        class _LLM:
            async def generate(self, prompt: str) -> str:
                return raw

        result = asyncio.run(kg.rebuild_product_graph("prod_1", llm=_LLM()))
        assert result["success"] is True
        assert result["pages"] == 2
        assert result["nodes"] == 4  # 2 terms + 2 page nodes
        assert result["edges"] == 3  # 2 mentions + 1 term relation
        graph = asyncio.run(kg.load_product_graph("prod_1", fresh=True))
        assert "term:alpha" in graph["nodes"]

    def test_rebuild_save_rides_write_lock_with_fresh_version(self, isolated_db, monkeypatch):
        """Q-4: the final rebuild save must re-read the CURRENT row under the
        write lock, so a concurrent graph merge during extraction is neither
        lost nor clobbered (no stale prev_version)."""
        from api.models import CodebaseORM, KnowledgeGraphORM, ProductORM

        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="P", summary=""))
            db.add(CodebaseORM(id="cb_1", product_id="prod_1", name="cb",
                               pages={"p1": {"title": "T", "content": "text"}}))
            db.add(KnowledgeGraphORM(product_id="prod_1", graph=kg.new_graph(), version=3))
            db.commit()

        real_save = kg._save_graph_sync

        async def _extract_with_race(pages, llm, *, batch_size, concurrency):
            # A concurrent post-indexation commit lands DURING extraction.
            real_save("prod_1", kg.new_graph(), 3)  # row v3 -> v4
            return {"p1": [{"term": "alpha", "weight": 0.7}]}, []

        monkeypatch.setattr(kg, "extract_pages_terms", _extract_with_race)

        saved = {}

        def _spy_save(pid, graph, prev_version):
            assert kg._graph_lock is not None and kg._graph_lock.locked()
            saved["prev"] = prev_version
            return real_save(pid, graph, prev_version)

        monkeypatch.setattr(kg, "_save_graph_sync", _spy_save)

        class _LLM:
            async def generate(self, prompt: str) -> str:
                return '{}'

        result = asyncio.run(kg.rebuild_product_graph("prod_1", llm=_LLM()))
        assert result["success"] is True
        assert saved["prev"] == 4  # fresh version read under the lock
        graph = asyncio.run(kg.load_product_graph("prod_1", fresh=True))
        assert "term:alpha" in graph["nodes"]
        assert kg.graph_stats()[0]["version"] == 5


# --- graph-boost recall leg --------------------------------------------------------
class TestGraphBoostedRows:
    def _prepare(self, isolated_db, product_id: str):
        _seed_product(isolated_db, product_id)
        graph = _graph_with_rms("cb1", "revenue management system")
        asyncio.run(kg.mutate_product_graph(product_id, lambda g: (g.update(graph), True)[1]))

    def test_boost_promotes_related_page(self, isolated_db, monkeypatch):
        from api.memory.pgvector_backend import PgVectorMemoryBackend

        self._prepare(isolated_db, "prod_1")
        backend = PgVectorMemoryBackend()

        async def _rows(qvec, product_id, top_k):
            return [
                ("unrelated but similar", "other::page::x", 0.20),
                ("RMS pricing rules", "cb1::page::p1", 0.30),
                ("legacy blob of related artifact", "cb1", 0.40),
            ]

        monkeypatch.setattr(backend, "_cosine_rows_with_source", _rows)
        rows = asyncio.run(
            backend._graph_boosted_rows("How does RMS price?", [0.1, 0.2], "prod_1", 2)
        )
        # The boosted related pages (sim 0.7/0.6 + 0.9*0.25) beat the closer
        # but unrelated row (sim 0.8).
        assert rows[0] == "RMS pricing rules"
        assert rows[1] == "legacy blob of related artifact"

    def test_misses_fall_back_to_empty(self, isolated_db, monkeypatch):
        from api.memory.pgvector_backend import PgVectorMemoryBackend

        self._prepare(isolated_db, "prod_1")
        backend = PgVectorMemoryBackend()
        # No token of the query matches a graph term.
        assert asyncio.run(
            backend._graph_boosted_rows("zzz qqq", [0.1], "prod_1", 5)
        ) == []
        # No query vector -> embedder-down fallback.
        assert asyncio.run(backend._graph_boosted_rows("RMS", None, "prod_1", 5)) == []
        # No graph at all.
        assert asyncio.run(backend._graph_boosted_rows("RMS", [0.1], "prod_nope", 5)) == []


# --- RMS disambiguation acceptance (hermetic) ---------------------------------------
class TestRmsDisambiguation:
    def _seed_two_products(self, isolated_db):
        from api.models import KnowledgeChunkORM

        _seed_product(isolated_db, "prod_shop")
        _seed_product(isolated_db, "prod_ml")
        asyncio.run(
            kg.mutate_product_graph(
                "prod_shop",
                lambda g: (g.update(_graph_with_rms("cb_shop", "revenue management system")), True)[1],
            )
        )
        asyncio.run(
            kg.mutate_product_graph(
                "prod_ml",
                lambda g: (g.update(_graph_with_rms("cb_ml", "root mean square error")), True)[1],
            )
        )
        with isolated_db.SessionLocal() as db:
            db.add(KnowledgeChunkORM(
                id="chunk_shop", product_id="prod_shop", source_type="codebase",
                source_id="cb_shop::page::p1", chunk_index=0,
                content="Revenue Management System sets nightly prices",
            ))
            db.add(KnowledgeChunkORM(
                id="chunk_ml", product_id="prod_ml", source_type="codebase",
                source_id="cb_ml::page::p1", chunk_index=0,
                content="Root-mean-square error of the forecast model",
            ))
            db.commit()

    def test_graph_scope_is_per_product(self, isolated_db):
        self._seed_two_products(isolated_db)
        for pid, artifact in (("prod_shop", "cb_shop"), ("prod_ml", "cb_ml")):
            graph = asyncio.run(kg.load_product_graph(pid, fresh=True))
            keys = kg.resolve_query_terms("explain the RMS metric", graph)
            assert keys == ["term:rms"]
            boosts = kg.related_source_boosts(graph, keys)
            assert set(boosts) == {f"{artifact}::page::p1", artifact}
            # The OTHER product's sources never leak into this namespace.
            other = "cb_ml" if artifact == "cb_shop" else "cb_shop"
            assert other not in boosts

    def test_search_hits_scope_is_per_product(self, isolated_db):
        from api.agents.tools import format_hits_with_citations, search_knowledge_hits

        self._seed_two_products(isolated_db)
        shop_text = format_hits_with_citations(
            asyncio.run(search_knowledge_hits("RMS", "prod_shop", top_k=5))
        )
        ml_text = format_hits_with_citations(
            asyncio.run(search_knowledge_hits("RMS", "prod_ml", top_k=5))
        )
        assert "nightly prices" in shop_text and "forecast" not in shop_text
        assert "forecast" in ml_text and "nightly prices" not in ml_text
