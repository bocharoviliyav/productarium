"""Unit tests for the MCP introspection disk cache (item 2.3b).

Hermetic: the state dir is a per-test tmp path, MCP tools/LLM/indexing are
fakes patched onto the ``api.docgen.database`` seams.

Covers:
- ``introspection_cache_key`` determinism + sensitivity (server/allowlist/
  bindings/budgets).
- ``store_introspection_cache`` / ``load_introspection_cache`` roundtrip,
  atomicity (no tmp leftovers), miss cases (corrupt file, key/version
  mismatch, TTL expiry, disabled cache).
- ``generate_database_docs`` integration: the second run takes introspection
  from the cache (the MCP mock counts calls), TTL=0 disables, a changed
  binding set invalidates, and a failed introspection is never cached.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.docgen.database as db_doc_mod
from api.docgen.introspection_cache import (
    CACHE_FORMAT_VERSION,
    cache_ttl_seconds,
    introspection_cache_key,
    load_introspection_cache,
    store_introspection_cache,
)


# ============================================================================
# Helpers (minimal fakes, independent of test_docgen_database)
# ============================================================================
class FakeTool:
    """Minimal MCP tool stand-in: name + canned ainvoke, counts calls."""

    def __init__(self, name, responses="[]"):
        self.name = name
        self.responses = responses
        self.calls: list = []

    async def ainvoke(self, args):
        self.calls.append(args)
        if callable(self.responses):
            return self.responses(args)
        return self.responses


def _default_tools():
    return [
        FakeTool("list_schemas", json.dumps(["public"])),
        FakeTool(
            "list_tables",
            lambda args: json.dumps(
                [{"table_name": "users"}, {"table_name": "orders"}]
                if (args.get("schema") or "") in (None, "", "public")
                else []
            ),
        ),
        FakeTool(
            "describe_table",
            lambda args: f"CREATE TABLE {args.get('table_name')} (id integer PRIMARY KEY);",
        ),
    ]


def _fake_entity(**overrides):
    entity = SimpleNamespace(
        id="db_1",
        name="Main DB",
        dsn_masked="postgresql://app:***REDACTED***@db:5432/prod",
        mcp_server_id=None,
        generated_docs=None,
        pages=None,
    )
    for key, value in overrides.items():
        setattr(entity, key, value)
    return entity


def _fake_product(pid="prod_1"):
    return SimpleNamespace(id=pid)


def _tool_call_count(tools):
    return sum(len(t.calls) for t in tools)


def _patch_flow(monkeypatch, *, tools=None, binding_inputs=None, llm_text="DB DOCS"):
    """Patch all generate_database_docs seams; return (indexing, resolve log)."""
    tools = tools if tools is not None else _default_tools()
    resolve_log: list = []

    async def _resolve(entity, product_id):
        resolve_log.append(product_id)
        return list(tools)

    monkeypatch.setattr(db_doc_mod, "_resolve_mcp_tools", _resolve)
    if binding_inputs is not None:
        monkeypatch.setattr(
            db_doc_mod, "_cache_binding_inputs", lambda pid, pin: binding_inputs
        )

    async def _llm(prompt, model, base_url=None, api_key=None):
        return llm_text

    monkeypatch.setattr(db_doc_mod, "_llm_or_none", _llm)
    monkeypatch.setattr(db_doc_mod, "_make_repair_llm", lambda *a, **kw: None)

    async def _repair(content, llm):
        return content, {}

    monkeypatch.setattr(db_doc_mod, "run_repair_loop", _repair)
    monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "false")

    indexing: list = []
    monkeypatch.setattr(
        db_doc_mod,
        "_index_in_background",
        lambda content, dataset, **kw: indexing.append((content, dataset, kw)),
    )
    return indexing, resolve_log, tools


# ============================================================================
# introspection_cache_key
# ============================================================================
class TestCacheKey:
    def test_deterministic(self):
        k1 = introspection_cache_key(binding_ids=["b2", "b1"], budgets={"n": 1})
        k2 = introspection_cache_key(binding_ids=["b1", "b2"], budgets={"n": 1})
        assert k1 == k2
        assert len(k1) == 64  # sha256 hex

    def test_server_vs_bindings_differ(self):
        pinned = introspection_cache_key(mcp_server_id="srv1", allowlist=[])
        gather = introspection_cache_key(binding_ids=["b1"])
        assert pinned != gather

    def test_sensitive_to_each_input(self):
        base = introspection_cache_key(mcp_server_id="srv1", allowlist=["t1"])
        assert base != introspection_cache_key(mcp_server_id="srv2", allowlist=["t1"])
        assert base != introspection_cache_key(mcp_server_id="srv1", allowlist=["t2"])
        assert base != introspection_cache_key(mcp_server_id="srv1", allowlist=["t1"], budgets={"n": 1})
        assert introspection_cache_key(binding_ids=["b1"]) != introspection_cache_key(
            binding_ids=["b1", "b2"]
        )

    def test_ttl_env_default_and_parse(self, monkeypatch):
        monkeypatch.delenv("DB_INTROSPECTION_CACHE_TTL_SECONDS", raising=False)
        assert cache_ttl_seconds() == 7 * 24 * 3600.0
        monkeypatch.setenv("DB_INTROSPECTION_CACHE_TTL_SECONDS", "0")
        assert cache_ttl_seconds() == 0.0
        monkeypatch.setenv("DB_INTROSPECTION_CACHE_TTL_SECONDS", "garbage")
        assert cache_ttl_seconds() == 7 * 24 * 3600.0  # invalid → default


# ============================================================================
# store / load roundtrip + miss cases
# ============================================================================
class TestStoreLoad:
    @pytest.fixture(autouse=True)
    def _state_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRODUCTARIUM_STATE_DIR", str(tmp_path / "state"))

    def test_roundtrip_preserves_payload(self):
        info = {
            "schemas": ["публичная"],
            "tables": {"public.users": {"schema": "public", "table": "users",
                                        "definition": "CREATE TABLE …"}},
            "tools_used": {"tables": "list_tables"},
        }
        key = introspection_cache_key(binding_ids=["b1"])
        assert store_introspection_cache(key, info) is True
        assert load_introspection_cache(key) == info

    def test_no_tmp_leftovers_and_owner_only(self):
        key = introspection_cache_key(binding_ids=["b1"])
        store_introspection_cache(key, {"tables": {}})
        root = os.path.join(
            os.environ["PRODUCTARIUM_STATE_DIR"], "introspection_cache"
        )
        files = sorted(os.listdir(root))
        assert files == [f"{key}.json"]  # только финальный файл, никаких .tmp
        if os.name == "posix":
            mode = os.stat(os.path.join(root, files[0])).st_mode & 0o777
            assert mode == 0o600

    def test_corrupt_file_is_miss(self):
        key = introspection_cache_key(binding_ids=["b1"])
        root = os.path.join(
            os.environ["PRODUCTARIUM_STATE_DIR"], "introspection_cache"
        )
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, f"{key}.json"), "w") as f:
            f.write("{not json")
        assert load_introspection_cache(key) is None

    def test_key_mismatch_is_miss(self):
        key = introspection_cache_key(binding_ids=["b1"])
        store_introspection_cache(key, {"tables": {}})
        other = introspection_cache_key(binding_ids=["b2"])
        assert load_introspection_cache(other) is None

    def test_version_mismatch_is_miss(self):
        key = introspection_cache_key(binding_ids=["b1"])
        store_introspection_cache(key, {"tables": {}})
        root = os.path.join(
            os.environ["PRODUCTARIUM_STATE_DIR"], "introspection_cache"
        )
        path = os.path.join(root, f"{key}.json")
        with open(path, "r") as f:
            payload = json.load(f)
        payload["version"] = CACHE_FORMAT_VERSION + 100
        with open(path, "w") as f:
            json.dump(payload, f)
        assert load_introspection_cache(key) is None

    def test_expired_entry_is_miss(self, monkeypatch):
        key = introspection_cache_key(binding_ids=["b1"])
        store_introspection_cache(key, {"tables": {}})
        monkeypatch.setenv("DB_INTROSPECTION_CACHE_TTL_SECONDS", "10")
        path = os.path.join(
            os.environ["PRODUCTARIUM_STATE_DIR"], "introspection_cache", f"{key}.json"
        )
        old = time.time() - 60
        os.utime(path, (old, old))
        assert load_introspection_cache(key) is None

    def test_ttl_zero_disables_both_directions(self, monkeypatch):
        monkeypatch.setenv("DB_INTROSPECTION_CACHE_TTL_SECONDS", "0")
        key = introspection_cache_key(binding_ids=["b1"])
        assert store_introspection_cache(key, {"tables": {}}) is False
        assert load_introspection_cache(key) is None
        root = os.path.join(
            os.environ["PRODUCTARIUM_STATE_DIR"], "introspection_cache"
        )
        assert not os.path.exists(root)  # ничего не создано


# ============================================================================
# generate_database_docs: cache-first introspection
# ============================================================================
class TestGenerateFlowCache:
    @pytest.fixture(autouse=True)
    def _state_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRODUCTARIUM_STATE_DIR", str(tmp_path / "state"))

    def test_second_run_takes_introspection_from_cache(self, monkeypatch):
        indexing, resolve_log, tools = _patch_flow(
            monkeypatch, binding_inputs={"binding_ids": ["b1"]}
        )

        entity1 = _fake_entity()
        asyncio.run(db_doc_mod.generate_database_docs(entity1, _fake_product()))
        assert len(resolve_log) == 1
        calls_after_first = _tool_call_count(tools)
        assert calls_after_first > 0  # MCP-интерфейс реально звался
        assert indexing[0][2]["source_type"] == "database"
        prov_miss = entity1.pages["page_overview"]["provenance"]
        assert prov_miss["introspection_cache"] == "miss"
        assert prov_miss["tools_used"] == {
            "schemas": "list_schemas",
            "tables": "list_tables",
            "describe": "describe_table",
        }

        entity2 = _fake_entity()
        asyncio.run(db_doc_mod.generate_database_docs(entity2, _fake_product()))

        # Кэш-хит: ни resolve, ни один MCP tool call не повторился.
        assert len(resolve_log) == 1
        assert _tool_call_count(tools) == calls_after_first
        # Документ собран из закэшированной схемы без обращения к MCP.
        assert entity2.generated_docs == "DB DOCS"
        assert entity2.pages["page_overview"]["provenance"]["introspection_cache"] == "hit"
        assert entity2.pages["page_overview"]["provenance"]["tools_used"] == {
            "schemas": "list_schemas",
            "tables": "list_tables",
            "describe": "describe_table",
        }

    def test_ttl_zero_introspects_every_run(self, monkeypatch):
        monkeypatch.setenv("DB_INTROSPECTION_CACHE_TTL_SECONDS", "0")
        _, resolve_log, tools = _patch_flow(
            monkeypatch, binding_inputs={"binding_ids": ["b1"]}
        )

        asyncio.run(
            db_doc_mod.generate_database_docs(_fake_entity(), _fake_product())
        )
        first = _tool_call_count(tools)
        asyncio.run(
            db_doc_mod.generate_database_docs(_fake_entity(), _fake_product())
        )

        assert len(resolve_log) == 2
        assert _tool_call_count(tools) == 2 * first

    def test_changed_bindings_invalidate_cache(self, monkeypatch):
        _, resolve_log, tools = _patch_flow(
            monkeypatch, binding_inputs={"binding_ids": ["b1"]}
        )
        asyncio.run(
            db_doc_mod.generate_database_docs(_fake_entity(), _fake_product())
        )

        # Другой набор привязок = другой ключ = промах.
        monkeypatch.setattr(
            db_doc_mod,
            "_cache_binding_inputs",
            lambda pid, pin: {"binding_ids": ["b1", "b2"]},
        )
        asyncio.run(
            db_doc_mod.generate_database_docs(_fake_entity(), _fake_product())
        )
        assert len(resolve_log) == 2

    def test_pinned_surface_keyed_by_server_and_allowlist(self, monkeypatch):
        _, resolve_log, tools = _patch_flow(
            monkeypatch,
            binding_inputs={"mcp_server_id": "srv1", "allowlist": []},
        )
        asyncio.run(
            db_doc_mod.generate_database_docs(
                _fake_entity(mcp_server_id="srv1"), _fake_product()
            )
        )
        second_calls = _tool_call_count(tools)
        asyncio.run(
            db_doc_mod.generate_database_docs(
                _fake_entity(mcp_server_id="srv1"), _fake_product()
            )
        )
        # Тот же pin + тот же allowlist → повтор из кэша.
        assert len(resolve_log) == 1
        assert _tool_call_count(tools) == second_calls

    def test_unbindable_pin_bypasses_cache(self, monkeypatch):
        # Pin не привязан/выключен → ключ None → живой путь и честная ошибка.
        monkeypatch.setattr(
            db_doc_mod, "_cache_binding_inputs", lambda pid, pin: None
        )

        async def _resolve(entity, product_id):
            return []

        monkeypatch.setattr(db_doc_mod, "_resolve_mcp_tools", _resolve)
        with pytest.raises(ValueError, match="No MCP tools"):
            asyncio.run(
                db_doc_mod.generate_database_docs(
                    _fake_entity(mcp_server_id="ghost"), _fake_product()
                )
            )

    def test_failed_introspection_not_cached(self, monkeypatch):
        tools = [
            FakeTool("list_schemas", json.dumps(["public"])),
            FakeTool("list_tables", "[]"),  # пустой листинг → ValueError
        ]
        _, resolve_log, _ = _patch_flow(
            monkeypatch, tools=tools, binding_inputs={"binding_ids": ["b1"]}
        )

        with pytest.raises(ValueError, match="produced no tables"):
            asyncio.run(
                db_doc_mod.generate_database_docs(_fake_entity(), _fake_product())
            )
        root = os.path.join(
            os.environ["PRODUCTARIUM_STATE_DIR"], "introspection_cache"
        )
        # Ничего не записано — повторный прогон снова пойдёт в MCP.
        assert not os.path.exists(root)

        with pytest.raises(ValueError, match="produced no tables"):
            asyncio.run(
                db_doc_mod.generate_database_docs(_fake_entity(), _fake_product())
            )
        assert len(resolve_log) == 2
