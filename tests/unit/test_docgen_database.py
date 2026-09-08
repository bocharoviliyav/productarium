"""Unit tests for ``api.docgen.database`` (Wave E: MCP reverse-engineering).

Hermetic: every MCP tool, LLM call, repair loop and indexing call is a fake
patched onto the module seams; no server, DB-with-data, or network access.

Covers:
- ``mask_dsn`` (the DSN secret choke point).
- ``_classify_introspection_tools`` (name-token heuristics).
- ``_parse_names`` (JSON / dict shapes / quoted fallback / ERROR results).
- ``_build_tool_args`` + ``_tool_arg_names`` ((schema, table) mapping).
- ``_call_tool`` (dict→JSON, exceptions, timeout, result cap).
- ``_render_skeleton`` (overview / schemas / tables).
- ``_introspect`` (schemas → tables → definitions walk).
- ``_tools_for_pinned_server`` (binding/allowlist/unreachable rules).
- ``generate_database_docs`` (happy path, skeleton fallback, masking,
  provenance + indexing, honest ValueErrors).
- ``api.docgen.jobs`` database dispatch (success + failure).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.docgen.database as db_doc_mod
import api.docgen.jobs as jobs_mod


# ============================================================================
# Helpers
# ============================================================================
async def _async_return(value):
    return value


async def _async_return_pair(value):
    return value


class FakeTool:
    """A minimal LangChain-tool stand-in: name, args schema, canned ainvoke."""

    def __init__(self, name, args=None, responses=None, delay=0.0, error=None):
        self.name = name
        self.args = args or {}
        self.responses = responses if responses is not None else "[]"
        self.delay = delay
        self.error = error
        self.calls: list = []

    async def ainvoke(self, args):
        import asyncio as _asyncio

        if self.delay:
            await _asyncio.sleep(self.delay)
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        if callable(self.responses):
            return self.responses(args)
        return self.responses


def _default_tools():
    """The standard fake introspection surface (one Postgres-ish schema)."""
    return [
        FakeTool(
            "list_schemas",
            {},
            json.dumps(["public"]),
        ),
        FakeTool(
            "list_tables",
            {"schema": {"type": "string"}},
            lambda args: json.dumps(
                [{"table_name": "users"}, {"table_name": "orders"}]
                if (args.get("schema") or "") in (None, "", "public")
                else []
            ),
        ),
        FakeTool(
            "describe_table",
            {"schema": {"type": "string"}, "table_name": {"type": "string"}},
            lambda args: f"CREATE TABLE {args.get('table_name')} (id integer PRIMARY KEY);",
        ),
        # Unclassified tool (query execution) — must be ignored by the walk.
        FakeTool("execute_query", {"sql": {"type": "string"}}, "ok"),
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


def _patch_generation(monkeypatch, *, llm_text, tools=None):
    """Patch all generation seams; return the indexing-call recorder."""
    if tools is None:
        tools = _default_tools()

    async def _resolve(entity, product_id):
        return tools

    monkeypatch.setattr(db_doc_mod, "_resolve_mcp_tools", _resolve)
    monkeypatch.setattr(
        db_doc_mod, "_llm_or_none", lambda *a, **kw: _async_return(llm_text)
    )
    monkeypatch.setattr(db_doc_mod, "_make_repair_llm", lambda *a, **kw: None)
    monkeypatch.setattr(
        db_doc_mod, "run_repair_loop", lambda content, llm: _async_return_pair((content, {}))
    )
    monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "false")

    indexing: list = []

    def track_indexing(content, dataset, *, source_type="codebase", source_id=None):
        indexing.append((content, dataset, source_type, source_id))

    monkeypatch.setattr(db_doc_mod, "_index_in_background", track_indexing)
    return indexing


# ============================================================================
# mask_dsn (the DSN secret choke point)
# ============================================================================
class TestMaskDsn:
    def test_password_masked_context_kept(self):
        from api.docgen.verification import mask_dsn

        masked = mask_dsn("postgresql://app:hunter2@db:5432/prod")
        assert masked == "postgresql://app:***REDACTED***@db:5432/prod"
        assert "hunter2" not in masked

    def test_user_without_password_kept(self):
        from api.docgen.verification import mask_dsn

        assert mask_dsn("oracle://app@db:1521/orcl") == "oracle://app@db:1521/orcl"

    def test_secret_query_param_masked(self):
        from api.docgen.verification import mask_dsn

        masked = mask_dsn("postgres://u:p@h/db?password=topsecret&sslmode=require")
        assert "topsecret" not in masked
        assert "p@" not in masked  # URL credentials also gone
        assert "password=***REDACTED***" in masked
        # NOTE: the generic assignment matcher treats the ``&``-joined tail
        # as one value, so params AFTER a secret param are over-masked too —
        # safe-by-design (secret-free beats pretty).

    def test_non_url_kv_string_goes_through_mask_secrets(self):
        from api.docgen.verification import mask_dsn

        masked = mask_dsn("host=db password=hunter2 dbname=prod")
        assert "hunter2" not in masked
        assert "***REDACTED***" in masked

    def test_idempotent(self):
        from api.docgen.verification import mask_dsn

        once = mask_dsn("postgresql://app:hunter2@db:5432/prod")
        assert mask_dsn(once) == once

    def test_empty(self):
        from api.docgen.verification import mask_dsn

        assert mask_dsn("") == ""
        assert mask_dsn(None) == ""

    def test_password_with_slash_masked(self):
        # Review #4 HIGH: '/' in the password used to break the URL match
        # entirely → NOTHING was masked and the raw DSN persisted.
        from api.docgen.verification import mask_dsn

        masked = mask_dsn("postgresql://app:pa/ss/w0rd@db:5432/prod")
        assert masked == "postgresql://app:***REDACTED***@db:5432/prod"
        assert "pa/ss" not in masked

    def test_password_with_at_masked(self):
        # Review #4 HIGH: '@' in the password used to cut the userinfo at the
        # FIRST '@', leaking the password tail into the "host" part.
        from api.docgen.verification import mask_dsn

        masked = mask_dsn("postgresql://app:p@ssw0rd@db:5432/prod")
        assert masked == "postgresql://app:***REDACTED***@db:5432/prod"
        assert "p@ssw0rd" not in masked

    def test_passwordless_url_dsn_unchanged(self):
        # Review #4 HIGH-functional: DSNs with no secret legitimately mask
        # to themselves — the leak guard must not treat that as an error.
        from api.docgen.verification import mask_dsn

        assert mask_dsn("postgres://localhost:5432/db") == "postgres://localhost:5432/db"
        assert mask_dsn("postgres://app@host/db") == "postgres://app@host/db"

    def test_ambiguous_multi_at_userinfo_masks_credentials(self):
        # Multiple '@' with no ':' separator: ambiguous userinfo — the whole
        # credentials block is masked instead of guessing a split.
        from api.docgen.verification import mask_dsn

        assert mask_dsn("postgres://a@b@host/db") == "postgres://***REDACTED***@host/db"


# ============================================================================
# Introspection budgets (review #4: overall deadline + tool-call cap)
# ============================================================================
class TestIntrospectionBudgets:
    def test_tool_call_cap_zero_fails_honestly(self, monkeypatch):
        monkeypatch.setattr(db_doc_mod, "MAX_TOOL_CALLS", 0)
        tools = [
            FakeTool("list_schemas", {}, '["public"]'),
            FakeTool(
                "list_tables", {},
                lambda a: json.dumps([{"table_name": "users"}]),
            ),
        ]
        roles = db_doc_mod._classify_introspection_tools(tools)
        with pytest.raises(ValueError, match="produced no tables"):
            asyncio.run(db_doc_mod._introspect(roles))

    def test_tool_call_cap_exhausted_mid_walk(self, monkeypatch):
        # Budget for the listing + ONE describe: the walk still completes;
        # the starved definition is the budget ERROR, not a real call.
        monkeypatch.setattr(db_doc_mod, "MAX_TOOL_CALLS", 2)
        tools = [
            FakeTool(
                "list_tables", {},
                lambda a: json.dumps(
                    [{"table_name": "users"}, {"table_name": "orders"}]
                ),
            ),
            FakeTool("describe_table", {}, "CREATE TABLE users (id int);"),
        ]
        roles = db_doc_mod._classify_introspection_tools(tools)
        info = asyncio.run(db_doc_mod._introspect(roles))
        assert set(info["tables"]) == {"users", "orders"}
        assert info["tables"]["users"]["definition"].startswith("CREATE TABLE")
        assert "budget" in info["tables"]["orders"]["definition"]

    def test_introspection_deadline_honest_failure(self, monkeypatch):
        # A slow MCP surface cannot occupy a docgen worker indefinitely:
        # the overall deadline turns into an honest FAILED job (ValueError).
        monkeypatch.setattr(db_doc_mod, "INTROSPECTION_TIMEOUT_SECONDS", 0.05)
        tools = [
            FakeTool(
                "list_tables", {"schema": {}},
                lambda a: json.dumps([{"table_name": "users"}]),
                delay=1.0,
            ),
        ]
        _patch_generation(monkeypatch, llm_text="x", tools=tools)
        with pytest.raises(ValueError, match="time budget"):
            asyncio.run(
                db_doc_mod.generate_database_docs(_fake_entity(), _fake_product())
            )


# ============================================================================
# build_database_doc_prompt — per-request language (review #4 LOW)
# ============================================================================
class TestDatabaseDocPromptLanguage:
    def test_language_substituted_en(self):
        prompt = db_doc_mod.build_database_doc_prompt(
            database_name="Main DB",
            dsn_masked="postgresql://app:***REDACTED***@db/x",
            skeleton="skel",
            schema_dump="dump",
            language="en",
        )
        assert "English" in prompt
        assert "{language_name}" not in prompt

    def test_language_defaults_to_ru(self):
        prompt = db_doc_mod.build_database_doc_prompt(
            database_name="Main DB", dsn_masked="", skeleton="", schema_dump="",
        )
        assert "Russian" in prompt
        assert "{language_name}" not in prompt

    def test_generate_passes_language_to_prompt(self, monkeypatch):
        captured = []

        async def fake_llm(prompt, model, base_url=None, api_key=None):
            captured.append(prompt)
            return "docs"

        _patch_generation(monkeypatch, llm_text="docs")
        monkeypatch.setattr(db_doc_mod, "_llm_or_none", fake_llm)
        asyncio.run(
            db_doc_mod.generate_database_docs(
                _fake_entity(), _fake_product(), language="en"
            )
        )
        assert captured
        assert "English" in captured[0]


# ============================================================================
# _classify_introspection_tools
# ============================================================================
class TestClassifyTools:
    def _classify(self, *names):
        tools = [FakeTool(n) for n in names]
        return db_doc_mod._classify_introspection_tools(tools)

    def test_common_spellings(self):
        roles = self._classify(
            "list_schemas", "list_tables", "describe_table", "get_table_ddl"
        )
        assert [t.name for t in roles["schemas"]] == ["list_schemas"]
        assert [t.name for t in roles["tables"]] == ["list_tables"]
        assert [t.name for t in roles["describe"]] == ["describe_table"]
        assert [t.name for t in roles["ddl"]] == ["get_table_ddl"]

    def test_alternative_spellings(self):
        roles = self._classify(
            "show_schemas", "search_tables", "get_table_info", "get_table_definition"
        )
        assert [t.name for t in roles["schemas"]] == ["show_schemas"]
        assert [t.name for t in roles["tables"]] == ["search_tables"]
        assert [t.name for t in roles["describe"]] == ["get_table_info"]
        assert [t.name for t in roles["ddl"]] == ["get_table_definition"]

    def test_ddl_checked_before_describe_and_tables(self):
        # "create_table_ddl" contains table+ddl → ddl, not tables/describe.
        roles = self._classify("create_table_ddl")
        assert roles["ddl"] and not roles["tables"] and not roles["describe"]

    def test_schema_keyword_without_list_token_ignored(self):
        # "migrate_schema" has "schema" but no list/show/get/all → not a
        # schema-listing tool (it is not a tables tool either).
        roles = self._classify("migrate_schema")
        assert roles["schemas"] == []

    def test_unrelated_tools_ignored(self):
        roles = self._classify("execute_query", "health_check", "insert_row")
        assert all(not v for v in roles.values())

    def test_empty_input(self):
        assert db_doc_mod._classify_introspection_tools([]) == {
            "schemas": [], "tables": [], "describe": [], "ddl": [],
        }

    def test_all_role_matches_collected(self):
        # Every tool matching a role is collected; the walk picks the first.
        roles = self._classify("list_tables", "other_list_tables")
        assert [t.name for t in roles["tables"]] == ["list_tables", "other_list_tables"]


# ============================================================================
# _parse_names
# ============================================================================
class TestParseNames:
    def test_json_list_of_strings(self):
        assert db_doc_mod._parse_names('["public","analytics"]') == [
            "public", "analytics",
        ]

    def test_json_list_of_dicts_with_name_keys(self):
        text = json.dumps([{"table_name": "users"}, {"name": "orders"}])
        assert db_doc_mod._parse_names(text, "tables") == ["users", "orders"]

    def test_dict_with_collection_key(self):
        text = json.dumps({"tables": ["a", "b"], "count": 2})
        assert db_doc_mod._parse_names(text, "tables") == ["a", "b"]

    def test_dict_keyed_by_schema_single_list(self):
        text = json.dumps({"public": ["t1", "t2"]})
        assert db_doc_mod._parse_names(text) == ["t1", "t2"]

    def test_quoted_json_string_result(self):
        # A JSON-encoded string result yields one name; bare prose without
        # quotes yields [] (best-effort by design — see quoted fallback).
        assert db_doc_mod._parse_names('"single_table"') == ["single_table"]
        assert db_doc_mod._parse_names("bare prose") == []

    def test_quoted_fallback(self):
        assert db_doc_mod._parse_names('the tables are "users" and "orders"') == [
            "users", "orders",
        ]

    def test_error_result_returns_empty(self):
        assert db_doc_mod._parse_names("ERROR: MCP tool 'x' failed (Boom).") == []

    def test_empty_returns_empty(self):
        assert db_doc_mod._parse_names("") == []
        assert db_doc_mod._parse_names("[]") == []

    def test_dedup_preserves_order(self):
        text = json.dumps(["b", "a", "b", "a"])
        assert db_doc_mod._parse_names(text) == ["b", "a"]


# ============================================================================
# _tool_arg_names / _build_tool_args
# ============================================================================
class TestBuildToolArgs:
    def test_schema_and_table_mapped(self):
        tool = FakeTool("describe_table", {"schema": {}, "table_name": {}})
        args = db_doc_mod._build_tool_args(tool, schema="public", table="users")
        assert args == {"schema": "public", "table_name": "users"}

    def test_database_spelled_param_gets_schema(self):
        tool = FakeTool("list_tables", {"database_name": {}})
        args = db_doc_mod._build_tool_args(tool, schema="public")
        assert args == {"database_name": "public"}

    def test_generic_name_params_get_table(self):
        for arg in ("table", "name", "object", "object_name", "entity"):
            tool = FakeTool("t", {arg: {}})
            assert db_doc_mod._build_tool_args(tool, table="users") == {arg: "users"}

    def test_kwargs_filtered(self):
        tool = FakeTool("t", {"kwargs": {}, "table": {}})
        args = db_doc_mod._build_tool_args(tool, table="users")
        assert args == {"table": "users"}

    def test_no_args_schema(self):
        tool = FakeTool("list_schemas", {})
        assert db_doc_mod._build_tool_args(tool) == {}
        # A tool without an args schema falls back to no declared args.
        assert db_doc_mod._tool_arg_names(SimpleNamespace()) == []

    def test_schema_takes_precedence_over_table_token(self):
        tool = FakeTool("t", {"table_schema": {}})
        args = db_doc_mod._build_tool_args(tool, schema="public", table="users")
        assert args == {"table_schema": "public"}


# ============================================================================
# _call_tool
# ============================================================================
class TestCallTool:
    def test_dict_result_json_serialized(self):
        tool = FakeTool("t", responses={"rows": [1, 2]})
        out = asyncio.run(db_doc_mod._call_tool(tool, {}))
        assert json.loads(out) == {"rows": [1, 2]}

    def test_string_result_passthrough(self):
        tool = FakeTool("t", responses="plain text")
        assert asyncio.run(db_doc_mod._call_tool(tool, {})) == "plain text"

    def test_exception_returns_error_string(self):
        tool = FakeTool("boom", error=RuntimeError("server down"))
        out = asyncio.run(db_doc_mod._call_tool(tool, {}))
        assert out.startswith("ERROR:")
        assert "RuntimeError" in out

    def test_timeout_returns_error_string(self):
        import api.mcp.manager as mcp_manager

        tool = FakeTool("slow", responses="x", delay=0.5)
        monkeypatch_target = pytest.MonkeyPatch()
        monkeypatch_target.setattr(mcp_manager, "tool_call_timeout", lambda: 0.05)
        try:
            out = asyncio.run(db_doc_mod._call_tool(tool, {}))
        finally:
            monkeypatch_target.undo()
        assert out.startswith("ERROR:")
        assert "timed out" in out

    def test_result_capped(self):
        import api.mcp.manager as mcp_manager

        tool = FakeTool("t", responses="x" * 5000)
        monkeypatch_target = pytest.MonkeyPatch()
        monkeypatch_target.setattr(mcp_manager, "tool_result_max_chars", lambda: 1000)
        try:
            out = asyncio.run(db_doc_mod._call_tool(tool, {}))
        finally:
            monkeypatch_target.undo()
        assert len(out) < 5000
        assert "truncated" in out


# ============================================================================
# _render_skeleton
# ============================================================================
class TestRenderSkeleton:
    def test_renders_all_sections(self):
        entity = _fake_entity()
        info = {
            "schemas": ["public"],
            "tables": {
                "public.users": {
                    "schema": "public",
                    "table": "users",
                    "definition": "CREATE TABLE users (id integer);",
                },
                "public.orders": {
                    "schema": "public",
                    "table": "orders",
                    "definition": "",
                },
            },
        }
        overview, schema_md, tables_md = db_doc_mod._render_skeleton(entity, info)
        assert "# Database: Main DB" in overview
        assert "postgresql://app:***REDACTED***@db:5432/prod" in overview
        assert "**Tables introspected:** 2" in overview
        assert "`public` — 2 table(s)" in schema_md
        assert "### `public.users`" in tables_md
        assert "```sql" in tables_md
        assert "### `public.orders`" in tables_md
        assert "(no definition available" in tables_md

    def test_no_schemas_renders_default(self):
        entity = _fake_entity(dsn_masked=None)
        info = {
            "schemas": [],
            "tables": {
                "users": {"schema": None, "table": "users", "definition": ""},
            },
        }
        overview, schema_md, _ = db_doc_mod._render_skeleton(entity, info)
        assert "(default)" in overview
        assert "Single default schema" in schema_md

    def test_error_definition_rendered_as_note(self):
        entity = _fake_entity()
        info = {
            "schemas": [],
            "tables": {
                "users": {
                    "schema": None,
                    "table": "users",
                    "definition": "ERROR: MCP tool 'describe_table' failed (Boom).",
                },
            },
        }
        _, _, tables_md = db_doc_mod._render_skeleton(entity, info)
        assert "_ERROR:" in tables_md
        assert "```sql" not in tables_md


# ============================================================================
# _introspect
# ============================================================================
class TestIntrospect:
    def test_walk_schemas_tables_definitions(self):
        roles = db_doc_mod._classify_introspection_tools(_default_tools())
        info = asyncio.run(db_doc_mod._introspect(roles))
        assert info["schemas"] == ["public"]
        assert sorted(info["tables"]) == ["public.orders", "public.users"]
        assert "CREATE TABLE users" in info["tables"]["public.users"]["definition"]
        assert info["tools_used"] == {
            "schemas": "list_schemas",
            "tables": "list_tables",
            "describe": "describe_table",
        }

    def test_no_tables_tool_raises(self):
        tools = [FakeTool("list_schemas", {}, '["public"]')]
        roles = db_doc_mod._classify_introspection_tools(tools)
        with pytest.raises(ValueError, match="No table-listing MCP tool"):
            asyncio.run(db_doc_mod._introspect(roles))

    def test_empty_listing_raises(self):
        tools = [
            FakeTool("list_schemas", {}, '["public"]'),
            FakeTool("list_tables", {}, "[]"),
        ]
        roles = db_doc_mod._classify_introspection_tools(tools)
        with pytest.raises(ValueError, match="produced no tables"):
            asyncio.run(db_doc_mod._introspect(roles))

    def test_no_schema_tool_walks_default_scope(self):
        tools = [
            FakeTool("list_tables", {}, json.dumps(["users"])),
            FakeTool("describe_table", {"table": {}}, lambda a: "CREATE TABLE users ();"),
        ]
        roles = db_doc_mod._classify_introspection_tools(tools)
        info = asyncio.run(db_doc_mod._introspect(roles))
        assert info["schemas"] == []
        assert list(info["tables"]) == ["users"]

    def test_ddl_used_when_describe_fails(self):
        tools = [
            FakeTool("list_tables", {}, json.dumps(["users"])),
            FakeTool("describe_table", {"table": {}}, error=RuntimeError("boom")),
            FakeTool("get_table_ddl", {"table": {}}, "CREATE TABLE users (id int);"),
        ]
        roles = db_doc_mod._classify_introspection_tools(tools)
        info = asyncio.run(db_doc_mod._introspect(roles))
        # describe failed → ddl fallback provided the definition (as an
        # ERROR-string from _call_tool, then the ddl tool's output wins).
        assert "CREATE TABLE users (id int);" in info["tables"]["users"]["definition"]


# ============================================================================
# _tools_for_pinned_server (binding rules)
# ============================================================================
class TestToolsForPinnedServer:
    @pytest.fixture()
    def seeded(self, isolated_db):
        from api.models import (
            McpServerORM,
            ProductMcpServerORM,
            ProductORM,
        )

        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_pin", name="P"))
            s.add(McpServerORM(
                id="mcp_live", name="live-db", transport="http",
                url="http://localhost:9000/sse", enabled=True,
            ))
            s.add(McpServerORM(
                id="mcp_off", name="off-db", transport="http",
                url="http://localhost:9001/sse", enabled=False,
            ))
            s.add(ProductMcpServerORM(
                id="pmb_live", product_id="prod_pin", mcp_server_id="mcp_live",
                enabled=True, allowed_tools=["list_tables"],
            ))
            s.add(ProductMcpServerORM(
                id="pmb_dis", product_id="prod_pin", mcp_server_id="mcp_off",
                enabled=True,
            ))
            s.commit()
        return isolated_db

    def _patch_manager(self, monkeypatch, tools=None, error=None):
        import api.mcp.manager as mcp_manager

        class FakeManager:
            async def discover_tools(self, server):
                if error is not None:
                    raise error
                return list(tools or [])

        monkeypatch.setattr(mcp_manager, "get_mcp_manager", lambda: FakeManager())

    def test_unknown_server_raises(self, seeded, monkeypatch):
        self._patch_manager(monkeypatch, tools=[])
        with pytest.raises(ValueError, match="no longer exists"):
            asyncio.run(db_doc_mod._tools_for_pinned_server("prod_pin", "mcp_ghost"))

    def test_unbound_server_raises(self, seeded, monkeypatch):
        # mcp_off exists but has no ENABLED binding usable here (it is also
        # disabled globally) → rejected.
        self._patch_manager(monkeypatch, tools=[])
        with pytest.raises(ValueError, match="not bound\\+enabled"):
            asyncio.run(db_doc_mod._tools_for_pinned_server("prod_pin", "mcp_off"))

    def test_server_disabled_raises(self, seeded, monkeypatch):
        from api.models import ProductMcpServerORM

        with seeded.SessionLocal() as s:
            s.add(ProductMcpServerORM(
                id="pmb_off2", product_id="prod_pin", mcp_server_id="mcp_off",
                enabled=True,
            ))
            s.commit()
        self._patch_manager(monkeypatch, tools=[])
        with pytest.raises(ValueError, match="not bound\\+enabled"):
            asyncio.run(db_doc_mod._tools_for_pinned_server("prod_pin", "mcp_off"))

    def test_binding_disabled_raises(self, seeded, monkeypatch):
        from api.models import ProductMcpServerORM

        # Disable the enabled binding so NO enabled binding remains (the
        # first binding found for (product, server) decides).
        with seeded.SessionLocal() as s:
            row = s.query(ProductMcpServerORM).filter_by(id="pmb_live").first()
            row.enabled = False
            s.add(ProductMcpServerORM(
                id="pmb_live_dis", product_id="prod_pin", mcp_server_id="mcp_live",
                enabled=False,
            ))
            s.commit()
        self._patch_manager(monkeypatch, tools=[])
        with pytest.raises(ValueError, match="not bound\\+enabled"):
            asyncio.run(db_doc_mod._tools_for_pinned_server("prod_pin", "mcp_live"))

    def test_allowlist_filters_tools(self, seeded, monkeypatch):
        tools = [FakeTool("list_tables"), FakeTool("describe_table")]
        self._patch_manager(monkeypatch, tools=tools)
        out = asyncio.run(db_doc_mod._tools_for_pinned_server("prod_pin", "mcp_live"))
        assert [t.name for t in out] == ["list_tables"]

    def test_unreachable_manager_raises(self, seeded, monkeypatch):
        self._patch_manager(monkeypatch, error=ConnectionError("refused"))
        with pytest.raises(ValueError, match="unreachable"):
            asyncio.run(db_doc_mod._tools_for_pinned_server("prod_pin", "mcp_live"))

    def test_resolve_uses_pinned_server(self, seeded, monkeypatch):
        tools = [FakeTool("list_tables")]
        self._patch_manager(monkeypatch, tools=tools)
        entity = _fake_entity(mcp_server_id="mcp_live")
        out = asyncio.run(db_doc_mod._resolve_mcp_tools(entity, "prod_pin"))
        assert [t.name for t in out] == ["list_tables"]


# ============================================================================
# generate_database_docs
# ============================================================================
class TestGenerateDatabaseDocs:
    def test_happy_path_llm_enrichment(self, monkeypatch):
        indexing = _patch_generation(monkeypatch, llm_text="ENRICHED DATABASE DOCS")
        entity = _fake_entity()
        product = _fake_product()

        result = asyncio.run(
            db_doc_mod.generate_database_docs(entity, product, model="test-model")
        )

        assert result == "ENRICHED DATABASE DOCS"
        assert entity.generated_docs == "ENRICHED DATABASE DOCS"
        # Pages: overview / schema / tables / documentation + provenance.
        assert set(entity.pages) == {
            "page_overview", "page_schema", "page_tables", "page_documentation",
        }
        assert entity.pages["page_documentation"]["content"] == result
        prov = entity.pages["page_overview"]["provenance"]
        assert prov["generator"] == "standard-llm"
        assert prov["prompt_file"] == "database_doc.md"
        assert prov["tools_used"] == {
            "schemas": "list_schemas",
            "tables": "list_tables",
            "describe": "describe_table",
        }
        assert prov["schema_fingerprint_source"] == "mcp_introspection"
        # Indexing: final docs, product dataset, database source scoping.
        assert len(indexing) == 1
        content, dataset, source_type, source_id = indexing[0]
        assert content == result
        assert dataset == "prod_prod_1"
        assert source_type == "database"
        assert source_id == "db_1"

    def test_skeleton_fallback_when_llm_empty(self, monkeypatch):
        _patch_generation(monkeypatch, llm_text="")
        entity = _fake_entity()

        result = asyncio.run(db_doc_mod.generate_database_docs(entity, _fake_product()))

        assert "## Tables" in result
        assert "`public.users`" in result
        assert entity.generated_docs == result
        # Skeleton source → no AI documentation page content.
        assert entity.pages["page_documentation"]["content"] == ""
        assert entity.pages["page_overview"]["provenance"]["generator"] == "skeleton"

    def test_secrets_masked_before_persist(self, monkeypatch):
        token = "ghp_" + "AB" * 15
        _patch_generation(monkeypatch, llm_text=f"Docs mention token: {token}")
        entity = _fake_entity()

        result = asyncio.run(db_doc_mod.generate_database_docs(entity, _fake_product()))

        assert token not in result
        assert "***REDACTED***" in result
        assert token not in entity.generated_docs

    def test_corroborate_filters_invented_identifiers(self, monkeypatch):
        """3.1: model-generated sentences naming identifiers that are NOT in
        the introspected schema are dropped; every removal lands in the
        provenance report."""
        llm_text = (
            "The schema stores users in `public.users`. "
            "Invented table `GhostArchiveTable` keeps audit rows."
        )
        _patch_generation(monkeypatch, llm_text=llm_text)
        entity = _fake_entity()

        result = asyncio.run(
            db_doc_mod.generate_database_docs(entity, _fake_product())
        )

        assert "GhostArchiveTable" not in result
        assert result == "The schema stores users in `public.users`."
        prov = entity.pages["page_documentation"]["provenance"]
        assert prov["corroborate"] == {"removed": ["GhostArchiveTable"]}
        assert "GhostArchiveTable" not in entity.generated_docs

    def test_corroborate_skips_skeleton_source(self, monkeypatch):
        """Skeleton output is deterministic evidence — never filtered, no
        corroborate key in provenance."""
        _patch_generation(monkeypatch, llm_text="")
        entity = _fake_entity()

        result = asyncio.run(
            db_doc_mod.generate_database_docs(entity, _fake_product())
        )

        assert "`public.users`" in result
        assert "corroborate" not in entity.pages["page_overview"]["provenance"]

    def test_corroborate_fail_open_when_filter_empties(self, monkeypatch):
        """A filter that would empty the whole doc keeps the original text
        (and records nothing as removed)."""
        _patch_generation(monkeypatch, llm_text="Only `GhostArchiveTable` here.")
        entity = _fake_entity()

        result = asyncio.run(
            db_doc_mod.generate_database_docs(entity, _fake_product())
        )

        assert result == "Only `GhostArchiveTable` here."
        assert "corroborate" not in entity.pages["page_overview"]["provenance"]

    def test_no_tools_raises(self, monkeypatch):
        _patch_generation(monkeypatch, llm_text="x", tools=[])
        with pytest.raises(ValueError, match="No MCP tools"):
            asyncio.run(
                db_doc_mod.generate_database_docs(_fake_entity(), _fake_product())
            )

    def test_no_introspection_tools_raises(self, monkeypatch):
        tools = [FakeTool("execute_query"), FakeTool("health_check")]
        _patch_generation(monkeypatch, llm_text="x", tools=tools)
        with pytest.raises(ValueError, match="introspection tools"):
            asyncio.run(
                db_doc_mod.generate_database_docs(_fake_entity(), _fake_product())
            )

    def test_empty_introspection_raises(self, monkeypatch):
        tools = [
            FakeTool("list_schemas", {}, '["public"]'),
            FakeTool("list_tables", {}, "[]"),
        ]
        _patch_generation(monkeypatch, llm_text="x", tools=tools)
        with pytest.raises(ValueError, match="produced no tables"):
            asyncio.run(
                db_doc_mod.generate_database_docs(_fake_entity(), _fake_product())
            )

    def test_prompt_contains_evidence(self, monkeypatch):
        captured = []

        async def fake_llm(prompt, model, base_url=None, api_key=None):
            captured.append(prompt)
            return "docs"

        _patch_generation(monkeypatch, llm_text="docs")
        monkeypatch.setattr(db_doc_mod, "_llm_or_none", fake_llm)

        asyncio.run(db_doc_mod.generate_database_docs(_fake_entity(), _fake_product()))

        assert len(captured) == 1
        prompt = captured[0]
        assert "Main DB" in prompt
        assert "public.users" in prompt
        # The masked DSN goes into the prompt; a raw one never exists here.
        assert "***REDACTED***" in prompt

    def test_mermaid_repair_failure_non_fatal(self, monkeypatch):
        _patch_generation(monkeypatch, llm_text="docs with mermaid")

        async def boom(content, llm):
            raise RuntimeError("repair failed")

        monkeypatch.setattr(db_doc_mod, "run_repair_loop", boom)
        result = asyncio.run(
            db_doc_mod.generate_database_docs(_fake_entity(), _fake_product())
        )
        assert result == "docs with mermaid"

    def test_judge_only_for_model_generated(self, monkeypatch):
        from api.docgen.verification import JudgeVerdict

        _patch_generation(monkeypatch, llm_text="")
        monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "true")
        calls = []

        async def fake_judge(section_id, draft, evidence, *, model=None):
            calls.append((section_id, draft))
            return JudgeVerdict(verdict="consistent", issues=[])

        monkeypatch.setattr(db_doc_mod, "judge_section", fake_judge)
        asyncio.run(db_doc_mod.generate_database_docs(_fake_entity(), _fake_product()))
        # Skeleton source → deterministic evidence → judge skipped.
        assert calls == []

        # Now with model-generated docs the judge runs (and never blocks).
        monkeypatch.setattr(
            db_doc_mod, "_llm_or_none", lambda *a, **kw: _async_return("LLM docs")
        )
        asyncio.run(db_doc_mod.generate_database_docs(_fake_entity(), _fake_product()))
        assert len(calls) == 1
        assert calls[0][1] == "LLM docs"


# ============================================================================
# jobs.py database dispatch
# ============================================================================
class TestJobsDatabaseDispatch:
    def _seed(self, isolated_db, with_database=True):
        from api.models import DatabaseORM, ProductORM

        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_job_db", name="P"))
            if with_database:
                db.add(DatabaseORM(
                    id="db_job_1", product_id="prod_job_db", name="Main DB",
                    source="manual",
                ))
            db.commit()

    def test_database_success(self, isolated_db, monkeypatch):
        self._seed(isolated_db)
        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        import api.docgen.database as database_mod

        async def fake_generate(entity, product, model=None, language="ru", progress=None):
            entity.generated_docs = "DB docs"
            return "DB docs"

        monkeypatch.setattr(database_mod, "generate_database_docs", fake_generate)

        job_id = jobs_mod.create_job("prod_job_db", "database", "db_job_1")
        asyncio.run(jobs_mod._run_docgen_job_async(
            job_id, "prod_job_db", "database", "db_job_1", None, "ru"
        ))

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "succeeded"
        assert job["docs_chars"] == len("DB docs")
        assert job["error"] is None

    def test_database_not_found_fails(self, isolated_db, monkeypatch):
        self._seed(isolated_db, with_database=False)
        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        job_id = jobs_mod.create_job("prod_job_db", "database", "missing_db")
        asyncio.run(jobs_mod._run_docgen_job_async(
            job_id, "prod_job_db", "database", "missing_db", None, "ru"
        ))

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "failed"
        assert "Database not found" in job["error"]

    def test_generator_error_fails_job(self, isolated_db, monkeypatch):
        self._seed(isolated_db)
        monkeypatch.setattr(jobs_mod, "SessionLocal", isolated_db.SessionLocal)

        import api.docgen.database as database_mod

        async def fake_generate(entity, product, model=None, language="ru", progress=None):
            raise ValueError("MCP server unreachable")

        monkeypatch.setattr(database_mod, "generate_database_docs", fake_generate)

        job_id = jobs_mod.create_job("prod_job_db", "database", "db_job_1")
        asyncio.run(jobs_mod._run_docgen_job_async(
            job_id, "prod_job_db", "database", "db_job_1", None, "ru"
        ))

        job = jobs_mod.get_job(job_id)
        assert job["status"] == "failed"
        assert "unreachable" in job["error"]
