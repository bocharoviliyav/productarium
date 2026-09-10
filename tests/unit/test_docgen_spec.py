"""Unit tests for api.docgen.spec (OpenAPI/AsyncAPI doc generation).

Covers: _parse_spec, _schema_field_table, _render_openapi_skeleton,
_render_asyncapi_skeleton, _render_raw_fallback, _generate_spec_doc,
generate_openapi_docs, generate_asyncapi_docs.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.docgen.spec as spec_mod


# ============================================================================
# _parse_spec
# ============================================================================
class TestParseSpec:
    def test_empty_returns_none(self):
        assert spec_mod._parse_spec("") is None
        assert spec_mod._parse_spec("   ") is None
        assert spec_mod._parse_spec(None) is None

    def test_valid_json(self):
        content = json.dumps({"openapi": "3.0.0", "info": {"title": "Test"}})
        result = spec_mod._parse_spec(content)
        assert result is not None
        assert result["openapi"] == "3.0.0"
        assert result["info"]["title"] == "Test"

    def test_valid_yaml(self):
        content = "openapi: 3.0.0\ninfo:\n  title: Test\n"
        result = spec_mod._parse_spec(content)
        assert result is not None
        assert result["openapi"] == "3.0.0"
        assert result["info"]["title"] == "Test"

    def test_invalid_returns_none(self):
        assert spec_mod._parse_spec("not valid: [json or yaml") is None

    def test_non_dict_returns_none(self):
        assert spec_mod._parse_spec("[1, 2, 3]") is None
        assert spec_mod._parse_spec('"just a string"') is None
        assert spec_mod._parse_spec("42") is None


# ============================================================================
# _schema_field_table
# ============================================================================
class TestSchemaFieldTable:
    def test_empty_schema(self):
        lines = spec_mod._schema_field_table({})
        assert lines == [
            "| Поле | Тип | Обязательное | Описание |",
            "|------|-----|--------------|----------|",
        ]

    def test_none_schema(self):
        lines = spec_mod._schema_field_table(None)
        assert len(lines) == 2  # just the header

    def test_with_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The ID"},
                "name": {"type": "string", "description": "The name"},
            },
            "required": ["id"],
        }
        lines = spec_mod._schema_field_table(schema)
        assert len(lines) == 4  # 2 header + 2 fields
        assert "`id`" in lines[2]
        assert "да" in lines[2]  # required
        assert "`name`" in lines[3]
        assert "нет" in lines[3]  # not required

    def test_ref_type(self):
        schema = {
            "properties": {
                "user": {"$ref": "#/components/schemas/User"},
            },
        }
        lines = spec_mod._schema_field_table(schema)
        assert "#/components/schemas/User" in lines[2]

    def test_list_type(self):
        schema = {
            "properties": {
                "tags": {"type": ["string", "null"]},
            },
        }
        lines = spec_mod._schema_field_table(schema)
        assert "string | null" in lines[2]

    def test_multiline_description(self):
        schema = {
            "properties": {
                "desc": {"type": "string", "description": "Line 1\nLine 2"},
            },
        }
        lines = spec_mod._schema_field_table(schema)
        assert "Line 1 Line 2" in lines[2]


# ============================================================================
# _render_openapi_skeleton
# ============================================================================
class TestRenderOpenapiSkeleton:
    def test_minimal_spec(self):
        spec = {"info": {"title": "My API", "version": "1.0.0"}}
        md = spec_mod._render_openapi_skeleton(spec)
        assert "# My API" in md
        assert "`1.0.0`" in md

    def test_with_description(self):
        spec = {"info": {"title": "My API", "version": "1.0", "description": "A test API"}}
        md = spec_mod._render_openapi_skeleton(spec)
        assert "A test API" in md

    def test_with_servers(self):
        spec = {
            "info": {"title": "My API"},
            "servers": [
                {"url": "https://api.example.com", "description": "Production"},
                {"url": "http://localhost:3000", "description": "Dev"},
            ],
        }
        md = spec_mod._render_openapi_skeleton(spec)
        assert "https://api.example.com" in md
        assert "Production" in md
        assert "http://localhost:3000" in md

    def test_with_paths(self):
        spec = {
            "info": {"title": "My API"},
            "paths": {
                "/users": {
                    "get": {"summary": "List users"},
                    "post": {"summary": "Create user"},
                },
                "/users/{id}": {
                    "delete": {"summary": "Delete user"},
                },
            },
        }
        md = spec_mod._render_openapi_skeleton(spec)
        assert "GET" in md
        assert "/users" in md
        assert "List users" in md
        assert "POST" in md
        assert "Create user" in md
        assert "DELETE" in md
        assert "/users/{id}" in md

    def test_paths_filters_non_methods(self):
        spec = {
            "info": {"title": "My API"},
            "paths": {
                "/users": {
                    "get": {"summary": "List"},
                    "parameters": [{"name": "filter"}],
                },
            },
        }
        md = spec_mod._render_openapi_skeleton(spec)
        assert "GET" in md
        assert "parameters" not in md  # not a method

    def test_with_schemas(self):
        spec = {
            "info": {"title": "My API"},
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                        },
                        "required": ["id"],
                    },
                },
            },
        }
        md = spec_mod._render_openapi_skeleton(spec)
        assert "### User" in md
        assert "`id`" in md
        assert "integer" in md

    def test_empty_spec(self):
        md = spec_mod._render_openapi_skeleton({})
        assert "OpenAPI" in md

    def test_none_info(self):
        md = spec_mod._render_openapi_skeleton({"info": None})
        assert "OpenAPI" in md


# ============================================================================
# _render_asyncapi_skeleton
# ============================================================================
class TestRenderAsyncapiSkeleton:
    def test_minimal_spec(self):
        spec = {"asyncapi": "2.6.0", "info": {"title": "My Async API"}}
        md = spec_mod._render_asyncapi_skeleton(spec)
        assert "# My Async API" in md
        assert "`2.6.0`" in md

    def test_with_version_and_description(self):
        spec = {
            "asyncapi": "2.6.0",
            "info": {"title": "My API", "version": "1.0", "description": "Event stream"},
        }
        md = spec_mod._render_asyncapi_skeleton(spec)
        assert "`1.0`" in md
        assert "Event stream" in md

    def test_with_servers(self):
        spec = {
            "asyncapi": "2.6.0",
            "info": {"title": "My API"},
            "servers": {
                "production": {"url": "mqtt://prod.example.com", "protocol": "mqtt", "description": "Prod"},
            },
        }
        md = spec_mod._render_asyncapi_skeleton(spec)
        assert "production" in md
        assert "mqtt://prod.example.com" in md
        assert "mqtt" in md
        assert "Prod" in md

    def test_with_channels(self):
        spec = {
            "asyncapi": "2.6.0",
            "info": {"title": "My API"},
            "channels": {
                "user/created": {
                    "subscribe": {"summary": "User created event", "message": {"name": "UserCreated"}},
                    "publish": {"summary": "Publish event", "message": "EventMsg"},
                },
            },
        }
        md = spec_mod._render_asyncapi_skeleton(spec)
        assert "user/created" in md
        assert "subscribe" in md
        assert "publish" in md
        assert "UserCreated" in md

    def test_channels_with_ref_message(self):
        spec = {
            "asyncapi": "2.6.0",
            "info": {"title": "My API"},
            "channels": {
                "test/channel": {
                    "subscribe": {"message": {"$ref": "#/components/messages/Test"}},
                },
            },
        }
        md = spec_mod._render_asyncapi_skeleton(spec)
        assert "#/components/messages/Test" in md

    def test_with_schemas(self):
        spec = {
            "asyncapi": "2.6.0",
            "info": {"title": "My API"},
            "components": {
                "schemas": {
                    "Event": {
                        "properties": {"id": {"type": "string"}},
                    },
                },
            },
        }
        md = spec_mod._render_asyncapi_skeleton(spec)
        assert "### Event" in md

    def test_empty_spec(self):
        md = spec_mod._render_asyncapi_skeleton({})
        assert "AsyncAPI" in md


# ============================================================================
# _render_raw_fallback
# ============================================================================
class TestRenderRawFallback:
    def test_renders_with_name(self):
        class FakeSpec:
            name = "MySpec"
        md = spec_mod._render_raw_fallback("OpenAPI", "raw: content here", FakeSpec())
        assert "# OpenAPI: MySpec" in md
        assert "raw: content here" in md
        assert "```yaml" in md

    def test_uses_label_when_no_name(self):
        class FakeSpec:
            name = None
        md = spec_mod._render_raw_fallback("AsyncAPI", "content", FakeSpec())
        assert "# AsyncAPI: AsyncAPI" in md

    def test_caps_long_content(self):
        class FakeSpec:
            name = "S"
        long_content = "x" * 10000
        md = spec_mod._render_raw_fallback("OpenAPI", long_content, FakeSpec())
        assert "обрезано" in md


# ============================================================================
# _generate_spec_doc (shared flow)
# ============================================================================
class TestGenerateSpecDoc:
    @pytest.fixture(autouse=True)
    def _hermetic_flow(self, monkeypatch):
        """Keep these (Wave-A era) tests on the standard-LLM seam: no react
        agent build (would hit a live model) and no judge stage."""
        monkeypatch.setattr(spec_mod, "_build_spec_agent", lambda *a, **kw: (None, None))
        monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "false")

    @pytest.fixture
    def fake_spec(self):
        class FakeSpec:
            name = "TestSpec"
            content = json.dumps({
                "openapi": "3.0.0",
                "info": {"title": "Test API", "version": "1.0"},
                "paths": {"/users": {"get": {"summary": "List users"}}},
            })
            kind = "openapi"
        return FakeSpec()

    @pytest.fixture
    def fake_product(self):
        class P:
            id = "prod_test"
        return P()

    def test_empty_content_raises(self, fake_product):
        class EmptySpec:
            name = "Empty"
            content = ""
        with pytest.raises(ValueError, match="empty content"):
            asyncio.run(spec_mod.generate_openapi_docs(EmptySpec(), fake_product))

    def test_whitespace_content_raises(self, fake_product):
        class WsSpec:
            name = "Ws"
            content = "   \n  "
        with pytest.raises(ValueError):
            asyncio.run(spec_mod.generate_openapi_docs(WsSpec(), fake_product))

    def test_openapi_happy_path_with_llm(self, fake_spec, fake_product, monkeypatch):
        """LLM enrichment returns text -> that text is used."""
        monkeypatch.setattr(spec_mod, "_llm_or_none", lambda *a, **kw: _async_return("Enriched API documentation"))
        monkeypatch.setattr(spec_mod, "_make_repair_llm", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "_index_in_background", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "run_repair_loop", lambda *a, **kw: _async_return_pair(("Enriched API documentation", {})))

        result = asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert "Enriched API documentation" in result
        assert fake_spec.content == "Enriched API documentation"

    def test_openapi_falls_back_to_skeleton_when_llm_empty(self, fake_spec, fake_product, monkeypatch):
        """LLM returns empty -> skeleton is used."""
        monkeypatch.setattr(spec_mod, "_llm_or_none", lambda *a, **kw: _async_return(""))
        monkeypatch.setattr(spec_mod, "_make_repair_llm", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "_index_in_background", lambda *a, **kw: None)
        # run_repair_loop must pass through the content unchanged (no repair needed)
        monkeypatch.setattr(spec_mod, "run_repair_loop", lambda content, llm: _async_return_pair((content, {})))

        result = asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert "Test API" in result
        assert "/users" in result

    def test_openapi_invalid_spec_uses_raw_fallback(self, fake_product, monkeypatch):
        class BadSpec:
            name = "BadSpec"
            content = "this is not valid: [yaml or json"
        monkeypatch.setattr(spec_mod, "_llm_or_none", lambda *a, **kw: _async_return(""))
        monkeypatch.setattr(spec_mod, "_make_repair_llm", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "_index_in_background", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "run_repair_loop", lambda content, llm: _async_return_pair((content, {})))

        result = asyncio.run(spec_mod.generate_openapi_docs(BadSpec(), fake_product))
        assert "OpenAPI: BadSpec" in result
        assert "```yaml" in result

    def test_asyncapi_happy_path(self, fake_product, monkeypatch):
        class AsyncSpec:
            name = "AsyncSpec"
            content = json.dumps({
                "asyncapi": "2.6.0",
                "info": {"title": "Async API", "version": "1.0"},
                "channels": {"test": {"subscribe": {"summary": "Test sub"}}},
            })
            kind = "asyncapi"
        monkeypatch.setattr(spec_mod, "_llm_or_none", lambda *a, **kw: _async_return("Enriched async docs"))
        monkeypatch.setattr(spec_mod, "_make_repair_llm", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "_index_in_background", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "run_repair_loop", lambda *a, **kw: _async_return_pair(("Enriched async docs", {})))

        result = asyncio.run(spec_mod.generate_asyncapi_docs(AsyncSpec(), fake_product))
        assert "Enriched async docs" in result

    def test_indexing_called(self, fake_spec, fake_product, monkeypatch):
        indexing_calls = []
        def track_indexing(content, dataset, *, source_type="codebase", source_id=None):
            indexing_calls.append((content, dataset, source_type, source_id))
        monkeypatch.setattr(spec_mod, "_llm_or_none", lambda *a, **kw: _async_return(""))
        monkeypatch.setattr(spec_mod, "_make_repair_llm", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "_index_in_background", track_indexing)
        monkeypatch.setattr(spec_mod, "run_repair_loop", lambda content, llm: _async_return_pair((content, {})))

        asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert len(indexing_calls) == 1
        assert indexing_calls[0][1] == "prod_prod_test"
        # The rerouted caller now passes source_type=spec so the memory backend
        # can scope the upsert (idempotent per source). source_id comes from
        # spec.id; the FakeSpec fixture has no id, so it is None here.
        assert indexing_calls[0][2] == "spec"
        assert indexing_calls[0][3] is None

    def test_spec_content_write_failure_non_fatal(self, fake_product, monkeypatch):
        class WriteFailSpec:
            name = "WF"
            def __init__(self):
                self._content = json.dumps({"openapi": "3.0.0", "info": {"title": "T"}})
            @property
            def content(self):
                return self._content
            @content.setter
            def content(self, v):
                raise RuntimeError("cannot write")

        monkeypatch.setattr(spec_mod, "_llm_or_none", lambda *a, **kw: _async_return("docs"))
        monkeypatch.setattr(spec_mod, "_make_repair_llm", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "_index_in_background", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "run_repair_loop", lambda *a, **kw: _async_return_pair(("docs", {})))

        # Should not raise even though spec.content setter fails
        result = asyncio.run(spec_mod.generate_openapi_docs(WriteFailSpec(), fake_product))
        assert "docs" in result

    def test_mermaid_repair_failure_non_fatal(self, fake_spec, fake_product, monkeypatch):
        monkeypatch.setattr(spec_mod, "_llm_or_none", lambda *a, **kw: _async_return("docs with mermaid"))
        monkeypatch.setattr(spec_mod, "_make_repair_llm", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "_index_in_background", lambda *a, **kw: None)

        async def boom_repair(content, llm):
            raise RuntimeError("repair failed")
        monkeypatch.setattr(spec_mod, "run_repair_loop", boom_repair)

        result = asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert "docs with mermaid" in result


# ============================================================================
# Wave D: _lookup_path (dot-path resolution over the parsed spec)
# ============================================================================
class TestLookupPath:
    SPEC = {
        "info": {"title": "Test API", "version": "1.0"},
        "paths": {
            "/users": {"get": {"summary": "List users"}},
        },
        "channels": [{"name": "ch0"}, {"name": "ch1"}],
        "a.b": "dotted-key-value",
    }

    def test_nested_keys(self):
        assert spec_mod._lookup_path(self.SPEC, "info.title") == "Test API"
        assert spec_mod._lookup_path(self.SPEC, "info.version") == "1.0"

    def test_openapi_dotted_path_key(self):
        # Path keys contain dots/slashes; the longest join must win.
        value = spec_mod._lookup_path(self.SPEC, "paths./users.get")
        assert value == {"summary": "List users"}

    def test_key_containing_dots(self):
        assert spec_mod._lookup_path(self.SPEC, "a.b") == "dotted-key-value"

    def test_list_index(self):
        assert spec_mod._lookup_path(self.SPEC, "channels.0") == {"name": "ch0"}
        assert spec_mod._lookup_path(self.SPEC, "channels.1.name") == "ch1"

    def test_missing_returns_sentinel(self):
        assert spec_mod._lookup_path(self.SPEC, "nope.nope") is spec_mod._MISSING
        assert spec_mod._lookup_path(self.SPEC, "info.nope") is spec_mod._MISSING

    def test_out_of_range_and_bad_index(self):
        assert spec_mod._lookup_path(self.SPEC, "channels.9") is spec_mod._MISSING
        assert spec_mod._lookup_path(self.SPEC, "channels.abc") is spec_mod._MISSING

    def test_descend_into_scalar_is_missing(self):
        assert spec_mod._lookup_path(self.SPEC, "info.title.deeper") is spec_mod._MISSING

    def test_empty_path_returns_root(self):
        assert spec_mod._lookup_path(self.SPEC, "") == self.SPEC


class TestSpecLookupTool:
    def test_invoke_returns_pretty_json(self):
        tool = spec_mod.make_spec_lookup_tool({"info": {"title": "Test API"}})
        out = tool.invoke({"path": "info"})
        assert "Test API" in out

    def test_invoke_missing_path_error(self):
        tool = spec_mod.make_spec_lookup_tool({"info": {}})
        assert "ERROR" in tool.invoke({"path": "ghost.path"})


class _FakeSpecAgent:
    """Stub react agent: ainvoke returns a scripted message list."""

    def __init__(self, text="Agent enriched docs", error=None):
        from langchain_core.messages import AIMessage, HumanMessage

        self.error = error
        self._messages = [HumanMessage(content="task"), AIMessage(content=text)]

    async def ainvoke(self, payload, config=None):
        if self.error:
            raise self.error
        return {"messages": self._messages}


# ============================================================================
# Wave D: enrich node agent path + guard node verification
# ============================================================================
class TestSpecAgentFlow:
    @pytest.fixture
    def fake_spec(self):
        class FakeSpec:
            name = "TestSpec"
            content = json.dumps({
                "openapi": "3.0.0",
                "info": {"title": "Test API", "version": "1.0"},
                "paths": {"/users": {"get": {"summary": "List users"}}},
            })
        return FakeSpec()

    @pytest.fixture
    def fake_product(self):
        class P:
            id = "prod_agent"
        return P()

    def _patch_flow(self, monkeypatch, agent):
        monkeypatch.setattr(
            spec_mod, "_build_spec_agent", lambda *a, **kw: (agent, None)
        )
        monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "false")
        monkeypatch.setattr(spec_mod, "_make_repair_llm", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "_index_in_background", lambda *a, **kw: None)
        monkeypatch.setattr(
            spec_mod, "run_repair_loop",
            lambda content, llm: _async_return_pair((content, {})),
        )

    def test_agent_text_used_and_written_back(self, fake_spec, fake_product, monkeypatch):
        self._patch_flow(monkeypatch, _FakeSpecAgent(text="Agent enriched docs"))
        # The standard-LLM fallback must NOT be consulted when the agent wins.
        monkeypatch.setattr(
            spec_mod, "_llm_or_none", lambda *a, **kw: _async_return("LLM fallback text")
        )

        result = asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert result == "Agent enriched docs"
        assert fake_spec.content == "Agent enriched docs"

    def test_agent_failure_falls_back_to_standard_llm(self, fake_spec, fake_product, monkeypatch):
        self._patch_flow(monkeypatch, _FakeSpecAgent(error=RuntimeError("agent down")))
        monkeypatch.setattr(
            spec_mod, "_llm_or_none", lambda *a, **kw: _async_return("LLM fallback text")
        )

        result = asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert result == "LLM fallback text"
        assert fake_spec.content == "LLM fallback text"

    def test_guard_masks_secrets_from_llm_text(self, fake_spec, fake_product, monkeypatch):
        """The guard node masks secret-looking values in model output before
        the doc is persisted (spec.content) or indexed."""
        self._patch_flow(monkeypatch, _FakeSpecAgent(error=RuntimeError("down")))
        token = "ghp_" + "AB" * 15
        monkeypatch.setattr(
            spec_mod, "_llm_or_none",
            lambda *a, **kw: _async_return(f"Docs mention token: {token}"),
        )

        result = asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert token not in result
        assert "***REDACTED***" in result
        assert token not in fake_spec.content


class TestSpecJudgePolicy:
    """The judge runs ONLY for model-generated docs (agent / standard-llm);
    the deterministic skeleton is grounded in the spec by construction."""

    @pytest.fixture
    def fake_spec(self):
        class FakeSpec:
            name = "TestSpec"
            content = json.dumps({
                "openapi": "3.0.0",
                "info": {"title": "Test API"},
            })
        return FakeSpec()

    @pytest.fixture
    def fake_product(self):
        class P:
            id = "prod_judge"
        return P()

    def _common_patches(self, monkeypatch):
        from api.docgen.verification import JudgeVerdict

        monkeypatch.setattr(spec_mod, "_build_spec_agent", lambda *a, **kw: (None, None))
        monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "true")
        monkeypatch.setattr(spec_mod, "_make_repair_llm", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "_index_in_background", lambda *a, **kw: None)
        monkeypatch.setattr(
            spec_mod, "run_repair_loop",
            lambda content, llm: _async_return_pair((content, {})),
        )
        calls = []

        async def fake_judge(section_id, draft, evidence, *, model=None):
            calls.append((section_id, draft))
            return JudgeVerdict(verdict="consistent", issues=[])

        monkeypatch.setattr(spec_mod, "judge_section", fake_judge)
        return calls

    def test_judge_skipped_for_skeleton_source(self, fake_spec, fake_product, monkeypatch):
        calls = self._common_patches(monkeypatch)
        monkeypatch.setattr(
            spec_mod, "_llm_or_none", lambda *a, **kw: _async_return("")
        )

        result = asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert "Test API" in result  # skeleton used
        assert calls == []  # deterministic skeleton -> no judge call

    def test_judge_called_for_model_generated_source(self, fake_spec, fake_product, monkeypatch):
        calls = self._common_patches(monkeypatch)
        monkeypatch.setattr(
            spec_mod, "_llm_or_none", lambda *a, **kw: _async_return("LLM docs")
        )

        asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert len(calls) == 1
        assert calls[0][1] == "LLM docs"  # the draft is judged


class TestSpecGraphRuntime:
    """The LangGraph runtime and the straight-line fallback both work."""

    @pytest.fixture
    def fake_spec(self):
        class FakeSpec:
            name = "TestSpec"
            content = json.dumps({"openapi": "3.0.0", "info": {"title": "T"}})
        return FakeSpec()

    @pytest.fixture
    def fake_product(self):
        class P:
            id = "prod_graph"
        return P()

    def _patches(self, monkeypatch):
        monkeypatch.setattr(spec_mod, "_build_spec_agent", lambda *a, **kw: (None, None))
        monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "false")
        monkeypatch.setattr(spec_mod, "_make_repair_llm", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "_index_in_background", lambda *a, **kw: None)
        monkeypatch.setattr(
            spec_mod, "run_repair_loop",
            lambda content, llm: _async_return_pair((content, {})),
        )

    def test_graph_compiles(self):
        assert spec_mod._get_spec_graph() is not None

    def test_graph_flow_end_to_end(self, fake_spec, fake_product, monkeypatch):
        """Through the REAL compiled graph (parse → enrich → guard)."""
        self._patches(monkeypatch)
        monkeypatch.setattr(
            spec_mod, "_llm_or_none", lambda *a, **kw: _async_return("Graph docs")
        )

        result = asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert result == "Graph docs"
        assert fake_spec.content == "Graph docs"

    def test_straight_line_fallback(self, fake_spec, fake_product, monkeypatch):
        """With the graph unavailable the SAME nodes run straight-line."""
        self._patches(monkeypatch)
        monkeypatch.setattr(spec_mod, "_get_spec_graph", lambda: None)
        monkeypatch.setattr(
            spec_mod, "_llm_or_none", lambda *a, **kw: _async_return("Straight-line docs")
        )

        result = asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert result == "Straight-line docs"
        assert fake_spec.content == "Straight-line docs"


# ============================================================================
# Cross-context for CODEBASE docgen: spec digest + spec_lookup tool
# ============================================================================
class TestSpecContextEnabled:
    def test_default_on(self, monkeypatch):
        monkeypatch.delenv("DOCGEN_SPEC_CONTEXT_ENABLED", raising=False)
        assert spec_mod.spec_context_enabled() is True

    def test_env_off_variants(self, monkeypatch):
        for off in ("false", "0", "no", "off"):
            monkeypatch.setenv("DOCGEN_SPEC_CONTEXT_ENABLED", off)
            assert spec_mod.spec_context_enabled() is False

    def test_env_legacy_truthy_variants(self, monkeypatch):
        # Anything except the explicit off-words (empty included) is ON.
        for on in ("true", "1", "yes", "garbage", ""):
            monkeypatch.setenv("DOCGEN_SPEC_CONTEXT_ENABLED", on)
            assert spec_mod.spec_context_enabled() is True


_OPENAPI_SPEC = json.dumps({
    "openapi": "3.0.0",
    "info": {"title": "Billing API", "version": "2.1"},
    "servers": [{"url": "https://api.example.com"}],
    "paths": {
        "/orders": {
            "get": {"summary": "List orders"},
            "post": {"summary": "Create order"},
        },
    },
    "components": {
        "schemas": {
            "Order": {"type": "object", "properties": {"id": {"type": "integer"}}},
            "User": {"type": "object", "properties": {"id": {"type": "integer"}}},
        }
    },
}, ensure_ascii=False)

_ASYNCAPI_SPEC = json.dumps({
    "asyncapi": "2.6.0",
    "info": {"title": "Events", "version": "1.0"},
    "servers": {"prod": {"protocol": "kafka", "url": "kafka:9092"}},
    "channels": {
        "orders.created": {"publish": {"summary": "Order created"}},
        "orders.updated": {"subscribe": {"summary": "Order updated"}},
    },
}, ensure_ascii=False)


class TestProductSpecContext:
    @pytest.fixture()
    def seeded(self, isolated_db):
        from api.models import ProductORM, SpecORM

        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_specctx", name="P"))
            s.add(SpecORM(
                id="spec_api", product_id="prod_specctx", name="Public API",
                kind="openapi", content=_OPENAPI_SPEC,
            ))
            s.add(SpecORM(
                id="spec_events", product_id="prod_specctx", name="Events",
                kind="asyncapi", content=_ASYNCAPI_SPEC,
            ))
            s.add(SpecORM(
                id="spec_broken", product_id="prod_specctx", name="Broken",
                kind="openapi", content="not valid: [json or yaml",
            ))
            s.add(SpecORM(
                id="spec_empty", product_id="prod_specctx", name="Empty",
                kind="openapi", content="   ",
            ))
            s.commit()
        return isolated_db

    def test_digest_menu_lines_and_markers(self, seeded):
        out = spec_mod.product_spec_context("prod_specctx")
        assert out.startswith("### Контракт-контекст продукта (спецификации API)")
        # Explicit "do not cite as paths" marker for the citation guard.
        assert "не цитировать как пути" in out
        assert "`spec_lookup`" in out
        # openapi menu line: name, kind, title+version, op/schema counts.
        assert (
            "**Public API** (openapi; Billing API v2.1; операций 2, схем 2)"
            in out
        )
        assert "`GET /orders`" in out
        assert "`POST /orders`" in out
        assert "схемы: `Order`" in out
        assert "`https://api.example.com`" in out
        # asyncapi menu line: channels + ops + server protocol.
        assert "**Events** (asyncapi; Events v1.0; каналов 2, схем 0)" in out
        assert "`orders.created publish`" in out
        assert "`orders.updated subscribe`" in out
        assert "`prod`:kafka" in out
        # Unparseable spec degrades to a head-of-text line, never raises.
        assert "**Broken** (openapi; не разобрана)" in out
        # The empty-content spec is skipped entirely.
        assert "Empty" not in out

    def test_digest_caps_long_output(self, seeded):
        out = spec_mod.product_spec_context("prod_specctx", max_chars=300)
        # cap() appends its own truncation marker on top of the slice.
        assert len(out) <= 300 + 100

    def test_no_rows_returns_empty(self, isolated_db):
        assert spec_mod.product_spec_context("prod_none") == ""

    def test_empty_product_returns_empty(self):
        assert spec_mod.product_spec_context("") == ""


class TestBuildSpecTools:
    @pytest.fixture()
    def seeded(self, isolated_db):
        from api.models import ProductORM, SpecORM

        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_spectools", name="P"))
            s.add(SpecORM(
                id="spec_tool_api", product_id="prod_spectools", name="Public API",
                kind="openapi", content=_OPENAPI_SPEC,
            ))
            s.commit()
        return isolated_db

    def test_tool_name_and_hit(self, seeded):
        tools = spec_mod.build_spec_tools("prod_spectools")
        assert [t.name for t in tools] == ["spec_lookup"]
        out = tools[0].invoke({"spec_name": "Public API", "path": "info.title"})
        assert "не цитировать как пути" in out
        assert "Billing API" in out

    def test_path_with_dots_in_key(self, seeded):
        tools = spec_mod.build_spec_tools("prod_spectools")
        out = tools[0].invoke({
            "spec_name": "Public API", "path": "paths./orders.get.summary",
        })
        assert "List orders" in out

    def test_case_insensitive_spec_name(self, seeded):
        tools = spec_mod.build_spec_tools("prod_spectools")
        out = tools[0].invoke({"spec_name": "public api", "path": "info.title"})
        assert "Billing API" in out

    def test_unknown_spec_lists_available(self, seeded):
        tools = spec_mod.build_spec_tools("prod_spectools")
        out = tools[0].invoke({"spec_name": "Ghost", "path": ""})
        assert out.startswith("ERROR: unknown spec")
        assert "`Public API`" in out

    def test_missing_path_lists_root_keys(self, seeded):
        tools = spec_mod.build_spec_tools("prod_spectools")
        out = tools[0].invoke({"spec_name": "Public API", "path": "nope.nope"})
        assert "ERROR: path not found" in out
        assert "`paths`" in out

    def test_result_capped(self, seeded):
        from api.models import ProductORM, SpecORM

        big = json.loads(_OPENAPI_SPEC)
        big["components"] = {"schemas": {
            f"S{i}": {"type": "object", "properties": {
                f"field_{j}": {"type": "string", "description": "d" * 200}
                for j in range(20)
            }}
            for i in range(40)
        }}
        with seeded.SessionLocal() as s:
            s.add(ProductORM(id="prod_spectools_big", name="P2"))
            s.add(SpecORM(
                id="spec_big", product_id="prod_spectools_big", name="Big",
                kind="openapi", content=json.dumps(big),
            ))
            s.commit()

        tools = spec_mod.build_spec_tools("prod_spectools_big")
        assert len(tools) == 1
        out = tools[0].invoke({"spec_name": "Big", "path": "components"})
        assert len(out) <= spec_mod._SPEC_TOOL_MAX_CHARS + 100
        assert "обрезано для контекста LLM" in out

    def test_no_specs_returns_empty(self, isolated_db):
        from api.models import ProductORM

        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_nospec", name="P"))
            s.commit()
        assert spec_mod.build_spec_tools("prod_nospec") == []

    def test_only_unparseable_specs_returns_empty(self, isolated_db):
        from api.models import ProductORM, SpecORM

        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_badonly", name="P"))
            s.add(SpecORM(
                id="spec_bad", product_id="prod_badonly", name="Bad",
                kind="openapi", content="not valid: [json or yaml",
            ))
            s.commit()
        # No parseable specs -> no dead tool in the subagent specs.
        assert spec_mod.build_spec_tools("prod_badonly") == []

    def test_empty_product_returns_empty(self):
        assert spec_mod.build_spec_tools("") == []


# ============================================================================
# Helpers
# ============================================================================
async def _async_return(value):
    return value


async def _async_return_pair(value):
    return value
