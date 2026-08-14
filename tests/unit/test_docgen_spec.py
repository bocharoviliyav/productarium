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
from api.docgen._common import _cognee_dataset


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
        def track_indexing(content, dataset):
            indexing_calls.append((content, dataset))
        monkeypatch.setattr(spec_mod, "_llm_or_none", lambda *a, **kw: _async_return(""))
        monkeypatch.setattr(spec_mod, "_make_repair_llm", lambda *a, **kw: None)
        monkeypatch.setattr(spec_mod, "_index_in_background", track_indexing)
        monkeypatch.setattr(spec_mod, "run_repair_loop", lambda content, llm: _async_return_pair((content, {})))

        asyncio.run(spec_mod.generate_openapi_docs(fake_spec, fake_product))
        assert len(indexing_calls) == 1
        assert indexing_calls[0][1] == "prod_prod_test"

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
# Helpers
# ============================================================================
async def _async_return(value):
    return value


async def _async_return_pair(value):
    return value
