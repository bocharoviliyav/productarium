"""Unit tests for api.docgen.summary (AI product summary generator).

Covers: _collect_summary_content (capping), _build_summary_prompt,
_clean_text, _SummaryLLM (mocked), _safe_build_summary_llm,
generate_product_summary (mocked LLM + empty content + LLM-unavailable).
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.docgen.summary as summary_mod


# ============================================================================
# Fixtures
# ============================================================================
class FakeCodebase:
    def __init__(self, name, docs):
        self.name = name
        self.generated_docs = docs
        self.id = f"cb_{name}"


class FakeSpec:
    def __init__(self, name, content, kind="openapi"):
        self.name = name
        self.content = content
        self.kind = kind
        self.id = f"spec_{name}"


class FakeNode:
    def __init__(self, title, md):
        self.title = title
        self.content_md = md
        self.id = f"node_{title}"


class FakeProduct:
    def __init__(self, name="MyProduct", pid="prod_123"):
        self.name = name
        self.id = pid


# ============================================================================
# _collect_summary_content
# ============================================================================
class TestCollectSummaryContent:
    def test_empty_all(self):
        assert summary_mod._collect_summary_content([], [], []) == ""

    def test_codebases_only(self):
        cbs = [FakeCodebase("app", "# App\nDocs here.")]
        result = summary_mod._collect_summary_content(cbs, [], [])
        assert "## Codebase: app" in result
        assert "Docs here." in result

    def test_specs_only(self):
        specs = [FakeSpec("api", "openapi: 3.0.0", kind="openapi")]
        result = summary_mod._collect_summary_content([], specs, [])
        assert "## Спецификация (openapi): api" in result
        assert "openapi: 3.0.0" in result

    def test_nodes_only(self):
        nodes = [FakeNode("Architecture", "# Architecture\nC4 model.")]
        result = summary_mod._collect_summary_content([], [], nodes)
        assert "## Страница базы знаний: Architecture" in result
        assert "C4 model." in result

    def test_all_combined(self):
        cbs = [FakeCodebase("app", "App docs")]
        specs = [FakeSpec("api", "spec content")]
        nodes = [FakeNode("Arch", "arch content")]
        result = summary_mod._collect_summary_content(cbs, specs, nodes)
        assert "## Codebase: app" in result
        assert "## Спецификация" in result
        assert "## Страница базы знаний: Arch" in result

    def test_skips_empty_content(self):
        cbs = [FakeCodebase("empty", ""), FakeCodebase("real", "content")]
        result = summary_mod._collect_summary_content(cbs, [], [])
        assert "## Codebase: real" in result
        assert "## Codebase: empty" not in result

    def test_skips_whitespace_content(self):
        cbs = [FakeCodebase("ws", "   \n  ")]
        result = summary_mod._collect_summary_content(cbs, [], [])
        assert result == ""

    def test_caps_large_content(self):
        cbs = [FakeCodebase("big", "x" * 50000)]
        result = summary_mod._collect_summary_content(cbs, [], [])
        assert len(result) <= summary_mod.SUMMARY_CONTEXT_MAX_CHARS + 100  # allow for cap suffix
        assert "обрезано" in result

    def test_uses_id_when_no_name(self):
        cb = FakeCodebase("named", "docs")
        cb.name = None
        result = summary_mod._collect_summary_content([cb], [], [])
        assert "cb_named" in result

    def test_spec_uses_id_when_no_name(self):
        spec = FakeSpec("named", "content")
        spec.name = None
        result = summary_mod._collect_summary_content([], [spec], [])
        assert "spec_named" in result

    def test_node_uses_id_when_no_title(self):
        node = FakeNode("titled", "content")
        node.title = None
        result = summary_mod._collect_summary_content([], [], [node])
        assert "node_titled" in result

    def test_spec_kind_defaults(self):
        spec = FakeSpec("s", "content")
        spec.kind = None
        result = summary_mod._collect_summary_content([], [spec], [])
        assert "## Спецификация (spec):" in result

    def test_none_inputs(self):
        assert summary_mod._collect_summary_content(None, None, None) == ""


# ============================================================================
# _build_summary_prompt
# ============================================================================
class TestBuildSummaryPrompt:
    def test_substitutes_placeholders(self):
        prompt = summary_mod._build_summary_prompt("MyProduct", "some content here")
        assert "MyProduct" in prompt
        assert "some content here" in prompt
        assert "{product_name}" not in prompt
        assert "{content}" not in prompt

    def test_empty_content(self):
        prompt = summary_mod._build_summary_prompt("Prod", "")
        assert "Prod" in prompt
        assert "{content}" not in prompt


# ============================================================================
# _clean_text
# ============================================================================
class TestCleanText:
    def test_empty(self):
        assert summary_mod._clean_text("") == ""
        assert summary_mod._clean_text(None) == ""

    def test_strips_whitespace(self):
        assert summary_mod._clean_text("  hello  ") == "hello"

    def test_strips_markdown_fence(self):
        text = "```markdown\n# Title\ncontent\n```"
        result = summary_mod._clean_text(text)
        assert "```" not in result
        assert "# Title" in result

    def test_strips_plain_fence(self):
        text = "```\ncode\n```"
        assert summary_mod._clean_text(text) == "code"

    def test_strips_fence_with_lang(self):
        text = "```python\nprint('hi')\n```"
        assert summary_mod._clean_text(text) == "print('hi')"


# ============================================================================
# _safe_build_summary_llm
# ============================================================================
class TestSafeBuildSummaryLlm:
    def test_returns_none_on_exception(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("no adalflow")
        monkeypatch.setattr(summary_mod, "_SummaryLLM", boom)
        assert summary_mod._safe_build_summary_llm("model") is None


# ============================================================================
# _SummaryLLM (mocked generator)
# ============================================================================
class TestSummaryLLM:
    def test_generate_success(self):
        llm = summary_mod._SummaryLLM.__new__(summary_mod._SummaryLLM)

        class FakeResult:
            error = None
            data = "summary text"

        class FakeGenerator:
            def __call__(self, prompt_kwargs=None):
                return FakeResult()

        llm.generator = FakeGenerator()
        result = asyncio.run(llm.generate("prompt"))
        assert result == "summary text"

    def test_generate_error_returns_empty(self):
        llm = summary_mod._SummaryLLM.__new__(summary_mod._SummaryLLM)

        class FakeResult:
            error = Exception("model error")
            data = None
            response = None
            answer = None
            raw_response = None
            output = None

        class FakeGenerator:
            def __call__(self, prompt_kwargs=None):
                return FakeResult()

        llm.generator = FakeGenerator()
        result = asyncio.run(llm.generate("prompt"))
        assert result == ""

    def test_generate_exception_propagates(self):
        """_SummaryLLM.generate does NOT catch generator exceptions; the caller
        (generate_product_summary) is responsible for catching them."""
        llm = summary_mod._SummaryLLM.__new__(summary_mod._SummaryLLM)

        class FakeGenerator:
            def __call__(self, prompt_kwargs=None):
                raise RuntimeError("boom")

        llm.generator = FakeGenerator()
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(llm.generate("prompt"))

    def test_generate_falls_through_attrs(self):
        llm = summary_mod._SummaryLLM.__new__(summary_mod._SummaryLLM)

        class FakeResult:
            error = None
            data = None
            response = None
            answer = None
            raw_response = None
            output = "found in output attr"

        class FakeGenerator:
            def __call__(self, prompt_kwargs=None):
                return FakeResult()

        llm.generator = FakeGenerator()
        result = asyncio.run(llm.generate("prompt"))
        assert result == "found in output attr"


# ============================================================================
# generate_product_summary
# ============================================================================
class TestGenerateProductSummary:
    def test_no_content_returns_empty(self):
        product = FakeProduct()
        result = asyncio.run(
            summary_mod.generate_product_summary(product, [], [], [])
        )
        assert result == ""

    def test_all_empty_content_returns_empty(self):
        product = FakeProduct()
        cbs = [FakeCodebase("e", "")]
        result = asyncio.run(
            summary_mod.generate_product_summary(product, cbs, [], [])
        )
        assert result == ""

    def test_llm_unavailable_returns_empty(self, monkeypatch):
        product = FakeProduct()
        cbs = [FakeCodebase("app", "docs content")]
        monkeypatch.setattr(summary_mod, "_safe_build_summary_llm", lambda *a, **kw: None)
        result = asyncio.run(
            summary_mod.generate_product_summary(product, cbs, [], [])
        )
        assert result == ""

    def test_llm_returns_summary(self, monkeypatch):
        product = FakeProduct()
        cbs = [FakeCodebase("app", "app docs")]
        specs = [FakeSpec("api", "spec content")]
        nodes = [FakeNode("arch", "arch content")]

        class FakeLLM:
            async def generate(self, prompt):
                return "Generated summary text"
        monkeypatch.setattr(summary_mod, "_safe_build_summary_llm", lambda *a, **kw: FakeLLM())
        result = asyncio.run(
            summary_mod.generate_product_summary(product, cbs, specs, nodes)
        )
        assert result == "Generated summary text"

    def test_llm_returns_fenced_text(self, monkeypatch):
        product = FakeProduct()
        cbs = [FakeCodebase("app", "docs")]

        class FakeLLM:
            async def generate(self, prompt):
                return "```markdown\nSummary inside fence\n```"
        monkeypatch.setattr(summary_mod, "_safe_build_summary_llm", lambda *a, **kw: FakeLLM())
        result = asyncio.run(
            summary_mod.generate_product_summary(product, cbs, [], [])
        )
        assert result == "Summary inside fence"

    def test_llm_raises_returns_empty(self, monkeypatch):
        product = FakeProduct()
        cbs = [FakeCodebase("app", "docs")]

        class FakeLLM:
            async def generate(self, prompt):
                raise RuntimeError("LLM down")
        monkeypatch.setattr(summary_mod, "_safe_build_summary_llm", lambda *a, **kw: FakeLLM())
        result = asyncio.run(
            summary_mod.generate_product_summary(product, cbs, [], [])
        )
        assert result == ""

    def test_product_name_fallback(self, monkeypatch):
        class P:
            name = ""
            id = "prod_x"
        cbs = [FakeCodebase("app", "docs")]

        class FakeLLM:
            async def generate(self, prompt):
                # Verify the product name fallback is used
                assert "prod_x" in prompt or "product" in prompt
                return "summary"
        monkeypatch.setattr(summary_mod, "_safe_build_summary_llm", lambda *a, **kw: FakeLLM())
        result = asyncio.run(
            summary_mod.generate_product_summary(P(), cbs, [], [])
        )
        assert result == "summary"

    def test_explicit_model_overrides_config(self, monkeypatch):
        product = FakeProduct()
        cbs = [FakeCodebase("app", "docs")]

        captured = {}
        def fake_build(model, base_url=None, api_key=None):
            captured["model"] = model
            class FakeLLM:
                async def generate(self, prompt):
                    return "summary"
            return FakeLLM()
        monkeypatch.setattr(summary_mod, "_safe_build_summary_llm", fake_build)
        result = asyncio.run(
            summary_mod.generate_product_summary(product, cbs, [], [], model="custom/model")
        )
        assert result == "summary"
        assert captured["model"] == "custom/model"
