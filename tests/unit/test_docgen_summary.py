"""Unit tests for api.docgen.summary (AI product summary generator).

Covers: context collection via _common.collect_entity_parts (all entity
kinds, per-entity fair cap, DB pages fallback, links JSON),
_build_summary_prompt, _clean_text, _SummaryLLM (mocked),
_safe_build_summary_llm, generate_product_summary (mocked LLM + empty
content + LLM-unavailable).
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


class FakeDatabase:
    def __init__(self, name, docs="", pages=None):
        self.name = name
        self.generated_docs = docs
        self.pages = pages or {}
        self.id = f"db_{name}"


class FakeLink:
    def __init__(self, name, content):
        self.name = name
        self.content = content
        self.id = f"link_{name}"


class FakeProduct:
    def __init__(self, name="MyProduct", pid="prod_123"):
        self.name = name
        self.id = pid


# ============================================================================
# Context collection via _common.collect_entity_parts (through the prompt)
# ============================================================================
class _PromptCaptureLLM:
    def __init__(self):
        self.prompt = ""

    async def generate(self, prompt):
        self.prompt = prompt
        return "summary text"


def _capture_prompt(monkeypatch, product, cbs=(), specs=(), nodes=(), **kw):
    """Run generate_product_summary with a mocked LLM; return (prompt, result)."""
    llm = _PromptCaptureLLM()
    monkeypatch.setattr(summary_mod, "_safe_build_summary_llm", lambda *a, **k: llm)
    result = asyncio.run(
        summary_mod.generate_product_summary(product, cbs, specs, nodes, **kw)
    )
    return llm.prompt, result


class TestSummaryContextCollector:
    def test_all_entity_kinds_reach_prompt(self, monkeypatch):
        links = [FakeLink("docs", '[{"url": "https://x.dev", "description": "Docs portal"}]')]
        prompt, result = _capture_prompt(
            monkeypatch,
            FakeProduct(),
            cbs=[FakeCodebase("app", "app docs")],
            specs=[FakeSpec("api", "spec content")],
            nodes=[FakeNode("arch", "arch content")],
            databases=[FakeDatabase("main", docs="db docs")],
            links=links,
        )
        assert result == "summary text"
        assert "## Codebase: app" in prompt
        assert "## Database: main" in prompt
        assert "## Спецификация (openapi): api" in prompt
        assert "## Страница базы знаний: arch" in prompt
        assert "## Ссылки: docs" in prompt
        assert "- https://x.dev: Docs portal" in prompt

    def test_later_entity_not_displaced_by_huge_codebase(self, monkeypatch):
        # Fair per-part cap: the huge FIRST codebase is truncated instead of
        # starving the entity that comes later in the collector's order.
        async def fake_ctx(**kw):
            return 8192  # hermetic window -> budget floor 20k, 10k per part

        monkeypatch.setattr("api.utils.get_model_context_window_async", fake_ctx)
        prompt, _ = _capture_prompt(
            monkeypatch,
            FakeProduct(),
            cbs=[FakeCodebase("big", "x" * 60_000)],
            nodes=[FakeNode("small", "tiny but present")],
        )
        assert "обрезано" in prompt  # codebase part capped, not global tail-cut
        assert "## Страница базы знаний: small" in prompt
        assert "tiny but present" in prompt

    def test_database_pages_fallback(self, monkeypatch):
        db = FakeDatabase("legacy", docs="", pages={
            "p1": {"content": "page one body"},
            "p2": {"content": "page two body"},
        })
        prompt, _ = _capture_prompt(monkeypatch, FakeProduct(), databases=[db])
        assert "## Database: legacy" in prompt
        assert "page one body" in prompt
        assert "page two body" in prompt

    def test_links_json_rendered(self, monkeypatch):
        raw = '[{"url": "https://a.dev", "description": "A"}, {"url": "https://b.dev", "description": "B"}]'
        prompt, _ = _capture_prompt(
            monkeypatch, FakeProduct(), links=[FakeLink("res", raw)]
        )
        assert "## Ссылки: res" in prompt
        assert "- https://a.dev: A" in prompt
        assert "- https://b.dev: B" in prompt

    def test_old_positional_signature_still_works(self, monkeypatch):
        # Legacy callers pass only (product, codebases, specs, nodes).
        prompt, result = _capture_prompt(
            monkeypatch, FakeProduct(), cbs=[FakeCodebase("app", "docs")]
        )
        assert result == "summary text"
        assert "## Codebase: app" in prompt

    def test_empty_databases_links_only_returns_empty(self):
        result = asyncio.run(
            summary_mod.generate_product_summary(
                FakeProduct(), [], [], [],
                databases=[FakeDatabase("empty", docs="", pages={})],
                links=[FakeLink("none", "")],
            )
        )
        assert result == ""

    def test_budget_scales_with_context_window(self, monkeypatch):
        async def fake_ctx(**kw):
            return 131_072  # 128k window -> budget capped at 120k chars

        monkeypatch.setattr("api.utils.get_model_context_window_async", fake_ctx)
        prompt, _ = _capture_prompt(
            monkeypatch, FakeProduct(), cbs=[FakeCodebase("big", "x" * 60_000)]
        )
        assert "обрезано" not in prompt  # fits well inside the scaled budget


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
            raise RuntimeError("no api.llm")
        monkeypatch.setattr(summary_mod, "_SummaryLLM", boom)
        assert summary_mod._safe_build_summary_llm("model") is None


# ============================================================================
# _SummaryLLM (wrapped api.llm.GenerateLLM)
# ============================================================================
class TestSummaryLLM:
    def test_generate_delegates_to_wrapped_llm(self):
        llm = summary_mod._SummaryLLM.__new__(summary_mod._SummaryLLM)

        class FakeGenerateLLM:
            async def generate(self, prompt: str) -> str:
                return "summary text"

        llm._llm = FakeGenerateLLM()
        result = asyncio.run(llm.generate("prompt"))
        assert result == "summary text"

    def test_generate_failure_returns_empty(self):
        llm = summary_mod._SummaryLLM.__new__(summary_mod._SummaryLLM)

        class FakeGenerateLLM:
            async def generate(self, prompt: str) -> str:
                # GenerateLLM contract: "" on failure, never raises.
                return ""

        llm._llm = FakeGenerateLLM()
        result = asyncio.run(llm.generate("prompt"))
        assert result == ""

    def test_generate_exception_propagates(self):
        """A hard error from the underlying GenerateLLM propagates; the caller
        (generate_product_summary) is responsible for catching them."""
        llm = summary_mod._SummaryLLM.__new__(summary_mod._SummaryLLM)

        class FakeGenerateLLM:
            async def generate(self, prompt: str) -> str:
                raise RuntimeError("boom")

        llm._llm = FakeGenerateLLM()
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(llm.generate("prompt"))


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

    def test_llm_output_secrets_masked(self, monkeypatch):
        """Deterministic secret guard: the persisted summary never carries a
        leaked token even when the model echoes one from the context."""
        product = FakeProduct()
        cbs = [FakeCodebase("app", "docs")]
        token = "ghp_" + "AB" * 15

        class FakeLLM:
            async def generate(self, prompt):
                return f"Summary leaks {token}"
        monkeypatch.setattr(summary_mod, "_safe_build_summary_llm", lambda *a, **kw: FakeLLM())
        result = asyncio.run(
            summary_mod.generate_product_summary(product, cbs, [], [])
        )
        assert token not in result
        assert "***REDACTED***" in result

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
