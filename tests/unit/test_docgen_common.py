"""Unit tests for api.docgen._common shared helpers.

Covers: set_main_event_loop / get_main_event_loop, _clean_llm_text,
_repo_name_from_url, _product_name, _StandardLLM (mocked generator),
_safe_build_llm, _llm_or_none, _make_repair_llm, _persist_artifact,
_cognee_dataset, _index_in_background, _with_verification_guard,
_resolve_docgen_model.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.docgen._common as c


# ============================================================================
# Event loop capture
# ============================================================================
class TestEventLoop:
    def test_set_and_get_main_loop(self):
        c.set_main_event_loop(None)
        assert c.get_main_event_loop() is None
        loop = asyncio.new_event_loop()
        try:
            c.set_main_event_loop(loop)
            assert c.get_main_event_loop() is loop
        finally:
            c.set_main_event_loop(None)
            loop.close()


# ============================================================================
# _with_verification_guard
# ============================================================================
class TestWithVerificationGuard:
    def test_empty_prompt_returns_unchanged(self):
        assert c._with_verification_guard("") == ""

    def test_none_prompt_returns_none(self):
        assert c._with_verification_guard(None) is None

    def test_appends_guard_when_available(self, monkeypatch):
        # The real function imports VERIFICATION_GUARD from api.prompts; test
        # that a non-empty guard is appended by calling the real function with
        # a stubbed import.
        import api.prompts as prompts_mod
        monkeypatch.setattr(prompts_mod, "VERIFICATION_GUARD", "GUARD_TEXT")
        result = c._with_verification_guard("prompt body")
        assert "prompt body" in result
        assert "GUARD_TEXT" in result

    def test_no_guard_leaves_prompt(self, monkeypatch):
        import api.prompts as prompts_mod
        monkeypatch.setattr(prompts_mod, "VERIFICATION_GUARD", "")
        assert c._with_verification_guard("prompt body") == "prompt body"


# ============================================================================
# _clean_llm_text
# ============================================================================
class TestCleanLlmText:
    def test_empty(self):
        assert c._clean_llm_text("") == ""
        assert c._clean_llm_text(None) == ""

    def test_strips_whitespace(self):
        assert c._clean_llm_text("  hello  ") == "hello"

    def test_stips_markdown_fence(self):
        text = "```markdown\n# Title\ncontent\n```"
        result = c._clean_llm_text(text)
        assert "```" not in result
        assert "# Title" in result
        assert "content" in result

    def test_strips_plain_fence(self):
        text = "```\ncode here\n```"
        result = c._clean_llm_text(text)
        assert result == "code here"

    def test_strips_fence_with_lang(self):
        text = "```python\nprint('hi')\n```"
        result = c._clean_llm_text(text)
        assert result == "print('hi')"

    def test_preserves_inner_content(self):
        text = "some intro\n\n```mermaid\nflowchart TD\n  A --> B\n```\n\ntail"
        result = c._clean_llm_text(text)
        assert "flowchart TD" in result
        assert "some intro" in result


# ============================================================================
# _repo_name_from_url
# ============================================================================
class TestRepoNameFromUrl:
    def test_github_url(self):
        assert c._repo_name_from_url("https://github.com/owner/repo") == "repo"

    def test_with_git_suffix(self):
        assert c._repo_name_from_url("https://github.com/owner/repo.git") == "repo"

    def test_trailing_slash(self):
        assert c._repo_name_from_url("https://github.com/owner/repo/") == "repo"

    def test_bare_name(self):
        assert c._repo_name_from_url("myrepo") == "myrepo"

    def test_empty_returns_original(self):
        assert c._repo_name_from_url("") == ""


# ============================================================================
# _product_name
# ============================================================================
class TestProductName:
    def test_product_name_present(self):
        class P:
            name = "MyProduct"
        class A:
            repo_url = "https://github.com/o/repo"
        assert c._product_name(P(), A()) == "MyProduct"

    def test_falls_back_to_repo_url(self):
        class P:
            name = None
        class A:
            repo_url = "https://github.com/o/myrepo"
            name = "artifact"
        assert c._product_name(P(), A()) == "myrepo"

    def test_falls_back_to_artifact_name(self):
        class P:
            name = None
        class A:
            repo_url = None
            name = "myartifact"
        assert c._product_name(P(), A()) == "myartifact"

    def test_falls_back_to_product_string(self):
        class P:
            name = ""
        class A:
            repo_url = None
            name = ""
        assert c._product_name(P(), A()) == "product"

    def test_product_none(self):
        class A:
            repo_url = "https://github.com/o/therepo"
            name = "x"
        assert c._product_name(None, A()) == "therepo"


# ============================================================================
# _persist_artifact
# ============================================================================
class TestPersistArtifact:
    def test_writes_docs_and_pages(self):
        class FakeArtifact:
            generated_docs = None
            pages = None
        a = FakeArtifact()
        c._persist_artifact(a, "markdown text", {"page_1": {}})
        assert a.generated_docs == "markdown text"
        assert a.pages == {"page_1": {}}

    def test_persist_failure_is_non_fatal(self):
        class BrokenArtifact:
            @property
            def generated_docs(self):
                return None
            @generated_docs.setter
            def generated_docs(self, v):
                raise RuntimeError("cannot set")
        a = BrokenArtifact()
        # Must not raise
        c._persist_artifact(a, "md", {})


# ============================================================================
# _cognee_dataset
# ============================================================================
class TestCogneeDataset:
    def test_uses_product_id(self):
        class P:
            id = "prod_abc"
        assert c._cognee_dataset(P()) == "prod_prod_abc"

    def test_falls_back_to_product_id_attr(self):
        class P:
            id = None
            product_id = "prod_xyz"
        assert c._cognee_dataset(P()) == "prod_prod_xyz"

    def test_falls_back_to_unknown(self):
        class P:
            id = None
            product_id = None
        assert c._cognee_dataset(P()) == "prod_unknown"


# ============================================================================
# _resolve_docgen_model
# ============================================================================
class TestResolveDocgenModel:
    def test_returns_model_with_defaults(self, monkeypatch):
        def fake_cfg(task):
            return {"model": "qwen/custom", "base_url": None, "api_key": None}
        import api.config.abstraction as ab
        monkeypatch.setattr(ab, "get_task_config", fake_cfg)
        model, base_url, api_key = c._resolve_docgen_model(None)
        assert model == "qwen/custom"
        assert base_url is None
        assert api_key is None

    def test_explicit_model_wins(self, monkeypatch):
        def fake_cfg(task):
            return {"model": "qwen/custom", "base_url": None, "api_key": None}
        import api.config.abstraction as ab
        monkeypatch.setattr(ab, "get_task_config", fake_cfg)
        model, _, _ = c._resolve_docgen_model("my/model")
        assert model == "my/model"

    def test_fallback_on_exception(self, monkeypatch):
        import api.config.abstraction as ab
        def boom(task):
            raise RuntimeError("no db")
        monkeypatch.setattr(ab, "get_task_config", boom)
        model, base_url, api_key = c._resolve_docgen_model(None)
        assert model == "qwen/qwen3.6-27b"
        assert base_url is None
        assert api_key is None


# ============================================================================
# _StandardLLM / _safe_build_llm (mocked adalflow)
# ============================================================================
class TestStandardLLM:
    def test_safe_build_llm_returns_none_on_exception(self, monkeypatch):
        # Force the import inside _StandardLLM.__init__ to fail
        def boom(*a, **kw):
            raise RuntimeError("no adalflow")
        monkeypatch.setattr(c, "_StandardLLM", boom)
        assert c._safe_build_llm("model") is None

    def test_standard_llm_generate_success(self, monkeypatch):
        """Test the generate() retry logic with a mocked generator."""
        llm = c._StandardLLM.__new__(c._StandardLLM)

        class FakeResult:
            error = None
            data = "generated text"

        class FakeGenerator:
            def __call__(self, prompt_kwargs=None):
                return FakeResult()

        llm.generator = FakeGenerator()

        async def _run():
            return await llm.generate("prompt")
        result = asyncio.run(_run())
        assert result == "generated text"

    def test_standard_llm_generate_error_retries(self, monkeypatch):
        llm = c._StandardLLM.__new__(c._StandardLLM)
        call_count = {"n": 0}

        class FakeResult:
            error = Exception("429 rate limit")
            data = None
            response = None
            answer = None
            raw_response = None
            output = None

        class FakeGenerator:
            def __call__(self, prompt_kwargs=None):
                call_count["n"] += 1
                return FakeResult()

        llm.generator = FakeGenerator()

        async def _run():
            return await llm.generate("prompt")
        result = asyncio.run(_run())
        # Error path returns "" after exhausting retries (which include sleeps).
        assert result == ""

    def test_standard_llm_generate_exception_returns_empty(self, monkeypatch):
        llm = c._StandardLLM.__new__(c._StandardLLM)

        class FakeGenerator:
            def __call__(self, prompt_kwargs=None):
                raise RuntimeError("connection error")

        llm.generator = FakeGenerator()

        async def _run():
            return await llm.generate("prompt")
        result = asyncio.run(_run())
        assert result == ""


# ============================================================================
# _llm_or_none
# ============================================================================
class TestLlmOrNone:
    def test_empty_prompt_returns_empty(self):
        async def _run():
            return await c._llm_or_none("", "model")
        assert asyncio.run(_run()) == ""

    def test_llm_build_fails_returns_empty(self, monkeypatch):
        monkeypatch.setattr(c, "_safe_build_llm", lambda *a, **kw: None)
        async def _run():
            return await c._llm_or_none("prompt", "model")
        assert asyncio.run(_run()) == ""

    def test_llm_generate_returns_cleaned_text(self, monkeypatch):
        class FakeLLM:
            async def generate(self, prompt):
                return "```markdown\nresult text\n```"
        monkeypatch.setattr(c, "_safe_build_llm", lambda *a, **kw: FakeLLM())
        async def _run():
            return await c._llm_or_none("prompt", "model")
        result = asyncio.run(_run())
        assert result == "result text"


# ============================================================================
# _make_repair_llm
# ============================================================================
class TestMakeRepairLlm:
    def test_returns_none_when_no_llm_and_build_fails(self, monkeypatch):
        monkeypatch.setattr(c, "_safe_build_llm", lambda *a, **kw: None)
        assert c._make_repair_llm("model") is None

    def test_reuses_existing_llm(self):
        existing = object()
        call = c._make_repair_llm("model", existing=existing)
        assert call is not None

    def test_callable_invokes_llm(self):
        class FakeLLM:
            async def generate(self, prompt):
                return "repair result"
        llm = FakeLLM()
        call = c._make_repair_llm("model", existing=llm)
        assert call is not None
        result = asyncio.run(call("prompt"))
        assert result == "repair result"

    def test_callable_swallows_exception(self):
        class FakeLLM:
            async def generate(self, prompt):
                raise RuntimeError("boom")
        llm = FakeLLM()
        call = c._make_repair_llm("model", existing=llm)
        result = asyncio.run(call("prompt"))
        assert result == ""


# ============================================================================
# _index_in_background
# ============================================================================
class TestIndexInBackground:
    def test_no_running_loop_logs_and_skips(self):
        # Called outside any event loop — should not raise.
        c._index_in_background("content", "dataset_x")

    def test_schedules_on_running_loop(self):
        async def _run():
            called = {"v": False}

            async def fake_add(content, dataset_name=None):
                called["v"] = True
                return ["ok"]

            import api.cognee as cognee_mod
            cognee_mod.add_and_index_document = fake_add
            c._index_in_background("content", "dataset_x")
            # Give the create_task a chance to run
            await asyncio.sleep(0.05)
            assert called["v"] is True

        asyncio.run(_run())

    def test_schedules_on_main_loop(self):
        main_loop = asyncio.new_event_loop()
        c.set_main_event_loop(main_loop)
        try:
            called = {"v": False}

            async def fake_add(content, dataset_name=None):
                called["v"] = True
                return ["ok"]

            import api.cognee as cognee_mod
            cognee_mod.add_and_index_document = fake_add

            async def _main():
                c._index_in_background("content", "dataset_x")
                # Let the scheduled coroutine run on this loop
                await asyncio.sleep(0.05)
                assert called["v"] is True

            main_loop.run_until_complete(_main())
        finally:
            c.set_main_event_loop(None)
            main_loop.close()

    def test_index_failure_is_non_fatal(self):
        async def _run():
            async def boom(content, dataset_name=None):
                raise RuntimeError("cognee down")

            import api.cognee as cognee_mod
            cognee_mod.add_and_index_document = boom
            # Must not raise
            c._index_in_background("content", "dataset_x")
            await asyncio.sleep(0.05)

        asyncio.run(_run())
