"""Unit tests for api.docgen.codebase (codebase doc generation).

Covers: _count_tokens, _build_file_blocks, _build_codebase_blob,
_chunk_file_blocks, _build_file_tree, _build_file_analysis, _read_readme,
_resolve_use_rlm, _section_pages, _raise_if_all_sections_unavailable,
generate_codebase_docs (small repo standard-LLM path, large repo RLM path,
error paths), _generate_section_text (single + map-reduce), _attempt_rlm.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.docgen.codebase as cb
from api.docgen.wiki import WikiSectionType


# ============================================================================
# Fixtures
# ============================================================================
class FakeDoc:
    """Minimal stand-in for an adalflow Document."""
    def __init__(self, text, file_path):
        self.text = text
        self.meta_data = {"file_path": file_path}


@pytest.fixture
def fake_documents():
    return [
        FakeDoc("import os\nprint('hello')\n", "src/main.py"),
        FakeDoc("def test_main():\n    pass\n", "tests/test_main.py"),
        FakeDoc("# My App\nA test application.\n", "README.md"),
    ]


# ============================================================================
# _count_tokens
# ============================================================================
class TestCountTokens:
    def test_empty(self):
        assert cb._count_tokens("") == 0

    def test_non_empty(self):
        result = cb._count_tokens("hello world")
        assert result > 0

    def test_fallback_ratio(self, monkeypatch):
        # Force tiktoken import to fail so the len//4 fallback is used
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("no tiktoken")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.setattr(cb, "_TIKTOKEN_ENC", None)
        result = cb._count_tokens("hello world test")
        assert result == len("hello world test") // 4


# ============================================================================
# _build_file_blocks / _build_codebase_blob
# ============================================================================
class TestBuildFileBlocks:
    def test_empty_documents(self):
        assert cb._build_file_blocks([]) == []

    def test_builds_blocks(self, fake_documents):
        blocks = cb._build_file_blocks(fake_documents)
        assert len(blocks) == 3
        assert "### File: src/main.py" in blocks[0]
        assert "import os" in blocks[0]
        assert "### File: tests/test_main.py" in blocks[1]

    def test_skips_empty_text(self):
        docs = [FakeDoc("", "empty.py"), FakeDoc("   ", "ws.py"), FakeDoc("code", "real.py")]
        blocks = cb._build_file_blocks(docs)
        assert len(blocks) == 1
        assert "real.py" in blocks[0]

    def test_large_file_split_into_parts(self):
        # Use text with newlines so _split_large_file_into_parts can split on
        # line boundaries. A single line with no newlines can't be split.
        big_text = "\n".join([f"line {i}" for i in range(2000)])  # ~14k chars, many lines
        doc = FakeDoc(big_text, "big.py")
        blocks = cb._build_file_blocks([doc], max_file_chunk_tokens=100)
        assert len(blocks) > 1
        assert "Part 1 of" in blocks[0]
        assert "Part" in blocks[-1]

    def test_build_codebase_blob(self, fake_documents):
        blob = cb._build_codebase_blob(fake_documents)
        assert "src/main.py" in blob
        assert "import os" in blob
        assert "README.md" in blob


# ============================================================================
# _chunk_file_blocks
# ============================================================================
class TestChunkFileBlocks:
    def test_empty_blocks(self):
        assert cb._chunk_file_blocks([], 1000) == []

    def test_single_block(self):
        blocks = ["### File: a.py\n```\ncode\n```\n"]
        chunks = cb._chunk_file_blocks(blocks, 1000)
        assert len(chunks) == 1
        assert chunks[0] == blocks[0]

    def test_multiple_blocks_fit_one_chunk(self):
        blocks = ["block1", "block2", "block3"]
        chunks = cb._chunk_file_blocks(blocks, 10000)
        assert len(chunks) == 1

    def test_blocks_split_by_budget(self):
        blocks = ["block1_content", "block2_content", "block3_content"]
        # Very small budget forces splitting
        chunks = cb._chunk_file_blocks(blocks, 5)
        assert len(chunks) >= 2

    def test_oversize_block_becomes_own_chunk(self):
        big = "x" * 1000
        blocks = ["small1", big, "small2"]
        chunks = cb._chunk_file_blocks(blocks, 10)
        # The big block should be its own chunk
        assert len(chunks) >= 2

    def test_zero_budget_returns_single_chunk(self):
        blocks = ["a", "b", "c"]
        chunks = cb._chunk_file_blocks(blocks, 0)
        assert len(chunks) == 1


# ============================================================================
# _build_file_tree
# ============================================================================
class TestBuildFileTree:
    def test_empty(self):
        assert cb._build_file_tree([]) == ""

    def test_sorted_unique(self):
        paths = ["b.py", "a.py", "b.py", "c.py"]
        result = cb._build_file_tree(paths)
        lines = result.split("\n")
        assert lines == ["a.py", "b.py", "c.py"]

    def test_max_lines(self):
        paths = [f"file_{i}.py" for i in range(300)]
        result = cb._build_file_tree(paths, max_lines=50)
        assert len(result.split("\n")) == 50


# ============================================================================
# _build_file_analysis
# ============================================================================
class TestBuildFileAnalysis:
    def test_empty_documents(self):
        result = cb._build_file_analysis([])
        assert result["file_count"] == 0
        assert result["primary_language"] == "unknown"
        assert result["main_directories"] == []

    def test_analysis(self, fake_documents):
        result = cb._build_file_analysis(fake_documents)
        assert result["file_count"] == 3
        assert "src" in result["main_directories"]
        assert "tests" in result["main_directories"]
        assert "main.py" in result["main_files"]
        assert result["primary_language"] == "Python"
        assert result["api_endpoints"] == []
        assert result["databases"] == []
        assert result["entities"] == []

    def test_config_files_detected(self):
        docs = [
            FakeDoc("{}", "package.json"),
            FakeDoc("deps", "requirements.txt"),
            FakeDoc("code", "src/main.py"),
        ]
        result = cb._build_file_analysis(docs)
        assert "package.json" in result["config_files"]
        assert "requirements.txt" in result["config_files"]

    def test_cicd_files_detected(self):
        docs = [FakeDoc("ci", ".github/workflows/ci.yml")]
        result = cb._build_file_analysis(docs)
        assert any(".github/" in f for f in result["cicd_files"])

    def test_docker_files_detected(self):
        docs = [FakeDoc("FROM python", "Dockerfile")]
        result = cb._build_file_analysis(docs)
        assert "Dockerfile" in result["docker_files"]

    def test_modules_from_directories(self):
        docs = [
            FakeDoc("code", "src/auth/login.py"),
            FakeDoc("code", "src/users/profile.py"),
        ]
        result = cb._build_file_analysis(docs)
        assert "src" in result["modules"]


# ============================================================================
# _read_readme
# ============================================================================
class TestReadReadme:
    def test_no_readme(self, tmp_path):
        assert cb._read_readme(str(tmp_path)) == ""

    def test_readme_md(self, tmp_path):
        p = tmp_path / "README.md"
        p.write_text("# My App\nDescription here.")
        assert "My App" in cb._read_readme(str(tmp_path))

    def test_readme_lowercase(self, tmp_path):
        p = tmp_path / "readme.md"
        p.write_text("# Lowercase")
        assert "Lowercase" in cb._read_readme(str(tmp_path))

    def test_readme_txt(self, tmp_path):
        p = tmp_path / "README.txt"
        p.write_text("Plain text readme")
        assert "Plain text readme" in cb._read_readme(str(tmp_path))


# ============================================================================
# _resolve_use_rlm
# ============================================================================
class TestResolveUseRlm:
    def test_explicit_true(self):
        assert cb._resolve_use_rlm(True, 100) is True

    def test_explicit_false(self):
        assert cb._resolve_use_rlm(False, 100000) is False

    def test_auto_small_blob(self, monkeypatch):
        import api.config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_rlm_mode", lambda task: "auto")
        assert cb._resolve_use_rlm(None, 1000) is False

    def test_auto_large_blob(self, monkeypatch):
        import api.config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_rlm_mode", lambda task: "auto")
        assert cb._resolve_use_rlm(None, cb.RLM_MIN_CHARS + 1) is True

    def test_mode_llm(self, monkeypatch):
        import api.config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_rlm_mode", lambda task: "llm")
        assert cb._resolve_use_rlm(None, 100000) is False

    def test_mode_rlm(self, monkeypatch):
        import api.config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_rlm_mode", lambda task: "rlm")
        assert cb._resolve_use_rlm(None, 100) is True

    def test_mode_auto_at_threshold(self, monkeypatch):
        import api.config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "get_rlm_mode", lambda task: "auto")
        assert cb._resolve_use_rlm(None, cb.RLM_MIN_CHARS) is True


# ============================================================================
# _section_pages
# ============================================================================
class TestSectionPages:
    def test_builds_pages_for_all_sections(self):
        sections = {st.value: f"content for {st.value}" for st in WikiSectionType}
        pages = cb._section_pages(sections, "ru")
        assert len(pages) == 7
        for st in WikiSectionType:
            page_id = f"page_{st.value}"
            assert page_id in pages
            assert pages[page_id]["content"] == f"content for {st.value}"
            assert pages[page_id]["title"]
            assert pages[page_id]["filePaths"] == []
            assert pages[page_id]["importance"] == "high"

    def test_empty_sections(self):
        pages = cb._section_pages({}, "ru")
        assert len(pages) == 7
        for page in pages.values():
            assert page["content"] == ""


# ============================================================================
# _raise_if_all_sections_unavailable
# ============================================================================
class TestRaiseIfAllUnavailable:
    def test_all_placeholder_raises(self):
        sections = {st.value: cb._SECTION_UNAVAILABLE_PLACEHOLDER for st in WikiSectionType}
        with pytest.raises(ValueError, match="Не удалось сгенерировать"):
            cb._raise_if_all_sections_unavailable(sections)

    def test_some_real_content_ok(self):
        sections = {st.value: cb._SECTION_UNAVAILABLE_PLACEHOLDER for st in WikiSectionType}
        sections["overview"] = "Real content"
        # Should not raise
        cb._raise_if_all_sections_unavailable(sections)

    def test_empty_dict_ok(self):
        cb._raise_if_all_sections_unavailable({})

    def test_whitespace_placeholder_raises(self):
        sections = {st.value: "  " + cb._SECTION_UNAVAILABLE_PLACEHOLDER + "  " for st in WikiSectionType}
        with pytest.raises(ValueError):
            cb._raise_if_all_sections_unavailable(sections)


# ============================================================================
# _attempt_rlm
# ============================================================================
class TestAttemptRlm:
    def test_circuit_breaker_trips(self):
        """When failures >= RLM_MAX_FAILURES, returns None immediately."""
        rlm_state = {"failures": cb.RLM_MAX_FAILURES}
        result = asyncio.run(cb._attempt_rlm("query", "model", rlm_state))
        assert result is None

    def test_rlm_success(self, monkeypatch):
        async def fake_run_rlm(query, model):
            return {"success": True, "results": "RLM generated text"}
        import api.rlm.runner as runner_mod
        monkeypatch.setattr(runner_mod, "run_rlm_task", fake_run_rlm)
        # Also patch the lazy import path
        import sys
        if "api.rlm.runner" in sys.modules:
            monkeypatch.setattr(sys.modules["api.rlm.runner"], "run_rlm_task", fake_run_rlm)

        result = asyncio.run(cb._attempt_rlm("query", "model", {"failures": 0}))
        assert "RLM generated text" in result

    def test_rlm_no_results(self, monkeypatch):
        async def fake_run_rlm(query, model):
            return {"success": True, "results": ""}
        import sys
        monkeypatch.setattr(sys.modules["api.rlm.runner"], "run_rlm_task", fake_run_rlm)

        rlm_state = {"failures": 0}
        result = asyncio.run(cb._attempt_rlm("query", "model", rlm_state))
        assert result is None
        assert rlm_state["failures"] == 1

    def test_rlm_failure_increments_state(self, monkeypatch):
        async def fake_run_rlm(query, model):
            return {"success": False}
        import sys
        monkeypatch.setattr(sys.modules["api.rlm.runner"], "run_rlm_task", fake_run_rlm)

        rlm_state = {"failures": 0}
        result = asyncio.run(cb._attempt_rlm("query", "model", rlm_state))
        assert result is None
        assert rlm_state["failures"] == 1


# ============================================================================
# _generate_section_text
# ============================================================================
class TestGenerateSectionText:
    def test_no_chunks_no_llm_returns_empty(self):
        async def _run():
            return await cb._generate_section_text("prompt", [], False, None, "model")
        assert asyncio.run(_run()) == ""

    def test_standard_llm_path(self, monkeypatch):
        class FakeLLM:
            async def generate(self, prompt):
                return "Section content from LLM"
        llm = FakeLLM()
        async def _run():
            return await cb._generate_section_text(
                "section prompt", ["codebase blob"], False, llm, "model"
            )
        result = asyncio.run(_run())
        assert "Section content from LLM" in result

    def test_rlm_single_chunk_success(self, monkeypatch):
        async def fake_run_rlm(query, model):
            return {"success": True, "results": "RLM section text"}
        import sys
        monkeypatch.setattr(sys.modules["api.rlm.runner"], "run_rlm_task", fake_run_rlm)

        class FakeLLM:
            async def generate(self, prompt):
                return "LLM fallback"
        async def _run():
            return await cb._generate_section_text(
                "prompt", ["chunk"], True, FakeLLM(), "model"
            )
        result = asyncio.run(_run())
        assert "RLM section text" in result

    def test_rlm_fallback_to_llm(self, monkeypatch):
        async def fake_run_rlm(query, model):
            return {"success": False}
        import sys
        monkeypatch.setattr(sys.modules["api.rlm.runner"], "run_rlm_task", fake_run_rlm)

        class FakeLLM:
            async def generate(self, prompt):
                return "LLM fallback text"
        async def _run():
            return await cb._generate_section_text(
                "prompt", ["chunk"], True, FakeLLM(), "model"
            )
        result = asyncio.run(_run())
        assert "LLM fallback text" in result


# ============================================================================
# generate_codebase_docs
# ============================================================================
class TestGenerateCodebaseDocs:
    @pytest.fixture
    def fake_repo_dir(self, tmp_path):
        """Create a temp repo dir with a couple of source files."""
        (tmp_path / "main.py").write_text("import os\nprint('hello')\n")
        (tmp_path / "README.md").write_text("# Test Repo\nA test.\n")
        return str(tmp_path)

    @pytest.fixture
    def fake_artifact(self):
        class A:
            repo_url = "https://github.com/owner/testrepo"
            repo_type = "github"
            token = None
            name = "testrepo"
            generated_docs = None
            pages = None
        return A()

    @pytest.fixture
    def fake_product(self):
        class P:
            id = "prod_123"
        return P()

    def test_no_repo_url_raises(self, fake_product):
        class A:
            repo_url = ""
        with pytest.raises(ValueError, match="no repo_url"):
            asyncio.run(cb.generate_codebase_docs(A(), fake_product))

    def test_whitespce_repo_url_raises(self, fake_product):
        class A:
            repo_url = "   "
        with pytest.raises(ValueError, match="no repo_url"):
            asyncio.run(cb.generate_codebase_docs(A(), fake_product))

    def test_small_repo_standard_llm_path(self, fake_artifact, fake_product, fake_repo_dir, monkeypatch):
        """Small repo (<20k chars) -> standard LLM path (no RLM)."""
        # Mock DatabaseManager._create_repo to set repo_paths
        class FakeDBManager:
            repo_paths = {"save_repo_dir": fake_repo_dir}
            def _create_repo(self, *a, **kw):
                pass

        import api.repositories.documents as docs_mod
        monkeypatch.setattr(docs_mod, "DatabaseManager", FakeDBManager)

        # Mock read_all_documents to return our fake docs
        def fake_read(repo_dir, **kw):
            return [
                FakeDoc("import os\nprint('hello')\n", "main.py"),
                FakeDoc("# Test Repo\nA test.\n", "README.md"),
            ]
        monkeypatch.setattr(docs_mod, "read_all_documents", fake_read)

        # Mock LLM
        class FakeLLM:
            async def generate(self, prompt):
                return f"Generated section content for prompt"
        monkeypatch.setattr(cb, "_safe_build_llm", lambda *a, **kw: FakeLLM())

        # Mock repair loop
        async def fake_repair(content, llm):
            return content, {}
        monkeypatch.setattr(cb, "run_repair_loop", fake_repair)

        # Mock indexing
        monkeypatch.setattr(cb, "_index_in_background", lambda *a, **kw: None)

        # Mock product knowledge retrieval (skip)
        import api.expert.knowledge as knowledge_mod
        async def fake_knowledge(pid, query):
            return ""
        monkeypatch.setattr(knowledge_mod, "_retrieve_product_knowledge", fake_knowledge)

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))
        assert "Документация по кодовой базе" in result
        assert fake_artifact.generated_docs is not None
        assert fake_artifact.pages is not None
        assert len(fake_artifact.pages) == 7

    def test_no_documents_raises(self, fake_artifact, fake_product, fake_repo_dir, monkeypatch):
        class FakeDBManager:
            repo_paths = {"save_repo_dir": fake_repo_dir}
            def _create_repo(self, *a, **kw):
                pass

        import api.repositories.documents as docs_mod
        monkeypatch.setattr(docs_mod, "DatabaseManager", FakeDBManager)
        monkeypatch.setattr(docs_mod, "read_all_documents", lambda *a, **kw: [])

        with pytest.raises(ValueError, match="No readable source files"):
            asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

    def test_repo_dir_not_found_raises(self, fake_artifact, fake_product, monkeypatch):
        class FakeDBManager:
            repo_paths = {"save_repo_dir": "/nonexistent/path"}
            def _create_repo(self, *a, **kw):
                pass

        import api.repositories.documents as docs_mod
        monkeypatch.setattr(docs_mod, "DatabaseManager", FakeDBManager)

        with pytest.raises(ValueError, match="Repository not available"):
            asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

    def test_all_sections_unavailable_raises(self, fake_artifact, fake_product, fake_repo_dir, monkeypatch):
        """When LLM returns nothing for all sections -> ValueError."""
        class FakeDBManager:
            repo_paths = {"save_repo_dir": fake_repo_dir}
            def _create_repo(self, *a, **kw):
                pass

        import api.repositories.documents as docs_mod
        monkeypatch.setattr(docs_mod, "DatabaseManager", FakeDBManager)
        monkeypatch.setattr(docs_mod, "read_all_documents", lambda *a, **kw: [
            FakeDoc("code", "main.py"),
        ])

        # LLM returns empty
        monkeypatch.setattr(cb, "_safe_build_llm", lambda *a, **kw: None)
        async def fake_repair(content, llm):
            return content, {}
        monkeypatch.setattr(cb, "run_repair_loop", fake_repair)
        monkeypatch.setattr(cb, "_index_in_background", lambda *a, **kw: None)
        import api.expert.knowledge as knowledge_mod
        async def fake_knowledge(pid, query):
            return ""
        monkeypatch.setattr(knowledge_mod, "_retrieve_product_knowledge", fake_knowledge)

        with pytest.raises(ValueError, match="Не удалось сгенерировать"):
            asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))


# ============================================================================
# _resolve_codebase_chunk_budget
# ============================================================================
class TestResolveChunkBudget:
    def test_returns_positive_budget(self):
        budget = cb._resolve_codebase_chunk_budget()
        assert budget >= 3000

    def test_respects_max_prompt_env(self, monkeypatch):
        monkeypatch.setenv("RLM_MAX_PROMPT_TOKENS", "10000")
        budget = cb._resolve_codebase_chunk_budget()
        assert budget >= 3000


# ============================================================================
# _split_large_file_into_parts
# ============================================================================
class TestSplitLargeFile:
    def test_small_file_single_part(self):
        parts = cb._split_large_file_into_parts("main.py", "short code", 10000)
        assert len(parts) == 1
        assert "Part" not in parts[0]

    def test_large_file_multiple_parts(self):
        big_text = "\n".join(f"line {i}" for i in range(1000))
        parts = cb._split_large_file_into_parts("big.py", big_text, 50)
        assert len(parts) > 1
        assert "Part 1 of" in parts[0]


# ============================================================================
# _generate_section_mapreduce
# ============================================================================
class TestGenerateSectionMapReduce:
    def test_multi_chunk_rlm_success_and_reduce(self, monkeypatch):
        """Multi-chunk path: RLM produces drafts, LLM reduces them."""
        async def fake_run_rlm(query, model):
            return {"success": True, "results": "draft text"}
        import sys
        monkeypatch.setattr(sys.modules["api.rlm.runner"], "run_rlm_task", fake_run_rlm)

        class FakeLLM:
            async def generate(self, prompt):
                return "merged section"
        async def _run():
            return await cb._generate_section_text(
                "prompt", ["chunk1", "chunk2"], True, FakeLLM(), "model"
            )
        result = asyncio.run(_run())
        assert "merged section" in result

    def test_multi_chunk_rlm_fails_agentic_fallback(self, monkeypatch):
        """RLM fails on all chunks -> agentic bottom-up path."""
        async def fake_run_rlm(query, model):
            return {"success": False}
        import sys
        monkeypatch.setattr(sys.modules["api.rlm.runner"], "run_rlm_task", fake_run_rlm)

        class FakeLLM:
            async def generate(self, prompt):
                return "agentic section content"
        async def _run():
            return await cb._generate_section_text(
                "prompt", ["chunk1", "chunk2"], True, FakeLLM(), "model"
            )
        result = asyncio.run(_run())
        assert "agentic section content" in result

    def test_multi_chunk_no_llm_returns_empty(self, monkeypatch):
        async def fake_run_rlm(query, model):
            return {"success": False}
        import sys
        monkeypatch.setattr(sys.modules["api.rlm.runner"], "run_rlm_task", fake_run_rlm)
        async def _run():
            return await cb._generate_section_text(
                "prompt", ["c1", "c2"], True, None, "model"
            )
        assert asyncio.run(_run()) == ""

    def test_multi_chunk_reduce_fails_returns_drafts(self, monkeypatch):
        """When reduce LLM call fails, concatenated drafts are returned."""
        async def fake_run_rlm(query, model):
            return {"success": True, "results": "draft content"}
        import sys
        monkeypatch.setattr(sys.modules["api.rlm.runner"], "run_rlm_task", fake_run_rlm)

        class FakeLLM:
            async def generate(self, prompt):
                raise RuntimeError("LLM down")
        async def _run():
            return await cb._generate_section_text(
                "prompt", ["c1", "c2"], True, FakeLLM(), "model"
            )
        result = asyncio.run(_run())
        assert "draft content" in result


# ============================================================================
# _reduce_section_drafts
# ============================================================================
class TestReduceSectionDrafts:
    def test_empty_drafts_returns_empty(self):
        assert asyncio.run(cb._reduce_section_drafts("p", [], None)) == ""

    def test_no_llm_returns_joined(self):
        result = asyncio.run(cb._reduce_section_drafts("p", ["d1", "d2"], None))
        assert "d1" in result
        assert "d2" in result

    def test_single_draft_returned_directly(self):
        assert asyncio.run(cb._reduce_section_drafts("p", ["only"], None)) == "only"

    def test_multiple_drafts_merged(self):
        class FakeLLM:
            async def generate(self, prompt):
                return "merged"
        result = asyncio.run(cb._reduce_section_drafts("p", ["d1", "d2"], FakeLLM()))
        assert result == "merged"

    def test_llm_raises_returns_empty(self):
        class FakeLLM:
            async def generate(self, prompt):
                raise RuntimeError("boom")
        result = asyncio.run(cb._reduce_section_drafts("p", ["d1", "d2"], FakeLLM()))
        assert result == ""


# ============================================================================
# _agentic_file_map_summary
# ============================================================================
class TestAgenticFileMapSummary:
    def test_no_llm_returns_empty(self):
        assert asyncio.run(cb._agentic_file_map_summary("chunk", None, 1000)) == ""

    def test_empty_chunk_returns_empty(self):
        class FakeLLM:
            async def generate(self, prompt):
                return "summary"
        assert asyncio.run(cb._agentic_file_map_summary("", FakeLLM(), 1000)) == ""

    def test_summary_produced(self):
        class FakeLLM:
            async def generate(self, prompt):
                return "file summary text"
        result = asyncio.run(cb._agentic_file_map_summary("code chunk", FakeLLM(), 1000))
        assert "file summary text" in result

    def test_llm_raises_returns_empty(self):
        class FakeLLM:
            async def generate(self, prompt):
                raise RuntimeError("err")
        result = asyncio.run(cb._agentic_file_map_summary("chunk", FakeLLM(), 1000))
        assert result == ""


# ============================================================================
# _agentic_bottom_up_docgen
# ============================================================================
class TestAgenticBottomUpDocgen:
    def test_no_llm_returns_empty(self):
        assert asyncio.run(cb._agentic_bottom_up_docgen("p", ["c1"], None)) == ""

    def test_no_chunks_returns_empty(self):
        class FakeLLM:
            async def generate(self, prompt):
                return "x"
        assert asyncio.run(cb._agentic_bottom_up_docgen("p", [], FakeLLM())) == ""

    def test_map_and_reduce_success(self):
        class FakeLLM:
            async def generate(self, prompt):
                if "<codebase_chunk>" in prompt:
                    return "file summary"
                return "final section"
        result = asyncio.run(cb._agentic_bottom_up_docgen("p", ["c1", "c2"], FakeLLM()))
        assert "final section" in result

    def test_map_empty_falls_back_to_direct_prompt(self):
        class FakeLLM:
            async def generate(self, prompt):
                if "<codebase_chunk>" in prompt:
                    return ""
                return "direct result"
        result = asyncio.run(cb._agentic_bottom_up_docgen("p", ["c1"], FakeLLM()))
        assert "direct result" in result


# ============================================================================
# generate_codebase_docs — product knowledge enrichment path
# ============================================================================
class TestGenerateCodebaseProductKnowledge:
    @pytest.fixture
    def fake_repo_dir(self, tmp_path):
        (tmp_path / "main.py").write_text("import os\nprint('hello')\n")
        (tmp_path / "README.md").write_text("# Test Repo\nA test.\n")
        return str(tmp_path)

    @pytest.fixture
    def fake_artifact(self):
        class A:
            repo_url = "https://github.com/owner/testrepo"
            repo_type = "github"
            token = None
            name = "testrepo"
            generated_docs = None
            pages = None
        return A()

    @pytest.fixture
    def fake_product(self):
        class P:
            id = "prod_kn"
        return P()

    def test_product_knowledge_enriches_readme(self, fake_artifact, fake_product, fake_repo_dir, monkeypatch):
        """When _retrieve_product_knowledge returns content, it's appended to readme."""
        class FakeDBManager:
            repo_paths = {"save_repo_dir": fake_repo_dir}
            def _create_repo(self, *a, **kw):
                pass

        import api.repositories.documents as docs_mod
        monkeypatch.setattr(docs_mod, "DatabaseManager", FakeDBManager)
        monkeypatch.setattr(docs_mod, "read_all_documents", lambda *a, **kw: [
            FakeDoc("import os\nprint('hello')\n", "main.py"),
            FakeDoc("# Test Repo\nA test.\n", "README.md"),
        ])

        class FakeLLM:
            async def generate(self, prompt):
                return "section content"
        monkeypatch.setattr(cb, "_safe_build_llm", lambda *a, **kw: FakeLLM())
        async def fake_repair(content, llm):
            return content, {}
        monkeypatch.setattr(cb, "run_repair_loop", fake_repair)
        monkeypatch.setattr(cb, "_index_in_background", lambda *a, **kw: None)

        import api.expert.knowledge as knowledge_mod
        async def fake_knowledge(pid, query):
            return "Confluence context here"
        monkeypatch.setattr(knowledge_mod, "_retrieve_product_knowledge", fake_knowledge)

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))
        assert "Документация по кодовой базе" in result

    def test_product_knowledge_retrieval_fails_non_fatal(self, fake_artifact, fake_product, fake_repo_dir, monkeypatch):
        """When _retrieve_product_knowledge raises, generation still succeeds."""
        class FakeDBManager:
            repo_paths = {"save_repo_dir": fake_repo_dir}
            def _create_repo(self, *a, **kw):
                pass

        import api.repositories.documents as docs_mod
        monkeypatch.setattr(docs_mod, "DatabaseManager", FakeDBManager)
        monkeypatch.setattr(docs_mod, "read_all_documents", lambda *a, **kw: [
            FakeDoc("import os\n", "main.py"),
        ])

        class FakeLLM:
            async def generate(self, prompt):
                return "section content"
        monkeypatch.setattr(cb, "_safe_build_llm", lambda *a, **kw: FakeLLM())
        async def fake_repair(content, llm):
            return content, {}
        monkeypatch.setattr(cb, "run_repair_loop", fake_repair)
        monkeypatch.setattr(cb, "_index_in_background", lambda *a, **kw: None)

        import api.expert.knowledge as knowledge_mod
        async def fake_knowledge(pid, query):
            raise RuntimeError("knowledge retrieval failed")
        monkeypatch.setattr(knowledge_mod, "_retrieve_product_knowledge", fake_knowledge)

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))
        assert "Документация по кодовой базе" in result
