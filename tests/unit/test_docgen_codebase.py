"""Unit tests for api.docgen.codebase (codebase doc generation).

Covers: _count_tokens, _build_file_blocks, _build_codebase_blob,
_chunk_file_blocks, _build_file_tree, _build_file_analysis, _read_readme,
_section_pages, _raise_if_all_sections_unavailable,
generate_codebase_docs (standard-LLM path, error paths, orchestrated
subagent path, diff regeneration), the Wave-F scaffolding (repo brief,
router hints, section contracts, progress tracker, task-tool handler,
router JSON parsing, extraction), _generate_section_text (single call +
agentic map-reduce), _reduce_section_drafts, _agentic_file_map_summary,
_agentic_bottom_up_docgen, and a real deepagents graph smoke test.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Any, Dict, List

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

import api.docgen.codebase as cb


# ============================================================================
# Fixtures
# ============================================================================
class FakeDoc:
    """Minimal stand-in for a repositories.documents.Document."""
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

    def test_default_is_len_div_4_heuristic(self, monkeypatch):
        # P1-23: precise counting is OPT-IN; default is the cheap heuristic.
        monkeypatch.delenv("LLM_PRECISE_TOKENS", raising=False)
        import api.utils.llm_tokens as lt

        monkeypatch.setattr(lt, "_ENCODER", None)
        monkeypatch.setattr(lt, "_ENCODER_LOADED", False)
        result = cb._count_tokens("hello world test")
        assert result == len("hello world test") // 4

    def test_delegates_to_shared_counter(self, monkeypatch):
        seen = {}

        def _spy(text, precise=None):
            seen["text"] = text
            seen["precise"] = precise
            return 42

        import api.utils.llm_tokens as lt

        monkeypatch.setattr(lt, "count_tokens", _spy)
        assert cb._count_tokens("abc") == 42
        assert seen == {"text": "abc", "precise": None}

    def test_precise_opt_in_via_env(self, monkeypatch):
        """LLM_PRECISE_TOKENS=1 switches the shared counter to tiktoken."""
        monkeypatch.setenv("LLM_PRECISE_TOKENS", "1")
        import api.utils.llm_tokens as lt

        class _FakeEnc:
            def encode(self, text, disallowed_special=()):
                return [0] * 7  # deterministic "exact" count

        monkeypatch.setattr(lt, "_ENCODER", _FakeEnc())
        monkeypatch.setattr(lt, "_ENCODER_LOADED", True)
        assert cb._count_tokens("anything") == 7


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
        chunks = cb._chunk_file_blocks(blocks, 5)
        assert len(chunks) >= 2

    def test_oversize_block_becomes_own_chunk(self):
        big = "x" * 1000
        blocks = ["small1", big, "small2"]
        chunks = cb._chunk_file_blocks(blocks, 10)
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
        assert "Plain text" in cb._read_readme(str(tmp_path))


# ============================================================================
# _section_pages
# ============================================================================
class TestSectionPages:
    def test_builds_pages_for_all_sections(self):
        sections = {sid: f"content for {sid}" for sid in cb.SECTION_ORDER}
        pages = cb._section_pages(sections, "ru")
        assert len(pages) == 7
        for sid in cb.SECTION_ORDER:
            page_id = f"page_{sid}"
            assert page_id in pages
            assert pages[page_id]["content"] == f"content for {sid}"
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
        sections = {sid: cb._SECTION_UNAVAILABLE_PLACEHOLDER for sid in cb.SECTION_ORDER}
        with pytest.raises(ValueError, match="Не удалось сгенерировать"):
            cb._raise_if_all_sections_unavailable(sections)

    def test_some_real_content_ok(self):
        sections = {sid: cb._SECTION_UNAVAILABLE_PLACEHOLDER for sid in cb.SECTION_ORDER}
        sections["overview"] = "Real content"
        cb._raise_if_all_sections_unavailable(sections)

    def test_empty_dict_ok(self):
        cb._raise_if_all_sections_unavailable({})

    def test_whitespace_placeholder_raises(self):
        sections = {
            sid: "  " + cb._SECTION_UNAVAILABLE_PLACEHOLDER + "  "
            for sid in cb.SECTION_ORDER
        }
        with pytest.raises(ValueError):
            cb._raise_if_all_sections_unavailable(sections)


# ============================================================================
# _generate_section_text
# ============================================================================
class TestGenerateSectionText:
    def test_no_chunks_no_llm_returns_empty(self):
        async def _run():
            return await cb._generate_section_text("prompt", [], None)
        assert asyncio.run(_run()) == ""

    def test_single_chunk_standard_llm_path(self, monkeypatch):
        class FakeLLM:
            async def generate(self, prompt):
                return "Section content from LLM"
        llm = FakeLLM()
        async def _run():
            return await cb._generate_section_text(
                "section prompt", ["codebase blob"], llm
            )
        result = asyncio.run(_run())
        assert "Section content from LLM" in result

    def test_single_chunk_no_llm_returns_empty(self):
        async def _run():
            return await cb._generate_section_text("prompt", ["chunk"], None)
        assert asyncio.run(_run()) == ""


# ============================================================================
# generate_codebase_docs (agent path OFF — standard LLM / error paths)
# ============================================================================
class TestGenerateCodebaseDocs:
    @pytest.fixture(autouse=True)
    def _hermetic_agent_off(self, monkeypatch):
        """Keep these tests on the standard-LLM seam: building the real
        deepagents orchestrator would construct a live ChatOpenAI client, and
        the judge stage would call a real LLM. The subagent-path tests patch
        the orchestrator/router seams explicitly."""
        monkeypatch.setattr(cb, "_deepagents_available", lambda: False)
        monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "false")

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
            id = "prod_123"
        return P()

    def test_no_repo_url_raises(self, fake_product):
        class A:
            repo_url = ""
        with pytest.raises(ValueError, match="no repo_url"):
            asyncio.run(cb.generate_codebase_docs(A(), fake_product))

    def test_whitespace_repo_url_raises(self, fake_product):
        class A:
            repo_url = "   "
        with pytest.raises(ValueError, match="no repo_url"):
            asyncio.run(cb.generate_codebase_docs(A(), fake_product))

    def test_small_repo_standard_llm_path(self, fake_artifact, fake_product, fake_repo_dir, monkeypatch):
        """Small repo (<20k chars) -> single-call standard LLM path."""
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
                return "Generated section content for prompt"
        monkeypatch.setattr(cb, "_safe_build_llm", lambda *a, **kw: FakeLLM())

        async def fake_repair(content, llm):
            return content, {}
        monkeypatch.setattr(cb, "run_repair_loop", fake_repair)
        monkeypatch.setattr(cb, "_index_in_background", lambda *a, **kw: None)

        import api.expert.knowledge as knowledge_mod
        async def fake_knowledge(pid, query):
            return ""
        monkeypatch.setattr(knowledge_mod, "_retrieve_product_knowledge", fake_knowledge)

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))
        assert "Документация по кодовой базе" in result
        assert fake_artifact.generated_docs is not None
        assert fake_artifact.pages is not None
        assert len(fake_artifact.pages) == 7
        # The standard-LLM fallback prompt is the writer rules + contract.
        page = fake_artifact.pages["page_overview"]
        assert page["provenance"]["generator"] == "standard-llm"
        assert page["provenance"]["regen"] == "legacy-fallback"
        assert page["provenance"]["prompt_file"] == "docgen_sections.md"

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

    def test_budget_is_context_bounded(self):
        budget = cb._resolve_codebase_chunk_budget()
        assert 3000 <= budget <= 200_000


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
# _generate_section_mapreduce (agentic bottom-up)
# ============================================================================
class TestGenerateSectionMapReduce:
    def test_multi_chunk_map_and_reduce(self, monkeypatch):
        class FakeLLM:
            async def generate(self, prompt):
                if "<codebase_chunk>" in prompt:
                    return "draft text"
                return "merged section"
        async def _run():
            return await cb._generate_section_text(
                "prompt", ["chunk1", "chunk2"], FakeLLM()
            )
        result = asyncio.run(_run())
        assert "merged section" in result

    def test_multi_chunk_no_llm_returns_empty(self, monkeypatch):
        async def _run():
            return await cb._generate_section_text(
                "prompt", ["c1", "c2"], None
            )
        assert asyncio.run(_run()) == ""

    def test_multi_chunk_reduce_fails_returns_drafts(self, monkeypatch):
        class FakeLLM:
            async def generate(self, prompt):
                if "<codebase_chunk>" in prompt:
                    return "draft content"
                raise RuntimeError("LLM down")
        async def _run():
            return await cb._generate_section_text(
                "prompt", ["c1", "c2"], FakeLLM()
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

    # --- P1-24: bounded parallel MAP phase ----------------------------------- #
    def test_map_phase_bounded_parallel_and_ordered(self, monkeypatch):
        """Phase 1 runs chunk summaries with bounded parallelism (semaphore)
        and gather preserves the part i/N order regardless of completion."""
        import re

        import api.config.timeout as timeout_mod

        monkeypatch.setattr(timeout_mod, "resolve_docgen_map_concurrency", lambda: 2)
        peak = {"cur": 0, "max": 0}

        class FakeLLM:
            async def generate(self, prompt):
                if "<codebase_chunk>" not in prompt:
                    return "joined"
                peak["cur"] += 1
                peak["max"] = max(peak["max"], peak["cur"])
                await asyncio.sleep(0.03)
                peak["cur"] -= 1
                return "file summary"

        async def _fake_reduce(section_prompt, drafts, llm):
            return "\n\n".join(drafts)

        monkeypatch.setattr(cb, "_reduce_section_drafts", _fake_reduce)

        chunks = [f"chunk-{i}" for i in range(6)]
        result = asyncio.run(cb._agentic_bottom_up_docgen("p", chunks, FakeLLM()))
        assert 2 <= peak["max"] <= 2  # parallelism used AND bounded at 2
        labels = re.findall(r"часть (\d+)/6", result)
        assert labels == [str(i) for i in range(1, 7)]  # order preserved

    def test_map_phase_concurrency_one_is_sequential(self, monkeypatch):
        import api.config.timeout as timeout_mod

        monkeypatch.setattr(timeout_mod, "resolve_docgen_map_concurrency", lambda: 1)
        peak = {"cur": 0, "max": 0}

        class FakeLLM:
            async def generate(self, prompt):
                if "<codebase_chunk>" not in prompt:
                    return "joined"
                peak["cur"] += 1
                peak["max"] = max(peak["max"], peak["cur"])
                await asyncio.sleep(0.01)
                peak["cur"] -= 1
                return "file summary"

        async def _fake_reduce(section_prompt, drafts, llm):
            return "\n\n".join(drafts)

        monkeypatch.setattr(cb, "_reduce_section_drafts", _fake_reduce)

        chunks = [f"chunk-{i}" for i in range(4)]
        result = asyncio.run(cb._agentic_bottom_up_docgen("p", chunks, FakeLLM()))
        assert peak["max"] == 1  # strictly sequential
        assert result.count("file summary") == 4


# ============================================================================
# generate_codebase_docs — product knowledge enrichment path
# ============================================================================
class TestGenerateCodebaseProductKnowledge:
    @pytest.fixture(autouse=True)
    def _hermetic_agent_off(self, monkeypatch):
        monkeypatch.setattr(cb, "_deepagents_available", lambda: False)
        monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "false")

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


# ============================================================================
# Repo tools (read-only, path-confined to the clone)
# ============================================================================
class TestConfinedPath:
    def test_inside_ok(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1\n")
        expected = os.path.realpath(os.path.join(str(tmp_path), "main.py"))
        assert cb._confined_path(str(tmp_path), "main.py") == expected

    def test_escape_rejected(self, tmp_path):
        assert cb._confined_path(str(tmp_path), "../outside.py") is None
        assert cb._confined_path(str(tmp_path), "/etc/passwd") is None

    def test_git_internals_rejected(self, tmp_path):
        assert cb._confined_path(str(tmp_path), ".git/config") is None

    def test_skip_dirs_rejected(self, tmp_path):
        assert cb._confined_path(str(tmp_path), "node_modules/pkg/index.js") is None
        assert cb._confined_path(str(tmp_path), "__pycache__/mod.py") is None

    def test_empty_inputs(self, tmp_path):
        assert cb._confined_path(str(tmp_path), "") is None
        assert cb._confined_path("", "a.py") is None

    def test_null_byte_rejected(self, tmp_path):
        assert cb._confined_path(str(tmp_path), "main.py\x00.py") is None


class TestRepoTools:
    def _repo(self, tmp_path):
        (tmp_path / "main.py").write_text("import os\nvalue = 42\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "util.py").write_text("def helper():\n    return 'util'\n")
        return str(tmp_path)

    def test_list_files(self, tmp_path):
        tools = cb.build_repo_tools(self._repo(tmp_path))
        listing = tools[0].invoke({})
        assert "main.py" in listing
        assert "src/util.py" in listing

    def test_read_file(self, tmp_path):
        tools = cb.build_repo_tools(self._repo(tmp_path))
        content = tools[1].invoke({"path": "src/util.py"})
        assert "def helper" in content

    def test_read_file_missing_error(self, tmp_path):
        tools = cb.build_repo_tools(self._repo(tmp_path))
        assert "ERROR" in tools[1].invoke({"path": "no/such.py"})

    def test_read_file_escape_error(self, tmp_path):
        tools = cb.build_repo_tools(self._repo(tmp_path))
        assert "ERROR" in tools[1].invoke({"path": "../etc/passwd"})

    def test_grep_finds_match(self, tmp_path):
        tools = cb.build_repo_tools(self._repo(tmp_path))
        out = tools[2].invoke({"pattern": "value = 42"})
        assert "main.py:2:" in out

    def test_grep_no_matches(self, tmp_path):
        tools = cb.build_repo_tools(self._repo(tmp_path))
        assert "no matches" in tools[2].invoke({"pattern": "zzz_not_there"})

    def test_grep_invalid_regex_error(self, tmp_path):
        tools = cb.build_repo_tools(self._repo(tmp_path))
        assert "ERROR" in tools[2].invoke({"pattern": "("})

    def test_symlinked_file_skipped(self, tmp_path):
        repo = self._repo(tmp_path)
        outside = tmp_path.parent / "outside_secret.py"
        outside.write_text("token = 'leak'\n")
        os.symlink(outside, tmp_path / "linked.py")
        tools = cb.build_repo_tools(repo)
        listing = tools[0].invoke({})
        assert "linked.py" not in listing
        grep_out = tools[2].invoke({"pattern": "leak"})
        assert "linked.py" not in grep_out

    def test_symlinked_dir_skipped(self, tmp_path):
        repo = self._repo(tmp_path)
        other = tmp_path.parent / "other_pkg"
        other.mkdir()
        (other / "x.py").write_text("sneaky = 1\n")
        os.symlink(other, tmp_path / "linked_dir")
        tools = cb.build_repo_tools(repo)
        listing = tools[0].invoke({})
        assert "linked_dir" not in listing
        grep_out = tools[2].invoke({"pattern": "sneaky"})
        assert "linked_dir" not in grep_out

    def test_grep_nested_quantifier_rejected_fast(self, tmp_path):
        tools = cb.build_repo_tools(self._repo(tmp_path))
        out = tools[2].invoke({"pattern": "(a+a+)+b"})
        assert out.startswith("ERROR")
        assert "nested quantifiers" in out

    def test_grep_overlong_pattern_rejected(self, tmp_path):
        tools = cb.build_repo_tools(self._repo(tmp_path))
        out = tools[2].invoke(
            {"pattern": "a" * (cb._REPO_GREP_PATTERN_MAX_CHARS + 1)}
        )
        assert out.startswith("ERROR")

    def test_grep_wall_clock_budget_exhausted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cb, "_REPO_GREP_WALL_CLOCK_SECONDS", 0.0)
        tools = cb.build_repo_tools(self._repo(tmp_path))
        out = tools[2].invoke({"pattern": "value"})
        assert "wall-clock budget exhausted" in out


class TestGrepRejectionReason:
    def test_static_rejections(self):
        assert cb._grep_rejection_reason("") == "empty pattern"
        assert "longer than" in cb._grep_rejection_reason("a" * 201)
        assert "nested quantifiers" in cb._grep_rejection_reason("(a+a+)+b")
        assert "invalid regex" in cb._grep_rejection_reason("(")

    def test_benign_pattern_accepted(self):
        assert cb._grep_rejection_reason("value = 42") is None


class TestRepoTreeHash:
    def test_changes_when_file_edited(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        h1 = cb._compute_repo_tree_hash(str(tmp_path))
        (tmp_path / "a.py").write_text("x = 2\n")
        h2 = cb._compute_repo_tree_hash(str(tmp_path))
        assert h1 and h2 and h1 != h2

    def test_changes_when_file_added(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        h1 = cb._compute_repo_tree_hash(str(tmp_path))
        (tmp_path / "b.py").write_text("y = 2\n")
        h2 = cb._compute_repo_tree_hash(str(tmp_path))
        assert h1 != h2

    def test_empty_listing_returns_none(self, tmp_path):
        assert cb._compute_repo_tree_hash(str(tmp_path / "missing")) is None


# ============================================================================
# Agent result helpers
# ============================================================================
class TestAgentHelpers:
    def _result(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        return {
            "messages": [
                HumanMessage(content="task"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "repo_read_file", "args": {"path": "main.py"}, "id": "c1"},
                        {"name": "repo_read_file", "args": {"path": "src/util.py"}, "id": "c2"},
                        {"name": "repo_grep", "args": {"pattern": "x"}, "id": "c3"},
                        {"name": "repo_read_file", "args": {"path": "main.py"}, "id": "c4"},
                    ],
                ),
                ToolMessage(content="code", tool_call_id="c1"),
                AIMessage(content="## Final section text"),
            ]
        }

    def test_final_agent_text(self):
        assert cb._final_agent_text(self._result()) == "## Final section text"

    def test_final_agent_text_object_result(self):
        messages = self._result()["messages"]

        class Obj:
            pass

        obj = Obj()
        obj.messages = messages
        assert cb._final_agent_text(obj) == "## Final section text"

    def test_final_agent_text_empty(self):
        assert cb._final_agent_text({}) == ""
        assert cb._final_agent_text({"messages": []}) == ""

    def test_agent_files_read_dedup_only_reads(self):
        assert cb._agent_files_read(self._result()) == ["main.py", "src/util.py"]

    def test_agent_files_read_filters_error_results(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        result = {"messages": [
            HumanMessage(content="task"),
            AIMessage(content="", tool_calls=[
                {"name": "repo_read_file", "args": {"path": "main.py"}, "id": "c1"},
                {"name": "repo_read_file", "args": {"path": "src/util.py"}, "id": "c2"},
            ]),
            ToolMessage(
                content="ERROR: file not found inside the repository: main.py",
                tool_call_id="c1",
            ),
            ToolMessage(content="code", tool_call_id="c2"),
        ]}
        assert cb._agent_files_read(result) == ["src/util.py"]

    def test_run_agent_section_never_raises(self):
        class _Boom:
            async def ainvoke(self, payload, config=None):
                raise RuntimeError("agent down")

        text, files = asyncio.run(cb._run_agent_section(_Boom(), "task"))
        assert text == "" and files == []

    def test_run_agent_section_returns_text_and_files(self):
        result = self._result()

        class _Ok:
            async def ainvoke(self, payload, config=None):
                return result

        text, files = asyncio.run(cb._run_agent_section(_Ok(), "task"))
        assert text == "## Final section text"
        assert files == ["main.py", "src/util.py"]


# ============================================================================
# Wave F scaffolding: router JSON parsing + hint routing
# ============================================================================
class TestParseRouterJson:
    def test_plain_json(self):
        raw = '{"overview": {"files": ["a.py"], "focus": "readme"}}'
        assert cb._parse_router_json(raw) == {
            "overview": {"files": ["a.py"], "focus": "readme"}
        }

    def test_fenced_json(self):
        raw = '```json\n{"overview": {"files": [], "focus": ""}}\n```'
        assert cb._parse_router_json(raw) == {"overview": {"files": [], "focus": ""}}

    def test_prose_around_json(self):
        raw = 'Here is the mapping:\n{"overview": {"files": ["a"]}}\nDone.'
        assert cb._parse_router_json(raw) == {"overview": {"files": ["a"]}}

    def test_invalid_returns_none(self):
        assert cb._parse_router_json("no json here") is None
        assert cb._parse_router_json("") is None
        assert cb._parse_router_json("{broken") is None

    def test_non_dict_returns_none(self):
        assert cb._parse_router_json('["a", "b"]') is None


class TestRouteSectionHints:
    def _chat(self, content):
        from langchain_core.messages import AIMessage

        class Chat:
            async def ainvoke(self, messages):
                return AIMessage(content=content)
        return Chat()

    def test_filters_to_expected_sids(self):
        raw = json.dumps({
            "overview": {"files": ["README.md"], "focus": "intro"},
            "not_a_section": {"files": ["x"], "focus": "junk"},
            "architecture": "junk-not-a-dict",
        })
        hints = asyncio.run(cb._route_section_hints(
            self._chat(raw), "brief", "sections", ["overview"],
        ))
        assert hints == {"overview": {"files": ["README.md"], "focus": "intro"}}

    def test_chat_raises_returns_empty(self):
        class Boom:
            async def ainvoke(self, messages):
                raise RuntimeError("llm down")
        assert asyncio.run(cb._route_section_hints(
            Boom(), "brief", "sections", ["overview"],
        )) == {}

    def test_unparsable_response_returns_empty(self):
        assert asyncio.run(cb._route_section_hints(
            self._chat("utter nonsense"), "brief", "sections", ["overview"],
        )) == {}


# ============================================================================
# Wave F scaffolding: progress tracker + task tool handler
# ============================================================================
def _collector():
    events: List[Dict[str, Any]] = []

    def progress(**fields):
        events.append(fields)
    return events, progress


class TestSectionProgressTracker:
    def test_started_and_finished_events(self):
        events, progress = _collector()
        tracker = cb._SectionProgressTracker(progress, sections_total=7)
        tracker.section_started("overview")
        tracker.section_finished("overview")
        assert events[0] == {
            "phase": "sections", "sections_total": 7, "current_section": "overview",
        }
        assert events[1]["section_done"] == "overview"
        assert events[1]["sections_total"] == 7
        assert events[1]["section_seconds"] >= 0.0
        assert tracker.already_reported("overview") is True
        assert tracker.already_reported("architecture") is False

    def test_finished_without_started_reports_zero_seconds(self):
        events, progress = _collector()
        tracker = cb._SectionProgressTracker(progress, sections_total=7)
        tracker.section_finished("cicd")
        assert events[-1]["section_seconds"] == 0.0

    def test_none_progress_is_noop(self):
        tracker = cb._SectionProgressTracker(None, sections_total=7)
        tracker.section_started("overview")  # must not raise
        tracker.section_finished("overview")

    def test_repeated_start_keeps_first_timestamp(self):
        events, progress = _collector()
        tracker = cb._SectionProgressTracker(progress, sections_total=7)
        tracker.section_started("overview")
        tracker.section_started("overview")
        # Only one current_section event per start call, but the recorded
        # start time never resets (setdefault semantics).
        assert len([e for e in events if "current_section" in e]) == 2


class TestTaskToolProgressHandler:
    def _handler(self):
        events, progress = _collector()
        tracker = cb._SectionProgressTracker(progress, sections_total=7)
        return cb._TaskToolProgressHandler(tracker), events

    def test_dict_input_with_run_id(self):
        handler, events = self._handler()
        asyncio.run(handler.on_tool_start(
            {}, {"subagent_type": "section-overview", "description": "go"},
            run_id="r1",
        ))
        asyncio.run(handler.on_tool_end("done", run_id="r1"))
        assert events[0]["current_section"] == "overview"
        assert events[1]["section_done"] == "overview"

    def test_json_string_input(self):
        handler, events = self._handler()
        asyncio.run(handler.on_tool_start(
            {}, json.dumps({"subagent_type": "section-qa"}), run_id="r1",
        ))
        assert events[0]["current_section"] == "qa"

    def test_non_section_tool_ignored(self):
        handler, events = self._handler()
        asyncio.run(handler.on_tool_start(
            {}, {"subagent_type": "general-purpose"}, run_id="r1",
        ))
        asyncio.run(handler.on_tool_end("out", run_id="r1"))
        assert events == []

    def test_anon_lifo_without_run_id(self):
        handler, events = self._handler()
        asyncio.run(handler.on_tool_start({}, {"subagent_type": "section-cicd"}))
        asyncio.run(handler.on_tool_start({}, {"subagent_type": "section-qa"}))
        asyncio.run(handler.on_tool_end("out"))  # pops qa (LIFO)
        assert events[-1]["section_done"] == "qa"
        asyncio.run(handler.on_tool_end("out"))  # pops cicd
        assert events[-1]["section_done"] == "cicd"

    def test_works_through_real_callback_dispatch(self):
        """Regression (adversarial review): the handler must survive
        langchain-core's real event dispatch, which filters handlers by the
        ``run_inline``/``ignore_*`` base-class attributes. A plain
        duck-typed handler raised AttributeError on the first event and the
        orchestrator except-clause silently degraded EVERY run to the
        python-parallel fallback."""
        import uuid

        from langchain_core.callbacks.manager import ahandle_event

        handler, events = self._handler()
        run_id = uuid.uuid4()

        async def dispatch():
            # Same shapes AsyncCallbackManager uses for tool events.
            await ahandle_event(
                [handler], "on_tool_start", "ignore_agent",
                {}, {"subagent_type": "section-overview", "description": "go"},
                run_id=run_id, parent_run_id=None, tags=None, metadata=None,
            )
            await ahandle_event(
                [handler], "on_tool_end", "ignore_agent",
                "done", run_id=run_id, parent_run_id=None, tags=None,
            )

        asyncio.run(dispatch())
        assert events[0]["current_section"] == "overview"
        assert events[-1]["section_done"] == "overview"


# ============================================================================
# Wave F scaffolding: orchestrator transcript extraction
# ============================================================================
class TestExtractOrchestratedSections:
    def _transcript(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        return {"messages": [
            HumanMessage(content="generate"),
            AIMessage(content="", tool_calls=[
                {
                    "name": "task",
                    "args": {"description": "write overview", "subagent_type": "section-overview"},
                    "id": "t1",
                },
                {
                    "name": "task",
                    "args": {"description": "write qa", "subagent_type": "section-qa"},
                    "id": "t2",
                },
                {
                    "name": "task",
                    # Unknown subagent type: mapping exists but result is an
                    # error string that must be skipped.
                    "args": {"description": "broken", "subagent_type": "section-cicd"},
                    "id": "t3",
                },
            ]),
            ToolMessage(content="## Overview body", tool_call_id="t1"),
            ToolMessage(content="## QA body", tool_call_id="t2"),
            ToolMessage(
                content="We cannot invoke subagent with name section-cicd",
                tool_call_id="t3",
            ),
            AIMessage(content="all sections dispatched"),
        ]}

    def test_extracts_sections(self):
        sections = cb._extract_orchestrated_sections(self._transcript())
        assert sections == {"overview": "## Overview body", "qa": "## QA body"}

    def test_empty_result(self):
        assert cb._extract_orchestrated_sections({}) == {}
        assert cb._extract_orchestrated_sections({"messages": []}) == {}


# ============================================================================
# Wave F scaffolding: prompts rendering
# ============================================================================
class TestScaffoldingRendering:
    def test_sections_list_text_all(self):
        text = cb._sections_list_text("ru")
        lines = text.split("\n")
        assert len(lines) == 7
        assert lines[0].startswith("- `overview` — Общая информация")

    def test_sections_list_text_only_filter(self):
        text = cb._sections_list_text("en", only=["overview", "qa"])
        lines = text.split("\n")
        assert len(lines) == 2
        assert "`overview`" in lines[0] and "Overview" in lines[0]

    def test_sections_list_text_empty(self):
        assert cb._sections_list_text("ru", only=[]) == "(none)"

    def test_writer_rules_carry_language_and_guard(self):
        from api.prompts import VERIFICATION_GUARD

        rules = cb._section_writer_rules("ru")
        # docgen_agent_system.md has its {language_name} slot substituted...
        assert "Russian" in rules
        assert "{language_name}" not in rules
        # ...and the verification guard is appended verbatim at the end.
        assert VERIFICATION_GUARD.strip() and rules.endswith(VERIFICATION_GUARD.strip())

    def test_render_section_hints_full(self):
        hint = {"files": ["a.py", "b.py"], "focus": "entry points"}
        text = cb._render_section_hints(hint)
        assert "`a.py`" in text and "`b.py`" in text
        assert "entry points" in text

    def test_render_section_hints_caps(self):
        hint = {
            "files": [f"f{i}.py" for i in range(20)] + ["x" * 500],
            "focus": "y" * 1000,
        }
        text = cb._render_section_hints(hint)
        # Max 10 files, each capped; focus capped.
        assert len([ln for ln in text.split("\n") if ".py" in ln or "`f" in ln]) <= 2
        assert "y" * 500 not in text

    def test_render_section_hints_none(self):
        assert "explore" in cb._render_section_hints(None)

    def test_build_section_contract_substitutes_all_slots(self):
        contract = cb._build_section_contract(
            repo_url="https://github.com/o/r",
            repo_name="r",
            sid="overview",
            title="Overview",
            repo_brief="BRIEF TEXT",
            sections_list="- `overview` — Overview",
            hints={"files": ["a.py"], "focus": "start"},
        )
        for placeholder in (
            "{repo_url}", "{repo_name}", "{section_id}", "{section_title}",
            "{repo_brief}", "{sections_list}", "{section_hints}",
            "{section_instruction}",
        ):
            assert placeholder not in contract, f"unsubstituted {placeholder}"
        assert "BRIEF TEXT" in contract
        assert "`a.py`" in contract

    def test_build_section_agent_system_prompt_joins(self):
        prompt = cb._build_section_agent_system_prompt("RULES", "CONTRACT")
        assert prompt.startswith("RULES")
        assert prompt.endswith("CONTRACT")
        assert "\n\n---\n\n" in prompt

    def test_build_repo_brief(self):
        brief = cb._build_repo_brief(
            repo_url="https://github.com/o/r",
            repo_type="github",
            file_analysis={
                "primary_language": "Python",
                "file_count": 42,
                "main_directories": ["src"],
                "config_files": ["pyproject.toml"],
                "cicd_files": [],
                "docker_files": ["Dockerfile"],
            },
            file_tree="src/main.py",
            readme="# Head",
        )
        assert "https://github.com/o/r" in brief
        assert "Python" in brief and "42" in brief
        assert "`src`" in brief
        assert "`Dockerfile`" in brief
        assert "# Head" in brief
        assert "src/main.py" in brief

    def test_build_repo_brief_caps_output(self):
        brief = cb._build_repo_brief(
            repo_url="u", repo_type="github",
            file_analysis={}, file_tree="line\n" * 5000, readme="r" * 20_000,
        )
        assert len(brief) <= cb._BRIEF_MAX_CHARS + 100  # cap + truncation marker

    def test_section_concurrency_default_and_env(self, monkeypatch):
        monkeypatch.delenv(cb._SECTION_CONCURRENCY_ENV, raising=False)
        assert cb._section_concurrency() == cb._SECTION_CONCURRENCY_DEFAULT
        monkeypatch.setenv(cb._SECTION_CONCURRENCY_ENV, "5")
        assert cb._section_concurrency() == 5
        monkeypatch.setenv(cb._SECTION_CONCURRENCY_ENV, "0")
        assert cb._section_concurrency() == 1  # clamped to >= 1
        monkeypatch.setenv(cb._SECTION_CONCURRENCY_ENV, "not-a-number")
        assert cb._section_concurrency() == cb._SECTION_CONCURRENCY_DEFAULT


# ============================================================================
# generate_codebase_docs over the orchestrated subagent path
# ============================================================================
class _FakeOrchestrator:
    """Stub deepagents orchestrator: ainvoke returns a scripted transcript
    with one `task` tool call + ToolMessage per section."""

    def __init__(self, sections: Dict[str, str], error: Exception = None):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        self.error = error
        self.calls = 0
        calls = [
            (f"t{i}", sid, text)
            for i, (sid, text) in enumerate(sections.items())
        ]
        messages: List[Any] = [HumanMessage(content="generate the wiki")]
        if calls:
            messages.append(AIMessage(content="", tool_calls=[
                {
                    "name": "task",
                    "args": {
                        "description": f"write {sid}",
                        "subagent_type": f"section-{sid}",
                    },
                    "id": call_id,
                }
                for call_id, sid, _ in calls
            ]))
            for call_id, sid, text in calls:
                messages.append(ToolMessage(content=text, tool_call_id=call_id))
        messages.append(AIMessage(content="orchestrator report: all sections done"))
        self._messages = messages

    async def ainvoke(self, payload, config=None):
        self.calls += 1
        if self.error:
            raise self.error
        return {"messages": self._messages}


class TestDocgenAgentFlow:
    """Full generate_codebase_docs runs with the subagent seams patched in.

    Everything that could touch the network is patched: the chat-model
    factory, the router, the parallel fallback, the judge flag, the repair
    loop, and memory indexing. ``verify_section`` runs FOR REAL
    (masking/citations/fingerprint are deterministic and local).
    """

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
            id = "prod_agent"
        return P()

    def _patch_flow(self, monkeypatch, fake_repo_dir, orchestrator=None,
                    parallel=None, builder=None):
        import api.repositories.documents as docs_mod

        class FakeDBManager:
            repo_paths = {"save_repo_dir": fake_repo_dir}
            def _create_repo(self, *a, **kw):
                pass

        monkeypatch.setattr(docs_mod, "DatabaseManager", FakeDBManager)
        monkeypatch.setattr(docs_mod, "read_all_documents", lambda *a, **kw: [
            FakeDoc("import os\nprint('hello')\n", "main.py"),
            FakeDoc("# Test Repo\nA test.\n", "README.md"),
        ])
        # Never construct a real ChatOpenAI for the agent path.
        import api.llm.client as llm_client_mod
        monkeypatch.setattr(llm_client_mod, "build_chat_model", lambda **kw: object())
        monkeypatch.setattr(cb, "_deepagents_available", lambda: True)

        async def fake_route(chat, brief, sections_list, expected):
            return {}
        monkeypatch.setattr(cb, "_route_section_hints", fake_route)

        # Units restructure seams: no decomposer LLM call by default (tests
        # that need children re-patch it) and NO notes workspace (prevents
        # writes into the real ~/.productarium state dir).
        async def fake_decompose(chat, brief, sections_list, hints):
            return {}
        monkeypatch.setattr(cb, "_decompose_sections", fake_decompose)
        monkeypatch.setattr(cb, "_notes_dir_for", lambda cid: None)

        async def default_parallel(chat, repo_dir, system_prompts, dispatches, tracker, notes_dir=None):
            return {}, {}
        monkeypatch.setattr(cb, "_run_parallel_section_agents", parallel or default_parallel)

        def default_builder(chat, specs, system_prompt):
            return orchestrator
        monkeypatch.setattr(cb, "_build_orchestrator_agent", builder or default_builder)

        monkeypatch.setattr(cb, "_safe_build_llm", lambda *a, **kw: None)
        monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "false")
        # The opener-dedup guard (2.4) would otherwise build a REAL repair
        # client here: these tests repeat identical openers across sections
        # (scripted orchestrator), which is exactly what the guard repairs.
        monkeypatch.setattr(cb, "_make_repair_llm", lambda *a, **kw: None)

        async def fake_repair(content, llm):
            return content, {}
        monkeypatch.setattr(cb, "run_repair_loop", fake_repair)
        monkeypatch.setattr(cb, "_index_in_background", lambda *a, **kw: None)

        import api.expert.knowledge as knowledge_mod
        async def fake_knowledge(pid, query):
            return ""
        monkeypatch.setattr(knowledge_mod, "_retrieve_product_knowledge", fake_knowledge)

    def test_orchestrated_happy_path_provenance(self, fake_artifact, fake_product, fake_repo_dir, monkeypatch):
        orchestrator = _FakeOrchestrator({
            sid: "Agent wrote this section about `main.py`."
            for sid in cb.SECTION_ORDER
        })
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator)

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        assert "Agent wrote this section" in result
        assert "Документация по кодовой базе" in result
        assert orchestrator.calls == 1  # one orchestrated run, nothing reused
        # Legacy page contract preserved...
        assert len(fake_artifact.pages) == 7
        page = fake_artifact.pages["page_overview"]
        assert page["title"] and page["content"]
        # ...plus the additive provenance key with the subagent metadata.
        prov = page["provenance"]
        assert prov["generator"] == "deepagents"
        assert prov["regen"] == "generated"
        assert prov["prompt_file"] == "docgen_sections.md"
        # The orchestrated path tracks no reads: provenance falls back to the
        # citations the text actually contains.
        assert prov["source_files"] == ["main.py"]
        assert prov["model"]
        assert prov["fingerprint"]
        assert "main.py" in prov["citations"]["resolved"]
        assert "judge" not in prov  # judge disabled -> key omitted

    def test_opener_duplicate_reported_in_provenance(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        """2.4: the scripted orchestrator repeats one opener across sections —
        the guard reports the duplicate in page provenance; no repair here
        (the hermetic patcher's repair seam is None), text stays verbatim."""
        orchestrator = _FakeOrchestrator({
            sid: "Same opening line about `main.py`. Unique tail."
            for sid in cb.SECTION_ORDER
        })
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator)

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        dup = fake_artifact.pages["page_architecture"]["provenance"]["opener_duplicate"]
        assert dup["similar_to"] == "overview"
        assert dup["repaired"] is False
        assert dup["similarity"] == 1.0
        # Первая секция — эталон сравнения, отчёта о ней нет.
        assert "opener_duplicate" not in (
            fake_artifact.pages["page_overview"]["provenance"]
        )
        # Без ремонта текст не переписывается.
        assert "Same opening line" in result

    def test_opener_dedup_env_kill_switch(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        monkeypatch.setenv("DOCGEN_OPENER_DEDUP", "false")
        orchestrator = _FakeOrchestrator({
            sid: "Same opening line about `main.py`."
            for sid in cb.SECTION_ORDER
        })
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator)

        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        assert "opener_duplicate" not in (
            fake_artifact.pages["page_architecture"]["provenance"]
        )

    def test_corroborate_drops_fabricated_identifier(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        """3.1: a sentence naming an identifier that is nowhere in the
        section's sources (file contents + repo paths) is dropped from the
        persisted text and reported in provenance; the grounded sentence
        (the `main.py` citation) survives."""
        orchestrator = _FakeOrchestrator({
            sid: (
                "Agent wrote this section about `main.py`. "
                f"Ghost `GhostWidgetFactory` powers {sid}."
            )
            for sid in cb.SECTION_ORDER
        })
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator)

        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        page = fake_artifact.pages["page_overview"]
        assert "GhostWidgetFactory" not in page["content"]
        assert "Agent wrote this section about `main.py`." in page["content"]
        prov = page["provenance"]
        assert prov["corroborate"] == {"removed": ["GhostWidgetFactory"]}
        assert "GhostWidgetFactory" not in (fake_artifact.generated_docs or "")

    def test_subagent_specs_and_orchestrator_prompt(self, fake_artifact, fake_product, fake_repo_dir, monkeypatch):
        from api.prompts import SECTION_PROMPTS

        captured: Dict[str, Any] = {}

        def capture_builder(chat, specs, system_prompt):
            captured["specs"] = specs
            captured["system_prompt"] = system_prompt
            return _FakeOrchestrator({sid: "text" for sid in cb.SECTION_ORDER})

        self._patch_flow(monkeypatch, fake_repo_dir, builder=capture_builder)
        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        specs = captured["specs"]
        assert [s["name"] for s in specs] == [f"section-{sid}" for sid in cb.SECTION_ORDER]
        for spec in specs:
            assert spec["model"] is not None
            assert spec["tools"]  # repo tools attached
            assert "system_prompt" in spec
        # Each subagent system prompt = writer rules + ITS OWN contract.
        for spec, sid in zip(specs, cb.SECTION_ORDER):
            assert SECTION_PROMPTS[sid][:60] in spec["system_prompt"]
        # The orchestrator prompt names the repo and lists every section.
        assert "testrepo" in captured["system_prompt"]
        assert "- `overview`" in captured["system_prompt"]
        assert "(none)" in captured["system_prompt"]  # no reused sections

    def test_partial_orchestration_parallel_fallback(self, fake_artifact, fake_product, fake_repo_dir, monkeypatch):
        orchestrated = dict.fromkeys(cb.SECTION_ORDER[:3], "Orchestrated section text")
        orchestrator = _FakeOrchestrator(orchestrated)

        parallel_calls: Dict[str, Any] = {}

        async def fake_parallel(chat, repo_dir, system_prompts, dispatches, tracker, notes_dir=None):
            parallel_calls["sids"] = list(system_prompts)
            parallel_calls["prompts"] = dict(system_prompts)
            parallel_calls["dispatches"] = dict(dispatches)
            return (
                {sid: f"Parallel wrote {sid}" for sid in system_prompts},
                {sid: ["src/parallel.py"] for sid in system_prompts},
            )

        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator, parallel=fake_parallel)

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        missing = cb.SECTION_ORDER[3:]
        assert parallel_calls["sids"] == missing
        # The parallel fallback gets the SAME rendered contracts + dispatches.
        assert set(parallel_calls["prompts"]) == set(missing)
        assert set(parallel_calls["dispatches"]) == set(missing)
        assert "Orchestrated section text" in result
        assert f"Parallel wrote {missing[0]}" in result
        # Both paths are recorded as deepagent-generated.
        assert fake_artifact.pages["page_overview"]["provenance"]["regen"] == "generated"
        par_prov = fake_artifact.pages[f"page_{missing[-1]}"]["provenance"]
        assert par_prov["generator"] == "deepagents"
        assert par_prov["source_files"] == ["src/parallel.py"]

    def test_orchestrator_failure_full_parallel_success(self, fake_artifact, fake_product, fake_repo_dir, monkeypatch):
        orchestrator = _FakeOrchestrator({}, error=RuntimeError("orchestrator exploded"))

        async def fake_parallel(chat, repo_dir, system_prompts, dispatches, tracker, notes_dir=None):
            texts = {sid: f"Parallel wrote {sid}" for sid in system_prompts}
            files = {sid: ["main.py"] for sid in system_prompts}
            for sid in system_prompts:
                tracker.section_started(sid)
                tracker.section_finished(sid)
            return texts, files

        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator, parallel=fake_parallel)

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))
        assert f"Parallel wrote {cb.SECTION_ORDER[0]}" in result
        assert fake_artifact.pages["page_overview"]["provenance"]["generator"] == "deepagents"

    def test_all_agent_paths_fail_falls_back_to_standard_llm(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        orchestrator = _FakeOrchestrator({}, error=RuntimeError("orchestrator exploded"))
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator)

        class FakeLLM:
            async def generate(self, prompt):
                return "Fallback LLM section text"
        monkeypatch.setattr(cb, "_safe_build_llm", lambda *a, **kw: FakeLLM())

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))
        assert "Fallback LLM section text" in result
        page = fake_artifact.pages["page_overview"]
        assert page["provenance"]["generator"] == "standard-llm"
        assert page["provenance"]["regen"] == "legacy-fallback"

    def test_diff_regen_reuses_unchanged_sections(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        """Second run with identical sources reuses sections verbatim: no
        chat model, no orchestrator, content and fingerprints stay stable."""
        # The first-run text carries a `main.py` citation: reuse requires
        # non-empty provenance.source_files (extracted from citations), so a
        # citation-less text would store no source set and never reuse.
        orchestrator1 = _FakeOrchestrator(
            {sid: "FIRST RUN CONTENT `main.py`" for sid in cb.SECTION_ORDER}
        )
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator1)
        result1 = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))
        assert "FIRST RUN CONTENT" in result1
        first_fingerprint = fake_artifact.pages["page_overview"]["provenance"]["fingerprint"]

        orchestrator2 = _FakeOrchestrator(
            {sid: "SECOND RUN MUTATED" for sid in cb.SECTION_ORDER}
        )
        monkeypatch.setattr(
            cb, "_build_orchestrator_agent",
            lambda chat, specs, system_prompt: orchestrator2,
        )

        result2 = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))
        assert orchestrator2.calls == 0  # full reuse: no orchestrator at all
        assert "FIRST RUN CONTENT" in result2
        assert "SECOND RUN MUTATED" not in result2
        prov2 = fake_artifact.pages["page_overview"]["provenance"]
        assert prov2["regen"] == "reused-unchanged"
        assert prov2["fingerprint"] == first_fingerprint

    def test_diff_regen_after_source_change(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        """Changing a tracked source file invalidates the fingerprint and
        forces regeneration (content hashes, not mtimes)."""
        # Citation so run 1 stores main.py in provenance.source_files — the
        # second run then re-fingerprints it and sees the content hash change.
        orchestrator1 = _FakeOrchestrator(
            {sid: "BEFORE CHANGE `main.py`" for sid in cb.SECTION_ORDER}
        )
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator1)
        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        with open(os.path.join(fake_repo_dir, "main.py"), "a") as f:
            f.write("# changed\n")

        orchestrator2 = _FakeOrchestrator(
            {sid: "AFTER CHANGE" for sid in cb.SECTION_ORDER}
        )
        monkeypatch.setattr(
            cb, "_build_orchestrator_agent",
            lambda chat, specs, system_prompt: orchestrator2,
        )

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))
        assert orchestrator2.calls == 1
        assert "AFTER CHANGE" in result
        assert "BEFORE CHANGE" not in result
        assert fake_artifact.pages["page_overview"]["provenance"]["regen"] == "generated"

    def test_partial_reuse_skips_reused_in_dispatch(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        """A second run where SOME sections are reused only dispatches the
        rest; the orchestrator prompt lists the reused ones."""
        orchestrator1 = _FakeOrchestrator(
            {sid: "FIRST RUN CONTENT" for sid in cb.SECTION_ORDER}
        )
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator1)
        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        # Regenerate only: change a file, then monkeypatch the regen plan so
        # ONLY overview is regenerated (the rest is reused verbatim).
        with open(os.path.join(fake_repo_dir, "main.py"), "a") as f:
            f.write("# changed\n")

        captured: Dict[str, Any] = {}

        def capture_builder(chat, specs, system_prompt):
            captured["specs"] = specs
            captured["system_prompt"] = system_prompt
            return _FakeOrchestrator({"overview": "REGENERATED OVERVIEW"})

        monkeypatch.setattr(cb, "_build_orchestrator_agent", capture_builder)

        real_plan = cb.plan_regeneration

        def plan_only_overview(old_pages, old_sections, repo_dir, section_ids, **kw):
            # Force full reuse for everything except overview.
            from types import SimpleNamespace
            return SimpleNamespace(reuse={
                sid: old_sections[sid] for sid in section_ids if sid != "overview"
            })

        monkeypatch.setattr(cb, "plan_regeneration", plan_only_overview)

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))
        assert "REGENERATED OVERVIEW" in result
        assert "FIRST RUN CONTENT" in result  # reused sections kept
        # Only ONE subagent spec was built...
        assert [s["name"] for s in captured["specs"]] == ["section-overview"]
        # ...and the orchestrator prompt lists the reused sections.
        for sid in cb.SECTION_ORDER[1:]:
            assert f"`{sid}`" in captured["system_prompt"]

    def test_progress_events_complete_run(self, fake_artifact, fake_product, fake_repo_dir, monkeypatch):
        events, progress_cb = _collector()
        orchestrator = _FakeOrchestrator(
            {sid: "section body" for sid in cb.SECTION_ORDER}
        )
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator)

        asyncio.run(cb.generate_codebase_docs(
            fake_artifact, fake_product, progress=progress_cb,
        ))

        phases = [e["phase"] for e in events if "phase" in e]
        # Phase order is a subsequence of the canonical lifecycle.
        assert phases[0] == "cloning"
        assert phases.index("planning") < phases.index("sections")
        assert phases.index("sections") < phases.index("verifying")
        assert phases[-1] == "indexing"
        # Every section reports done EXACTLY once.
        done = [e["section_done"] for e in events if "section_done" in e]
        assert sorted(done) == sorted(cb.SECTION_ORDER)
        assert len(done) == len(set(done))
        # sections_total is attached to the section-phase events.
        assert all(
            e.get("sections_total") == 7
            for e in events
            if e.get("phase") == "sections" or "section_done" in e
        )

    def test_generated_docs_indexed_not_repo_path(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        """The generated MARKDOWN is indexed, not the clone path string."""
        orchestrator = _FakeOrchestrator({sid: "Indexed content here" for sid in cb.SECTION_ORDER})
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator)

        indexed = []

        def track_index(content, dataset, **kw):
            indexed.append((content, dataset, kw))
        monkeypatch.setattr(cb, "_index_in_background", track_index)

        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))
        assert len(indexed) == 1
        content, dataset, kw = indexed[0]
        assert content != fake_repo_dir
        assert "Indexed content here" in content
        assert dataset == "prod_prod_agent"
        assert kw.get("source_type") == "codebase"


# ============================================================================
# Real deepagents graph smoke (FakeChatModel over create_deep_agent)
# ============================================================================
class _ScriptedChatModel(BaseChatModel):
    """Minimal chat model returning scripted AIMessages (tool-call flow)."""

    responses: List[Any]
    _i: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-docgen"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        idx = min(self._i, len(self.responses) - 1)
        self._i += 1
        return ChatResult(
            generations=[ChatGeneration(message=self.responses[idx])]
        )


class TestDeepagentsSmoke:
    def test_section_agent_reads_file_and_finishes(self, tmp_path):
        """Build a REAL deepagents agent (repo tools + planning middleware)
        with the actual section-writer system prompt over a scripted fake
        chat model: the tool-call round trip must read the real file, finish,
        and report the file as provenance evidence."""
        from langchain_core.messages import AIMessage
        from deepagents import create_deep_agent

        (tmp_path / "main.py").write_text("import os\nprint('hello')\n")

        model = _ScriptedChatModel(responses=[
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "repo_read_file",
                    "args": {"path": "main.py"},
                    "id": "call_1",
                }],
            ),
            AIMessage(content="Final smoke section text"),
        ])
        rules = cb._section_writer_rules("en")
        contract = cb._build_section_contract(
            repo_url="https://github.com/o/r",
            repo_name="r",
            sid="overview",
            title="Overview",
            repo_brief="brief",
            sections_list="- `overview` — Overview",
            hints=None,
        )
        agent = create_deep_agent(
            model=model,
            tools=cb.build_repo_tools(str(tmp_path)),
            system_prompt=cb._build_section_agent_system_prompt(rules, contract),
        )
        text, files = asyncio.run(cb._run_agent_section(agent, "document this repo"))
        assert text == "Final smoke section text"
        assert files == ["main.py"]

    def test_orchestrator_agent_constructs_with_specs(self, tmp_path):
        """The orchestrator builds against the real create_deep_agent with
        declarative section subagent specs (construction-level smoke)."""
        chat = _ScriptedChatModel(responses=[])
        specs = cb._build_section_subagent_specs(
            chat, str(tmp_path),
            {"overview": "Write the overview section."},
            "en",
        )
        agent = cb._build_orchestrator_agent(
            chat, specs, "You are the orchestrator for `r`.",
        )
        assert agent is not None


# ============================================================================
# Stage D: adaptive small-context caps
# ============================================================================
class TestAdaptiveCaps:
    def test_ctx_scale_clamped(self, monkeypatch):
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 131_072)
        assert cb._ctx_scale() == 1.0  # clamped at the ceiling
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 16_384)
        assert cb._ctx_scale() == 0.5
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 8_192)
        assert cb._ctx_scale() == 0.25  # clamped at the floor
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: None)
        assert cb._ctx_scale() == 0.25  # unset resolves to 8k → floor

    def test_repo_read_max_chars(self, monkeypatch):
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 1_000_000)
        assert cb._repo_read_max_chars() == cb._REPO_READ_MAX_CHARS  # ceiling
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 16_384)
        assert cb._repo_read_max_chars() == 12_288  # (16384 // 4) * 3
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 1_024)
        assert cb._repo_read_max_chars() == 4_000  # floor

    def test_hint_caps_scale_with_floors(self, monkeypatch):
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 131_072)
        assert cb._hint_caps() == (
            cb._HINT_MAX_FILES, cb._HINT_FILE_MAX_CHARS, cb._HINT_FOCUS_MAX_CHARS,
        )
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 8_192)
        # 0.25 scale would give (2, 50, 100) — floors keep hints usable.
        assert cb._hint_caps() == (3, 60, 120)

    def test_section_instruction_max_chars(self, monkeypatch):
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 131_072)
        assert cb._section_instruction_max_chars() == cb._SECTION_INSTRUCTION_MAX_CHARS
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 8_192)
        assert cb._section_instruction_max_chars() == 3_000  # 12000 * 0.25

    def test_docgen_max_completion_tokens(self, monkeypatch):
        monkeypatch.setattr(
            "api.config.get_model_config", lambda model: {"model_kwargs": {}},
        )
        # Unset window → None (leave the model default).
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: None)
        assert cb._docgen_max_completion_tokens() is None

        def _boom():
            raise RuntimeError("no window")

        monkeypatch.setattr(cb, "_resolve_docgen_context_window", _boom)
        assert cb._docgen_max_completion_tokens() is None
        # Window-proportional reserve with a floor.
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 32_768)
        assert cb._docgen_max_completion_tokens() == 32_768 // 6
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 2_048)
        assert cb._docgen_max_completion_tokens() == 512  # floor
        # Explicit generator-config max_tokens wins when smaller.
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 32_768)
        monkeypatch.setattr(
            "api.config.get_model_config",
            lambda model: {"model_kwargs": {"max_tokens": 128}},
        )
        assert cb._docgen_max_completion_tokens() == 128

    def test_is_context_overflow(self):
        assert cb._is_context_overflow(
            ValueError("This model's maximum context length is 4096 tokens.")
        )
        assert cb._is_context_overflow(
            RuntimeError("Error code: 400 - 'context_length_exceeded'")
        )
        assert cb._is_context_overflow(Exception("message too long"))
        assert not cb._is_context_overflow(RuntimeError("connection error"))
        assert not cb._is_context_overflow(ValueError("rate limited"))

    def test_build_repo_tools_read_cap_override(self, tmp_path):
        (tmp_path / "big.py").write_text("x" * 500)
        tools = cb.build_repo_tools(str(tmp_path), read_max_chars=100)
        read = next(t for t in tools if t.name == "repo_read_file")
        out = read.invoke({"path": "big.py"})
        assert out.endswith("\n... (truncated)")
        assert len(out) < 120
        # Sanity: the override is honored exactly, not the adaptive default.
        body = out[: -len("\n... (truncated)")]
        assert len(body) == 100

    def test_brief_max_chars(self, monkeypatch):
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 131_072)
        assert cb._brief_max_chars() == cb._BRIEF_MAX_CHARS
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 16_384)
        assert cb._brief_max_chars() == 6_000  # 12000 * 0.5
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 8_192)
        assert cb._brief_max_chars() == 3_000  # floor keeps the brief usable

    def test_repo_brief_scales_for_small_window(self, monkeypatch):
        """At ctx=8192 the whole brief must fit a fraction of the window:
        12k chars (~3k tokens) of brief + 3k of instruction alone blew the
        6144-token prompt budget of the small-window fallback path."""
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 8_192)
        brief = cb._build_repo_brief(
            repo_url="https://github.com/o/r",
            repo_type="github",
            file_analysis={
                "primary_language": "Python", "file_count": 42,
                "main_directories": [f"d{i}" for i in range(30)],
                "config_files": [f"c{i}.toml" for i in range(30)],
                "cicd_files": [], "docker_files": [],
            },
            file_tree="\n".join(f"src/mod{i}/file.py" for i in range(200)),
            readme="readme " * 4_000,  # ~28k chars before the cap
        )
        # cap() appends its own truncation marker on top of the slice.
        assert len(brief) <= cb._brief_max_chars() + 100
        # ~750 tokens at the 0.25 floor — a fraction of the 6144 budget.
        assert cb._count_tokens(brief) <= 1_500


class TestFitFileBlocksToBudget:
    def _blocks(self, n, size=200):
        return [
            f"### File: f{i}.py\n```\n{'x' * size}\n```\n" for i in range(n)
        ]

    def test_fits_unchanged(self):
        text = "\n".join(self._blocks(2))
        out = cb._fit_file_blocks_to_budget(text, 10_000)
        assert out == text

    def test_empty_passthrough(self):
        assert cb._fit_file_blocks_to_budget("", 100) == ""

    def test_drops_whole_blocks_from_end(self):
        blocks = self._blocks(6)
        text = "\n".join(blocks)
        prefix = "P" * 50
        # Budget for exactly the first two blocks (prefix + the +1
        # per-block separator token are measured too).
        budget = (
            cb._count_tokens(prefix)
            + cb._count_tokens(blocks[0]) + 1
            + cb._count_tokens(blocks[1]) + 1
        )
        out = cb._fit_file_blocks_to_budget(text, budget, prefix=prefix)
        assert "### File: f0.py" in out
        assert "### File: f1.py" in out
        assert "### File: f2.py" not in out
        assert "### File: f5.py" not in out
        assert "(4 file block(s) omitted" in out
        # Blocks stay INTACT — no partial mid-block slicing.
        assert "```" in out

    def test_first_block_always_kept(self):
        text = "\n".join(self._blocks(4))
        out = cb._fit_file_blocks_to_budget(text, 5)  # far below one block
        assert "### File: f0.py" in out
        assert "(3 file block(s) omitted" in out

    def test_non_block_text_head_slice(self):
        text = "just some prose " * 500  # no "### File:" markers
        out = cb._fit_file_blocks_to_budget(text, 10)
        assert out.startswith("just some prose")
        assert out.endswith("... (truncated to fit the context budget)")
        assert len(out) < len(text)

    def test_suffix_measured_too(self):
        blocks = self._blocks(4)
        text = "\n".join(blocks)
        suffix = "S" * 40
        # Enough room for text alone but NOT for text + suffix.
        budget = cb._count_tokens(text) + cb._count_tokens(suffix) - 5
        out = cb._fit_file_blocks_to_budget(text, budget, suffix=suffix)
        assert "omitted" in out
        assert "### File: f0.py" in out

    def test_oversize_first_block_is_head_sliced(self):
        """When even the first block exceeds the budget, head-slice it
        instead of returning an over-budget prompt (guaranteed 400).
        """
        blocks = self._blocks(4, size=4_000)
        text = "\n".join(blocks)
        out = cb._fit_file_blocks_to_budget(text, 30)
        assert out.startswith("### File: f0.py")
        assert len(out) < 1_200  # ~1000-char head + notes, not a whole block
        assert "truncated to fit the context budget" in out
        assert "(3 file block(s) omitted" in out

    def test_negative_budget_still_returns_bounded_text(self):
        """budget <= 0 (oversized prefix) must not return the full corpus.
        """
        text = "\n".join(self._blocks(4, size=2_000))
        out = cb._fit_file_blocks_to_budget(text, 10, prefix="P" * 10_000)
        assert out.startswith("### File: f0.py")
        assert len(out) < 1_200
        assert "omitted" in out


# ============================================================================
# 2.3(a): per-section incremental persistence (crash safety)
# ============================================================================
class TestSectionCheckpoints:
    """A crash mid-run keeps every VERIFIED section durable in the DB, and a
    rerun reuses those sections via diff regeneration (fingerprint match).

    The artifact is a REAL CodebaseORM row over ``isolated_db`` (the worker
    pattern: a fresh session per run); the crash is simulated by raising from
    ``build_section_provenance`` on the Nth section — any exception escaping
    the section loop kills the run exactly like a worker kill would.
    """

    @pytest.fixture
    def fake_repo_dir(self, tmp_path):
        (tmp_path / "main.py").write_text("import os\nprint('hello')\n")
        (tmp_path / "README.md").write_text("# Test Repo\nA test.\n")
        return str(tmp_path)

    @pytest.fixture
    def seeded_row(self, isolated_db):
        from api.models import CodebaseORM, ProductORM

        with isolated_db.SessionLocal() as s:
            s.add(ProductORM(id="prod_ck", name="P"))
            s.add(CodebaseORM(
                id="cb_ck", product_id="prod_ck", name="testrepo",
                repo_url="https://github.com/owner/testrepo", repo_type="github",
            ))
            s.commit()
        return isolated_db

    @staticmethod
    def _load_detached(isolated_db, entity_id="cb_ck"):
        """Load the row DETACHED with all generate-read attributes preloaded:
        the checkpoint opens its own session while the run is in progress, and
        a lazy refresh from a second session would race on the shared
        StaticPool connection."""
        from api.models import CodebaseORM

        s = isolated_db.SessionLocal()
        try:
            entity = s.get(CodebaseORM, entity_id)
            _ = (entity.repo_url, entity.repo_type, entity.token, entity.name,
                 entity.pages, entity.generated_docs, entity.id)
            s.expunge(entity)
            return entity
        finally:
            s.close()

    def _patch_flow(self, monkeypatch, fake_repo_dir, orchestrator=None):
        # Reuse the agent-flow patcher (it ignores ``self``): everything that
        # could touch the network or the clone machinery is faked.
        TestDocgenAgentFlow._patch_flow(
            object(), monkeypatch, fake_repo_dir, orchestrator=orchestrator,
        )

    def test_crash_on_third_section_persists_two_and_rerun_reuses(
        self, seeded_row, fake_repo_dir, monkeypatch
    ):
        isolated_db = seeded_row
        entity = self._load_detached(isolated_db)
        orchestrator1 = _FakeOrchestrator({
            sid: f"RUN ONE section body citing `main.py` — {sid}"
            for sid in cb.SECTION_ORDER
        })
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator1)

        real_provenance = cb.build_section_provenance
        calls = {"n": 0}

        def crashing_provenance(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("simulated worker crash mid-run")
            return real_provenance(*a, **kw)

        monkeypatch.setattr(cb, "build_section_provenance", crashing_provenance)

        with pytest.raises(RuntimeError, match="simulated worker crash"):
            asyncio.run(
                cb.generate_codebase_docs(entity, SimpleNamespaceProduct("prod_ck"))
            )

        # Две ГОТОВЫЕ секции легли в БД ещё до обрыва; остальные пусты.
        from api.models import CodebaseORM

        with isolated_db.SessionLocal() as s:
            row = s.get(CodebaseORM, "cb_ck")
            done = [
                sid for sid in cb.SECTION_ORDER
                if (row.pages or {}).get(f"page_{sid}", {}).get("content", "").strip()
            ]
            assert done == cb.SECTION_ORDER[:2]
            assert all(
                "RUN ONE" in row.pages[f"page_{sid}"]["content"] for sid in done
            )
            assert "RUN ONE" in (row.generated_docs or "")
            checkpoint_contents = {
                sid: row.pages[f"page_{sid}"]["content"] for sid in done
            }
            fp_overview = row.pages["page_overview"]["provenance"]["fingerprint"]
            assert fp_overview  # fingerprint записан — без него reuse невозможен

        # Повторный прогон: свежая сессия (как новый job), источник — БД.
        entity2 = self._load_detached(isolated_db)
        orchestrator2 = _FakeOrchestrator({
            sid: f"RUN TWO REGENERATED — {sid}" for sid in cb.SECTION_ORDER
        })
        monkeypatch.setattr(cb, "build_section_provenance", real_provenance)
        monkeypatch.setattr(
            cb, "_build_orchestrator_agent",
            lambda *a, **kw: orchestrator2,
        )

        result = asyncio.run(
            cb.generate_codebase_docs(entity2, SimpleNamespaceProduct("prod_ck"))
        )

        # Первые две секции переиспользованы вербатим (orchestrator2 гонялся
        # только за остальными пятью), fingerprint совпал.
        assert orchestrator2.calls == 1
        assert "RUN ONE" in result
        # Переиспользованные тела — вербатим из чекпойнта в БД.
        for sid in done:
            assert entity2.pages[f"page_{sid}"]["content"] == checkpoint_contents[sid]
        prov = entity2.pages["page_overview"]["provenance"]
        assert prov["regen"] == "reused-unchanged"
        assert prov["fingerprint"] == fp_overview
        # Остальные секции — свежесгенерированные.
        third = cb.SECTION_ORDER[2]
        assert entity2.pages[f"page_{third}"]["provenance"]["regen"] == "generated"
        assert "RUN TWO REGENERATED" in entity2.pages[f"page_{third}"]["content"]
        # Все 7 страниц на месте после финального персиста.
        assert len(entity2.pages) == 7

    def test_crash_before_first_section_persists_nothing(
        self, seeded_row, fake_repo_dir, monkeypatch
    ):
        isolated_db = seeded_row
        entity = self._load_detached(isolated_db)
        orchestrator = _FakeOrchestrator({
            sid: f"BODY `main.py` — {sid}" for sid in cb.SECTION_ORDER
        })
        self._patch_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator)

        def crash_immediately(*a, **kw):
            raise RuntimeError("crash before any section finished")

        monkeypatch.setattr(cb, "build_section_provenance", crash_immediately)

        with pytest.raises(RuntimeError, match="crash before any section"):
            asyncio.run(
                cb.generate_codebase_docs(entity, SimpleNamespaceProduct("prod_ck"))
            )

        from api.models import CodebaseORM

        with isolated_db.SessionLocal() as s:
            row = s.get(CodebaseORM, "cb_ck")
            assert row.pages is None
            assert row.generated_docs is None

    def test_all_placeholder_failure_writes_no_checkpoint(
        self, seeded_row, fake_repo_dir, monkeypatch
    ):
        """Тотальный отказ LLM (все секции — placeholder) не должен оставлять
        после себя чекпойнт-мусор: ни чекпойнтов, ни финального персиста."""
        isolated_db = seeded_row
        entity = self._load_detached(isolated_db)
        # Пустой оркестратор: ни одной task-диспетчеризации → все секции
        # уходят в fallback (llm=None → "") → placeholder → ValueError.
        self._patch_flow(
            monkeypatch, fake_repo_dir, orchestrator=_FakeOrchestrator({})
        )

        with pytest.raises(ValueError, match="Не удалось сгенерировать"):
            asyncio.run(
                cb.generate_codebase_docs(entity, SimpleNamespaceProduct("prod_ck"))
            )

        from api.models import CodebaseORM

        with isolated_db.SessionLocal() as s:
            row = s.get(CodebaseORM, "cb_ck")
            assert row.pages is None  # placeholder-чекпойнты не писались
            assert row.generated_docs is None

    def test_checkpoint_helper_missing_row_returns_false(self, seeded_row):
        from api.docgen._common import _checkpoint_partial_docs
        from api.models import CodebaseORM

        assert _checkpoint_partial_docs(
            "ghost-id", CodebaseORM, "md", {"page_overview": {}}
        ) is False
        assert _checkpoint_partial_docs(None, CodebaseORM, "md", {}) is False
        assert _checkpoint_partial_docs("x", None, "md", {}) is False


class SimpleNamespaceProduct:
    """Minimal product stand-in (id only) for the checkpoint tests."""

    def __init__(self, pid: str):
        self.id = pid


# ============================================================================
# Units restructure: slugs, decomposition parsing, notes workspace
# ============================================================================
class TestSlugify:
    def test_ascii_slug(self):
        assert cb._slugify("User Authentication & Sessions") == "user-authentication-sessions"

    def test_empty(self):
        assert cb._slugify("") == ""
        assert cb._slugify("   ") == ""
        assert cb._slugify("---") == ""

    def test_non_ascii_yields_empty(self):
        assert cb._slugify("Аутентификация пользователей") == ""

    def test_capped_at_64(self):
        assert len(cb._slugify("a" * 200)) == 64


class TestParseDecomposition:
    def test_valid_items_normalized(self):
        data = {
            "functional": [{"title": "Auth", "focus": "login flows", "slug": "ignored"}],
            "technical": [
                {"title": "REST API", "focus": "endpoints", "kind": "endpoint"},
                {"title": "Weird kind", "kind": "cron"},  # unknown -> reference
            ],
            "datamodel": [{"title": "Core tables", "kind": "anything"}],  # -> layer
        }
        out = cb._parse_decomposition(data)
        # Slug derives from the TITLE, not the LLM slug.
        assert out["functional"][0]["slug"] == "auth"
        assert out["technical"][0]["kind"] == "endpoint"
        assert out["technical"][1]["kind"] == "reference"
        assert out["datamodel"][0]["kind"] == "layer"

    def test_junk_dropped_and_cap_applied(self):
        data = {
            "functional": [{"focus": "no title"}, "string-item", None]
            + [{"title": f"Cap {i}"} for i in range(20)],
        }
        out = cb._parse_decomposition(data)
        assert len(out["functional"]) == cb._SUBPAGE_CAPS["functional"]

    def test_slug_dedupe_and_positional_fallback(self):
        data = {"technical": [
            {"title": "Same Name"},
            {"title": "Same Name"},
            {"title": "Аутентификация"},  # non-latin -> positional id
        ]}
        slugs = [i["slug"] for i in cb._parse_decomposition(data)["technical"]]
        assert slugs == ["same-name", "same-name-2", "u3"]

    def test_bad_input_returns_empty(self):
        assert cb._parse_decomposition(None) == {}
        assert cb._parse_decomposition([1, 2]) == {}
        # Non-subpage sections are ignored even if present.
        assert cb._parse_decomposition({"overview": [{"title": "x"}]}) == {}

    def test_decompose_sections_never_raises(self):
        class Boom:
            async def ainvoke(self, messages):
                raise RuntimeError("llm down")
        assert asyncio.run(cb._decompose_sections(Boom(), "b", "s", {})) == {}

    def test_decompose_sections_parses_llm_json(self):
        from langchain_core.messages import AIMessage

        class Chat:
            async def ainvoke(self, messages):
                return AIMessage(content=json.dumps({
                    "functional": [{"title": "Auth"}],
                }))
        out = asyncio.run(cb._decompose_sections(Chat(), "b", "s", {}))
        assert out["functional"][0]["slug"] == "auth"


class TestNotesWorkspace:
    def test_dir_for_valid_and_invalid_ids(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PRODUCTARIUM_STATE_DIR", str(tmp_path))
        d = cb._notes_dir_for("abc-123_X")
        assert d == os.path.join(str(tmp_path), "docgen_notes", "abc-123_X")
        assert cb._notes_dir_for(None) is None
        assert cb._notes_dir_for("") is None
        assert cb._notes_dir_for("../evil") is None
        assert cb._notes_dir_for("has/slash") is None

    def test_write_read_roundtrip(self, tmp_path):
        nd = str(tmp_path / "notes")
        assert cb._notes_write(nd, "summary_overview.md", "hello world").startswith("wrote")
        assert cb._notes_read(nd, "summary_overview.md") == "hello world"
        assert "ERROR" in cb._notes_read(nd, "missing.md")

    def test_name_confinement(self, tmp_path):
        nd = str(tmp_path / "notes")
        os.makedirs(nd)
        for bad in ("../escape.md", "sub/dir.md", "", ".hidden.md", "with space.md"):
            assert "ERROR" in cb._notes_write(nd, bad, "x")
        assert not os.path.exists(str(tmp_path / "escape.md"))

    def test_write_caps_content(self, tmp_path):
        nd = str(tmp_path / "notes")
        cb._notes_write(nd, "big.md", "x" * (cb._NOTES_WRITE_MAX_CHARS + 5000))
        stored = cb._notes_read(nd, "big.md")
        assert len(stored) <= cb._NOTES_WRITE_MAX_CHARS + 20
        assert stored.endswith("(truncated)")

    def test_write_masks_secrets(self, tmp_path):
        nd = str(tmp_path / "notes")
        secret = "ghp_1234567890abcdefghijklmnopqrstuv"
        cb._notes_write(nd, "secret.md", f"token: {secret}")
        assert secret not in cb._notes_read(nd, "secret.md")

    def test_store_none_dir_is_noop(self):
        cb._notes_store(None, "x.md", "y")  # must not raise

    def test_build_notes_tools_roundtrip_and_confinement(self, tmp_path):
        tools = cb.build_notes_tools(str(tmp_path / "notes"))
        assert [t.name for t in tools] == ["notes_read", "notes_write"]
        assert tools[1].invoke({"name": "summary_x.md", "content": "note body"}).startswith("wrote")
        assert tools[0].invoke({"name": "summary_x.md"}) == "note body"
        assert "ERROR" in tools[0].invoke({"name": "../etc/passwd"})
        assert "ERROR" in tools[1].invoke({"name": "../evil.md", "content": "x"})

    def test_notes_inline_family_summaries(self, tmp_path):
        nd = str(tmp_path / "notes")
        parent = cb._DocUnit(unit_id="functional", sid="functional", title="F")
        c1 = cb._DocUnit(unit_id="functional__auth", sid="functional", slug="auth", title="A")
        c2 = cb._DocUnit(unit_id="functional__billing", sid="functional", slug="billing", title="B")
        units = [c1, c2, parent]
        cb._notes_store(nd, "summary_functional__auth.md", "AUTH SUMMARY")
        cb._notes_store(nd, "summary_functional__billing.md", "BILLING SUMMARY")
        # Parent sees its children's summaries.
        inline = cb._notes_inline(nd, parent, units)
        assert "AUTH SUMMARY" in inline and "BILLING SUMMARY" in inline
        # Child sees parent + siblings, not itself.
        inline_child = cb._notes_inline(nd, c1, units)
        assert "functional__billing" in inline_child
        assert "AUTH SUMMARY" not in inline_child
        # None notes dir -> no inline block at all.
        assert cb._notes_inline(None, parent, units) == ""


class TestUnitsPagesAndMarkdown:
    def _family(self):
        child = cb._DocUnit(
            unit_id="functional__auth", sid="functional", slug="auth",
            title="Authentication", kind="", focus="login",
        )
        parent = cb._DocUnit(unit_id="functional", sid="functional", title="Functional")
        contents = {"functional": "PARENT BODY", "functional__auth": "CHILD BODY"}
        return child, parent, contents

    def test_units_pages_children_shape(self):
        child, parent, contents = self._family()
        pages = cb._units_pages(contents, [child, parent], "en")
        assert len(pages) == 8
        cpage = pages["page_functional__auth"]
        ppage = pages["page_functional"]
        assert cpage["parent"] == "page_functional"
        assert cpage["relatedPages"] == ["page_functional"]
        assert cpage["importance"] == "medium"
        assert cpage["title"] == "Authentication"
        assert cpage["content"] == "CHILD BODY"
        assert ppage["relatedPages"] == ["page_functional__auth"]
        assert ppage["importance"] == "high"

    def test_section_pages_wrapper_is_legacy_shape(self):
        sections = {sid: "x" for sid in cb.SECTION_ORDER}
        pages = cb._section_pages(sections, "ru")
        assert len(pages) == 7
        assert all("parent" not in p and p["importance"] == "high" for p in pages.values())

    def test_assemble_markdown_renders_children_under_parent(self):
        child, parent, contents = self._family()
        md = cb._assemble_markdown("repo", contents, "en", units=[child, parent])
        assert "## Functional Description" in md
        assert "### Authentication" in md
        assert "CHILD BODY" in md
        assert md.index("PARENT BODY") < md.index("CHILD BODY")

    def test_assemble_markdown_children_without_parent_content(self):
        child, parent, contents = self._family()
        contents["functional"] = ""
        md = cb._assemble_markdown("repo", contents, "en", units=[child, parent])
        assert "## Functional Description" in md and "### Authentication" in md

    def test_assemble_markdown_without_units_is_legacy(self):
        sections = {"overview": "A", "qa": "B"}
        md = cb._assemble_markdown("repo", sections, "ru")
        assert "## Общая информация" in md
        assert "## QA (Тестирование)" in md
        assert "## Системная архитектура" not in md

    def test_units_list_text_marks_children(self):
        child, parent, _ = self._family()
        text = cb._units_list_text([child, parent], "en")
        assert "- `functional__auth` — Authentication (subpage of `functional`)" in text
        assert "- `functional` — Functional Description" in text
        only = cb._units_list_text([child, parent], "en", only=["functional__auth"])
        assert only == "- `functional__auth` — Authentication (subpage of `functional`)"


class TestChildUnitsFromOldPages:
    def _old_pages(self):
        return {
            "page_functional": {"id": "page_functional", "title": "F", "content": "P"},
            "page_functional__auth": {
                "id": "page_functional__auth", "title": "Authentication",
                "content": "C", "parent": "page_functional",
                "provenance": {"subpage": {"kind": "", "focus": "login"}},
            },
            # Wrong parent marker: must be ignored.
            "page_technical__orphan": {
                "id": "page_technical__orphan", "title": "Orphan", "content": "O",
                "parent": "page_functional",
            },
        }

    def test_recovers_children_with_identity(self):
        units = cb._child_units_from_old_pages("functional", self._old_pages())
        assert len(units) == 1
        u = units[0]
        assert u.unit_id == "functional__auth"
        assert u.slug == "auth"
        assert u.title == "Authentication"
        assert u.kind == "" and u.focus == "login"
        assert u.is_child

    def test_missing_subpage_block_recovers_empty_identity(self):
        pages = {
            "page_datamodel__core": {
                "title": "Core", "content": "C", "parent": "page_datamodel",
            },
        }
        units = cb._child_units_from_old_pages("datamodel", pages)
        assert units[0].kind == "" and units[0].focus == ""

    def test_empty_content_ignored(self):
        pages = {
            "page_functional__empty": {
                "title": "E", "content": "  ", "parent": "page_functional",
            },
        }
        assert cb._child_units_from_old_pages("functional", pages) == []

    def test_child_prompt_hash_stable_and_identity_sensitive(self):
        child = cb._DocUnit(
            unit_id="functional__auth", sid="functional", slug="auth",
            title="Authentication", kind="endpoint", focus="login",
        )
        h1 = cb._child_prompt_hash("rules", child)
        assert h1 == cb._child_prompt_hash("rules", child)
        child2 = cb._DocUnit(
            unit_id="functional__auth", sid="functional", slug="auth",
            title="Authentication v2", kind="endpoint", focus="login",
        )
        assert h1 != cb._child_prompt_hash("rules", child2)


class TestUnitsTrackerEvents:
    def test_child_events_use_unit_counters(self):
        events, progress = _collector()
        tracker = cb._SectionProgressTracker(
            progress, sections_total=7,
            unit_sections={"functional__auth": "functional", "functional__billing": "functional"},
        )
        tracker.section_started("functional__auth")
        tracker.section_finished("functional__auth")
        start = events[0]
        assert start["current_section"] == "functional"
        assert start["current_unit"] == "functional__auth"
        assert start["units_total"] == 2
        finish = events[-1]
        assert finish["unit_done"] == "functional__auth"
        assert finish["units_done"] == 1
        assert finish["units_total"] == 2
        assert "section_done" not in finish  # parents-only accounting

    def test_parent_events_unchanged_with_map(self):
        events, progress = _collector()
        tracker = cb._SectionProgressTracker(
            progress, sections_total=7,
            unit_sections={"functional__auth": "functional"},
        )
        tracker.section_started("overview")
        tracker.section_finished("overview")
        assert events[0] == {
            "phase": "sections", "sections_total": 7, "current_section": "overview",
        }
        assert events[-1]["section_done"] == "overview"


# ============================================================================
# Units restructure: full pipeline flow with decomposition
# ============================================================================
class TestUnitsAgentFlow:
    """Full generate_codebase_docs runs WITH a scripted decomposer."""

    _DECOMP = {
        "functional": [
            {"slug": "auth", "title": "Authentication", "focus": "login/logout flows"},
            {"slug": "billing", "title": "Billing", "focus": "invoices"},
        ],
        "technical": [
            {
                "slug": "rest-api", "title": "REST API", "focus": "endpoints",
                "kind": "endpoint",
            },
        ],
    }
    _CHILD_IDS = ["functional__auth", "functional__billing", "technical__rest-api"]

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
            id = "prod_units"
        return P()

    def _patch_units_flow(self, monkeypatch, fake_repo_dir, orchestrator=None,
                          builder=None, notes_root=None):
        TestDocgenAgentFlow._patch_flow(
            object(), monkeypatch, fake_repo_dir,
            orchestrator=orchestrator, builder=builder,
        )

        async def fake_decompose(chat, brief, sections_list, hints):
            return {
                sid: [dict(item) for item in items]
                for sid, items in self._DECOMP.items()
            }

        monkeypatch.setattr(cb, "_decompose_sections", fake_decompose)
        if notes_root is not None:
            monkeypatch.setattr(cb, "_notes_dir_for", lambda cid: str(notes_root))

    @staticmethod
    def _texts(parent_body="PARENT {sid} citing `main.py`.",
               child_body="CHILD {uid} citing `main.py`.", include_children=True):
        texts = {sid: parent_body.format(sid=sid) for sid in cb.SECTION_ORDER}
        if include_children:
            texts.update({
                uid: child_body.format(uid=uid) for uid in TestUnitsAgentFlow._CHILD_IDS
            })
        return texts

    def test_units_flow_generates_children_before_parent(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        orchestrator = _FakeOrchestrator(self._texts())
        captured: Dict[str, Any] = {}

        def capture_builder(chat, specs, system_prompt):
            captured["specs"] = specs
            captured["system_prompt"] = system_prompt
            return orchestrator

        self._patch_units_flow(monkeypatch, fake_repo_dir, builder=capture_builder)

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        # Children dispatched BEFORE their parent, sections canonical.
        names = [s["name"] for s in captured["specs"]]
        expected = [
            "section-functional__auth", "section-functional__billing",
            "section-functional",
            "section-technical__rest-api", "section-technical",
        ]
        for name in expected:
            assert name in names
        assert names.index("section-functional__billing") < names.index("section-functional")
        assert names.index("section-technical__rest-api") < names.index("section-technical")

        # Pages: parents + children with family links.
        assert len(fake_artifact.pages) == 7 + 3
        child_page = fake_artifact.pages["page_technical__rest-api"]
        assert child_page["parent"] == "page_technical"
        assert child_page["provenance"]["prompt_file"] == "docgen_subpages.md"
        assert child_page["provenance"]["subpage"] == {
            "kind": "endpoint", "focus": "endpoints",
        }
        assert (
            fake_artifact.pages["page_functional"]["provenance"]["prompt_file"]
            == "docgen_sections.md"
        )
        # Markdown nests children under the parent heading.
        assert "### Authentication" in result
        assert "### REST API" in result
        assert "CHILD functional__auth" in result
        # The orchestrator prompt lists units with subpage markers.
        assert "subpage of `functional`" in captured["system_prompt"]

    def test_units_flow_progress_reports_unit_counters(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        orchestrator = _FakeOrchestrator(self._texts())
        self._patch_units_flow(monkeypatch, fake_repo_dir, orchestrator=orchestrator)
        events, progress = _collector()

        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product, progress=progress))

        child_finishes = [e for e in events if "unit_done" in e]
        assert {e["unit_done"] for e in child_finishes} == set(self._CHILD_IDS)
        assert all(e["units_total"] == 2 for e in child_finishes if e["unit_done"].startswith("functional"))
        # Parents still report section_done (7 sections, no child inflation).
        section_dones = [e["section_done"] for e in events if "section_done" in e]
        assert section_dones == cb.SECTION_ORDER

    def test_units_flow_writes_summary_notes(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch, tmp_path
    ):
        orchestrator = _FakeOrchestrator(self._texts())
        notes_root = tmp_path / "notes"
        self._patch_units_flow(
            monkeypatch, fake_repo_dir, orchestrator=orchestrator,
            notes_root=notes_root,
        )

        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        stored = os.listdir(str(notes_root))
        assert "repo_brief.md" in stored
        assert "decomposition.json" in stored
        assert "summary_functional__auth.md" in stored
        assert "summary_overview.md" in stored

    def test_units_full_reuse_skips_everything(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        """Second run with identical sources AND identical decomposition:
        parents AND children reused verbatim — no orchestrator call."""
        self._patch_units_flow(
            monkeypatch, fake_repo_dir,
            orchestrator=_FakeOrchestrator(self._texts()),
        )
        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        orchestrator2 = _FakeOrchestrator(
            self._texts(parent_body="MUTATED {sid}", child_body="MUTATED {uid}")
        )
        monkeypatch.setattr(
            cb, "_build_orchestrator_agent",
            lambda *a, **kw: orchestrator2,
        )

        result = asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        assert orchestrator2.calls == 0  # full reuse incl. children
        assert "CHILD functional__auth" in result
        assert "MUTATED" not in result
        prov = fake_artifact.pages["page_functional__auth"]["provenance"]
        assert prov["regen"] == "reused-unchanged"

    def test_reused_parent_keeps_children_and_drop_applies_on_regen(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        """A REUSED parent keeps its stored children verbatim (the new
        decomposition is not re-planned); when the parent's contract changes
        it regenerates, the fresh decomposition runs, and a dropped child
        page disappears while its surviving sibling is reused via pass B."""
        self._patch_units_flow(
            monkeypatch, fake_repo_dir,
            orchestrator=_FakeOrchestrator(self._texts()),
        )
        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        # --- run 2a: decomposition drops billing, but the parent is reused
        # (sources unchanged) → children stay exactly as stored.
        monkeypatch.setattr(
            cb, "_decompose_sections",
            _async_return({
                "functional": [dict(self._DECOMP["functional"][0])],
                "technical": [dict(self._DECOMP["technical"][0])],
            }),
        )
        orchestrator_unchanged = _FakeOrchestrator(self._texts(parent_body="MUT {sid}"))
        monkeypatch.setattr(
            cb, "_build_orchestrator_agent",
            lambda *a, **kw: orchestrator_unchanged,
        )

        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        assert orchestrator_unchanged.calls == 0  # full reuse again
        assert "page_functional__billing" in fake_artifact.pages
        assert "CHILD functional__billing" in fake_artifact.pages["page_functional__billing"]["content"]

        # --- run 2b: the functional CONTRACT changes → the parent regenerates
        # → the fresh (billing-less) decomposition applies → billing vanishes,
        # the surviving sibling is fingerprint-reused, the parent rewrites.
        real_instruction = cb._section_instruction

        def changed_instruction(sid):
            if sid == "functional":
                return real_instruction(sid) + "\nUpdated contract body."
            return real_instruction(sid)

        monkeypatch.setattr(cb, "_section_instruction", changed_instruction)
        orchestrator2 = _FakeOrchestrator(
            self._texts(
                parent_body="REGEN PARENT {sid} citing `main.py`.",
                child_body="SHOULD NOT APPEAR {uid}.",
            )
        )
        monkeypatch.setattr(
            cb, "_build_orchestrator_agent",
            lambda *a, **kw: orchestrator2,
        )

        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        # Only the functional parent was missing → dispatched exactly once.
        assert orchestrator2.calls == 1
        assert "REGEN PARENT functional" in fake_artifact.pages["page_functional"]["content"]
        # The dropped child's page is gone.
        assert "page_functional__billing" not in fake_artifact.pages
        # The surviving sibling was REUSED (pass B): stored text, not the
        # orchestrator's fresh mutation.
        auth_page = fake_artifact.pages["page_functional__auth"]
        assert "CHILD functional__auth" in auth_page["content"]
        assert "SHOULD NOT APPEAR" not in auth_page["content"]
        assert auth_page["provenance"]["regen"] == "reused-unchanged"
        # Untouched family kept as-is.
        assert "CHILD technical__rest-api" in (
            fake_artifact.pages["page_technical__rest-api"]["content"]
        )

    def test_placeholder_child_recovers_next_run(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        """A child the orchestrator missed (→ placeholder, no source files)
        cannot be reused and recovers on the next run."""
        # Child texts absent from the orchestrator transcript.
        self._patch_units_flow(
            monkeypatch, fake_repo_dir,
            orchestrator=_FakeOrchestrator(self._texts(include_children=False)),
        )
        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        child_page = fake_artifact.pages["page_functional__auth"]
        assert child_page["content"].strip() == cb._SECTION_UNAVAILABLE_PLACEHOLDER

        orchestrator2 = _FakeOrchestrator(
            self._texts(child_body="RECOVERED {uid} citing `main.py`.")
        )
        monkeypatch.setattr(
            cb, "_build_orchestrator_agent",
            lambda *a, **kw: orchestrator2,
        )

        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        recovered = fake_artifact.pages["page_functional__auth"]
        assert "RECOVERED" in recovered["content"]
        assert recovered["provenance"]["regen"] == "generated"

    def test_family_opener_dedup_reports_child_provenance(
        self, fake_artifact, fake_product, fake_repo_dir, monkeypatch
    ):
        """Identical openers across sibling subpages land in pass 2 (family)
        and are reported on the CHILD's provenance."""
        texts = self._texts(
            parent_body="Same opening line citing `main.py`. Parent {sid}.",
            child_body="Same opening line citing `main.py`. Child {uid}.",
        )
        self._patch_units_flow(
            monkeypatch, fake_repo_dir,
            orchestrator=_FakeOrchestrator(texts),
        )

        asyncio.run(cb.generate_codebase_docs(fake_artifact, fake_product))

        billing_prov = fake_artifact.pages["page_functional__billing"]["provenance"]
        assert "opener_duplicate" in billing_prov
        assert billing_prov["opener_duplicate"]["similar_to"] in {
            "functional", "functional__auth",
        }


def _async_return(value):
    async def _inner(*a, **kw):
        return value
    return _inner
