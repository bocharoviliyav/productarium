"""Unit tests for ``api.prompts`` (prompt loading + wrapping).

Covers:
- ``WIKI_SECTIONS`` structure + ``get_section_title`` (ru/en/unknown).
- ``_wrap_prompt`` (language block + detail block prepended).
- ``_maybe_wrap`` (wrapped prompts skipped, ``{language_name}`` content skipped,
  normal prompts wrapped).
- ``_default_language`` (from lang config, fallback ru).
- ``load_prompt_file`` (existing file loaded + wrapped, missing file -> fallback,
  fallback wrapped).
- ``reload_prompt_file`` (known file reloaded, unknown file -> False, missing
  file -> False, section prompt updates SECTION_PROMPTS).
- ``SECTION_PROMPTS`` registry (all 7 sections present, non-empty).
- ``PROMPT_FILES`` registry completeness.
- Module-level constants (``LANGUAGE_INSTRUCTION``, ``DETAIL_LEVEL_COMPREHENSIVE``,
  ``LANGUAGE_NAMES``, ``_UNWRAPPED_PROMPTS``).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.prompts as prompts_mod
from api.prompts import (
    DETAIL_LEVEL_COMPREHENSIVE,
    LANGUAGE_INSTRUCTION,
    LANGUAGE_NAMES,
    PROMPT_FILES,
    PROMPTS_DIR,
    SECTION_PROMPTS,
    VERIFICATION_GUARD,
    WIKI_SECTIONS,
    _UNWRAPPED_PROMPTS,
    _default_language,
    _maybe_wrap,
    _wrap_prompt,
    get_section_title,
    load_prompt_file,
    reload_prompt_file,
)


# ---------------------------------------------------------------------------
# WIKI_SECTIONS + get_section_title
# ---------------------------------------------------------------------------

class TestWikiSections:
    def test_has_seven_sections(self):
        assert len(WIKI_SECTIONS) == 7

    def test_section_ids(self):
        ids = [s["id"] for s in WIKI_SECTIONS]
        assert ids == [
            "overview",
            "architecture",
            "functional",
            "technical",
            "cicd",
            "lld",
            "datamodel",
        ]

    def test_each_section_has_ru_and_en_titles(self):
        for s in WIKI_SECTIONS:
            assert "title_ru" in s
            assert "title_en" in s
            assert s["title_ru"]
            assert s["title_en"]


class TestGetSectionTitle:
    def test_ru_title(self):
        assert get_section_title("overview", "ru") == "Общая информация"
        assert get_section_title("architecture", "ru") == "Системная архитектура"

    def test_en_title(self):
        assert get_section_title("overview", "en") == "Overview"
        assert get_section_title("architecture", "en") == "System Architecture"

    def test_default_ru(self):
        assert get_section_title("overview") == "Общая информация"

    def test_unknown_section_returns_id(self):
        assert get_section_title("nonexistent", "ru") == "nonexistent"
        assert get_section_title("nonexistent", "en") == "nonexistent"


# ---------------------------------------------------------------------------
# _wrap_prompt
# ---------------------------------------------------------------------------

class TestWrapPrompt:
    def test_wraps_with_language_and_detail(self):
        prompt = "Generate docs."
        result = _wrap_prompt(prompt, "ru")
        assert LANGUAGE_INSTRUCTION.format(language_name="Russian (Русский)") in result
        assert DETAIL_LEVEL_COMPREHENSIVE in result
        assert prompt in result
        # Language block comes before detail block comes before prompt
        assert result.index(LANGUAGE_INSTRUCTION.format(language_name="Russian (Русский)")) < result.index(prompt)

    def test_english_language(self):
        result = _wrap_prompt("test", "en")
        assert "English" in result

    def test_unknown_language_uses_raw_name(self):
        result = _wrap_prompt("test", "fr")
        assert "fr" in result

    def test_prompt_prepended_not_appended(self):
        prompt = "MY PROMPT"
        result = _wrap_prompt(prompt, "ru")
        # The prompt should be at the end, after the blocks
        assert result.endswith(prompt)


# ---------------------------------------------------------------------------
# _maybe_wrap
# ---------------------------------------------------------------------------

class TestMaybeWrap:
    def test_wraps_normal_prompt(self):
        content = "Generate documentation."
        result = _maybe_wrap("overview.md", content)
        # Should be wrapped (language + detail blocks prepended)
        assert "language" in result.lower()
        assert content in result

    def test_skips_unwrapped_prompt_filenames(self):
        content = "Expert system prompt."
        for filename in _UNWRAPPED_PROMPTS:
            result = _maybe_wrap(filename, content)
            assert result == content, f"{filename} should not be wrapped"

    def test_skips_content_with_language_name_placeholder(self):
        content = "Write in {language_name}."
        result = _maybe_wrap("custom.md", content)
        assert result == content

    def test_verification_guard_not_wrapped(self):
        content = "Anti-hallucination rules."
        result = _maybe_wrap("_verification_guard.md", content)
        assert result == content

    def test_mermaid_repair_not_wrapped(self):
        content = "Fix the mermaid diagram."
        result = _maybe_wrap("mermaid_repair.md", content)
        assert result == content


# ---------------------------------------------------------------------------
# _default_language
# ---------------------------------------------------------------------------

class TestDefaultLanguage:
    def test_returns_from_config(self):
        lang = _default_language()
        assert lang in ("ru", "en")

    def test_fallback_ru_on_exception(self, monkeypatch):
        # Force an exception in the config import
        import api.config
        original = api.config.configs
        monkeypatch.setattr(api.config, "configs", None)
        assert _default_language() == "ru"


# ---------------------------------------------------------------------------
# load_prompt_file
# ---------------------------------------------------------------------------

class TestLoadPromptFile:
    def test_loads_existing_file(self):
        # overview.md exists in refs/prompts/
        result = load_prompt_file("overview.md", "fallback")
        # Should not be the fallback (the file exists)
        assert result != "fallback"
        assert len(result) > 0

    def test_missing_file_uses_fallback(self):
        result = load_prompt_file("nonexistent_prompt.md", "my fallback")
        # Fallback should be wrapped (it's a normal prompt, not in _UNWRAPPED_PROMPTS)
        assert "my fallback" in result
        assert "language" in result.lower()

    def test_missing_file_fallback_for_unwrapped(self, monkeypatch, tmp_path):
        # Point PROMPTS_DIR to an empty dir so the file is "missing"
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        fallback = "expert fallback"
        result = load_prompt_file("expert_agent_system.md", fallback)
        # expert_agent_system.md is in _UNWRAPPED_PROMPTS, so fallback not wrapped
        assert result == fallback

    def test_existing_file_is_wrapped(self):
        result = load_prompt_file("overview.md", "")
        # overview.md is a generation prompt -> should be wrapped
        assert "language" in result.lower()
        assert "detail_level" in result.lower()

    def test_mermaid_repair_not_wrapped(self):
        result = load_prompt_file("mermaid_repair.md", "fallback")
        # mermaid_repair.md is in _UNWRAPPED_PROMPTS
        # If the file exists, content is returned as-is (not wrapped)
        assert "language" not in result.lower() or result == "fallback"


# ---------------------------------------------------------------------------
# reload_prompt_file
# ---------------------------------------------------------------------------

class TestReloadPromptFile:
    def test_unknown_file_returns_false(self):
        assert reload_prompt_file("nonexistent_file.md") is False

    def test_missing_file_on_disk_returns_false(self, monkeypatch, tmp_path):
        # Override PROMPTS_DIR to a temp dir where the file doesn't exist
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        assert reload_prompt_file("overview.md") is False

    def test_reload_known_section_prompt(self, monkeypatch, tmp_path):
        # Create a temp prompts dir with a section file
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "overview.md").write_text("New overview content.", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(prompts_dir))
        result = reload_prompt_file("overview.md")
        assert result is True
        assert prompts_mod.WIKI_OVERVIEW_PROMPT == _maybe_wrap("overview.md", "New overview content.")
        assert SECTION_PROMPTS["overview"] == _maybe_wrap("overview.md", "New overview content.")

    def test_reload_non_section_prompt(self, monkeypatch, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "structure.md").write_text("Structure prompt.", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(prompts_dir))
        result = reload_prompt_file("structure.md")
        assert result is True
        assert prompts_mod.WIKI_STRUCTURE_PROMPT == _maybe_wrap("structure.md", "Structure prompt.")

    def test_reload_verification_guard(self, monkeypatch, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "_verification_guard.md").write_text("Guard rules.", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(prompts_dir))
        result = reload_prompt_file("_verification_guard.md")
        assert result is True
        assert prompts_mod.VERIFICATION_GUARD == "Guard rules."

    def test_reload_mermaid_repair(self, monkeypatch, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "mermaid_repair.md").write_text("Repair rules.", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(prompts_dir))
        result = reload_prompt_file("mermaid_repair.md")
        assert result is True
        assert prompts_mod.MERMAID_REPAIR_PROMPT == "Repair rules."

    def test_reload_read_error_returns_false(self, monkeypatch, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        bad_file = prompts_dir / "overview.md"
        bad_file.write_text("content", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(prompts_dir))
        # Mock open to raise
        original_open = open

        def fake_open(path, *args, **kwargs):
            if str(path).endswith("overview.md"):
                raise IOError("read error")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        assert reload_prompt_file("overview.md") is False


# ---------------------------------------------------------------------------
# SECTION_PROMPTS registry
# ---------------------------------------------------------------------------

class TestSectionPrompts:
    def test_all_sections_present(self):
        for section in WIKI_SECTIONS:
            assert section["id"] in SECTION_PROMPTS, f"{section['id']} missing from SECTION_PROMPTS"

    def test_all_prompts_non_empty(self):
        for section_id, prompt in SECTION_PROMPTS.items():
            assert prompt, f"{section_id} prompt is empty"

    def test_section_prompts_are_wrapped(self):
        # All section prompts are generation prompts -> should be wrapped
        for section_id, prompt in SECTION_PROMPTS.items():
            assert "language" in prompt.lower(), f"{section_id} not wrapped"


# ---------------------------------------------------------------------------
# PROMPT_FILES registry
# ---------------------------------------------------------------------------

class TestPromptFiles:
    def test_all_files_mapped(self):
        assert len(PROMPT_FILES) >= 18

    def test_contains_all_sections(self):
        for section in WIKI_SECTIONS:
            assert f"{section['id']}.md" in PROMPT_FILES

    def test_contains_expert_prompts(self):
        assert "expert_agent_system.md" in PROMPT_FILES
        assert "expert_agent_doc.md" in PROMPT_FILES

    def test_contains_deep_research_prompts(self):
        assert "deep_research_first_iteration.md" in PROMPT_FILES
        assert "deep_research_final_iteration.md" in PROMPT_FILES
        assert "deep_research_intermediate_iteration.md" in PROMPT_FILES

    def test_contains_doc_prompts(self):
        assert "openapi_doc.md" in PROMPT_FILES
        assert "asyncapi_doc.md" in PROMPT_FILES
        assert "documentation_doc.md" in PROMPT_FILES

    def test_all_values_are_strings(self):
        for key, val in PROMPT_FILES.items():
            assert isinstance(val, str)
            assert val  # non-empty


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_prompts_dir_exists(self):
        assert os.path.isdir(PROMPTS_DIR)

    def test_language_names(self):
        assert LANGUAGE_NAMES["ru"] == "Russian (Русский)"
        assert LANGUAGE_NAMES["en"] == "English"

    def test_language_instruction_has_placeholder(self):
        assert "{language_name}" in LANGUAGE_INSTRUCTION

    def test_detail_level_has_comprehensive(self):
        assert "COMPREHENSIVE" in DETAIL_LEVEL_COMPREHENSIVE

    def test_unwrapped_prompts_is_frozenset(self):
        assert isinstance(_UNWRAPPED_PROMPTS, frozenset)

    def test_verification_guard_loaded(self):
        # VERIFICATION_GUARD is loaded at import time
        assert isinstance(VERIFICATION_GUARD, str)

    def test_all_prompt_constants_are_strings(self):
        assert isinstance(prompts_mod.WIKI_OVERVIEW_PROMPT, str)
        assert isinstance(prompts_mod.WIKI_ARCHITECTURE_PROMPT, str)
        assert isinstance(prompts_mod.WIKI_FUNCTIONAL_PROMPT, str)
        assert isinstance(prompts_mod.WIKI_TECHNICAL_PROMPT, str)
        assert isinstance(prompts_mod.WIKI_CICD_PROMPT, str)
        assert isinstance(prompts_mod.WIKI_LLD_PROMPT, str)
        assert isinstance(prompts_mod.WIKI_DATAMODEL_PROMPT, str)
        assert isinstance(prompts_mod.WIKI_STRUCTURE_PROMPT, str)
        assert isinstance(prompts_mod.WIKI_COMPACT_GENERATION_PROMPT, str)
        assert isinstance(prompts_mod.DEEP_RESEARCH_FIRST_ITERATION_PROMPT, str)
        assert isinstance(prompts_mod.DEEP_RESEARCH_FINAL_ITERATION_PROMPT, str)
        assert isinstance(prompts_mod.DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT, str)
        assert isinstance(prompts_mod.MERMAID_REPAIR_PROMPT, str)
