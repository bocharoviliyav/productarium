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
  file -> False, docgen_sections.md re-parses SECTION_PROMPTS).
- ``_parse_section_blocks`` (block parsing, dedup, malformed input).
- ``SECTION_PROMPTS`` registry (all 7 sections present, non-empty, raw — not
  wrapped with the language block).
- ``PROMPT_FILES`` registry completeness (exact match with the *.md files in
  EACH of refs/prompts/en and refs/prompts/ru — one file per prompt per
  language; the ru/en inventories must match).
- ``get_generation_language`` (admin setting > lang.json default; invalid
  stored value ignored) + ``prompts_dir``/``_resolve_prompt_path`` (language
  dirs with en fallback).
- Module-level constants (``LANGUAGE_INSTRUCTION``, ``DETAIL_LEVEL_COMPREHENSIVE``,
  ``LANGUAGE_NAMES``, ``_UNWRAPPED_PROMPTS``; the dead per-section WIKI_* and
  deep_research_*_iteration constants are GONE).
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
    PROMPT_LANGUAGES,
    PROMPTS_DIR,
    SECTION_PROMPTS,
    SUBPAGE_CONTRACTS,
    SUBPAGE_SECTION_IDS,
    SUBPAGE_TYPE_BY_SECTION,
    VERIFICATION_GUARD,
    WIKI_SECTIONS,
    _UNWRAPPED_PROMPTS,
    _default_language,
    _maybe_wrap,
    _parse_section_blocks,
    _parse_subpage_blocks,
    _resolve_prompt_path,
    _wrap_prompt,
    get_generation_language,
    get_section_title,
    load_prompt_file,
    prompts_dir,
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
            "qa",
            "datamodel",
        ]

    def test_cicd_title_mentions_sre(self):
        cicd = next(s for s in WIKI_SECTIONS if s["id"] == "cicd")
        assert "SRE" in cicd["title_ru"]
        assert "SRE" in cicd["title_en"]

    def test_subpage_sections_mapping(self):
        assert set(SUBPAGE_SECTION_IDS) == {"functional", "technical", "datamodel"}
        assert SUBPAGE_TYPE_BY_SECTION == {
            "functional": "functional_item",
            "technical": "technical_item",
            "datamodel": "datamodel_item",
        }
        for sid in SUBPAGE_SECTION_IDS:
            assert sid in {s["id"] for s in WIKI_SECTIONS}

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
        result = _maybe_wrap("product_summary.md", content)
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

    def test_docgen_pipeline_prompts_not_wrapped(self):
        content = "Contract text."
        for filename in (
            "docgen_sections.md",
            "docgen_subpages.md",
            "docgen_decomposer.md",
            "docgen_agent_system.md",
            "docgen_agent_section.md",
            "docgen_router.md",
            "docgen_orchestrator.md",
        ):
            assert _maybe_wrap(filename, content) == content


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
        monkeypatch.setattr(api.config, "configs", None)
        assert _default_language() == "ru"


# ---------------------------------------------------------------------------
# load_prompt_file
# ---------------------------------------------------------------------------

class TestLoadPromptFile:
    def test_loads_existing_file(self):
        # product_summary.md exists in refs/prompts/ and is a wrapped
        # generation prompt (not in _UNWRAPPED_PROMPTS).
        result = load_prompt_file("product_summary.md", "fallback")
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
        result = load_prompt_file("product_summary.md", "")
        # product_summary.md is a generation prompt -> should be wrapped
        assert "language" in result.lower()
        assert "detail_level" in result.lower()

    def test_docgen_sections_loaded_unwrapped(self):
        result = load_prompt_file("docgen_sections.md", "")
        # In _UNWRAPPED_PROMPTS: returned raw with its <section> blocks.
        assert "<language>" not in result
        assert '<section id="overview">' in result

    def test_mermaid_repair_not_wrapped(self):
        result = load_prompt_file("mermaid_repair.md", "fallback")
        assert "language" not in result.lower() or result == "fallback"

    def test_requested_language_dir_wins(self, monkeypatch, tmp_path):
        # Both language copies exist on disk: the requested language's copy
        # is loaded (never silently mixed).
        for lang, body in (("ru", "RU summary prompt."), ("en", "EN summary prompt.")):
            d = tmp_path / lang
            d.mkdir()
            (d / "product_summary.md").write_text(body, encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        assert "RU summary prompt." in load_prompt_file(
            "product_summary.md", "", language="ru"
        )
        assert "EN summary prompt." in load_prompt_file(
            "product_summary.md", "", language="en"
        )

    def test_missing_language_copy_falls_back_to_en(self, monkeypatch, tmp_path):
        # Only the English copy exists: a ru request reads it BEFORE falling
        # back to the in-code fallback text.
        d = tmp_path / "en"
        d.mkdir(parents=True)
        (d / "product_summary.md").write_text("EN only body.", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        result = load_prompt_file("product_summary.md", "the fallback", language="ru")
        assert "EN only body." in result
        assert "the fallback" not in result


# ---------------------------------------------------------------------------
# reload_prompt_file
# ---------------------------------------------------------------------------

class TestReloadPromptFile:
    @pytest.fixture(autouse=True)
    def _restore_prompt_state(self):
        """Snapshot + restore the module-level prompt state around each test.

        ``reload_prompt_file`` mutates module globals (and the in-place
        ``SECTION_PROMPTS`` registry) directly; monkeypatch restores
        ``PROMPTS_DIR`` but NOT those globals. Without this fixture the stub
        content leaks into later tests in this session (e.g. the registry
        tests below assert the real docgen_sections.md contracts).
        """
        saved_attrs = {
            attr: getattr(prompts_mod, attr)
            for attr in PROMPT_FILES.values()
            if hasattr(prompts_mod, attr)
        }
        saved_sections = dict(prompts_mod.SECTION_PROMPTS)
        saved_subpages = dict(prompts_mod.SUBPAGE_CONTRACTS)
        yield
        for attr, value in saved_attrs.items():
            setattr(prompts_mod, attr, value)
        prompts_mod.SECTION_PROMPTS.clear()
        prompts_mod.SECTION_PROMPTS.update(saved_sections)
        prompts_mod.SUBPAGE_CONTRACTS.clear()
        prompts_mod.SUBPAGE_CONTRACTS.update(saved_subpages)

    def test_unknown_file_returns_false(self):
        assert reload_prompt_file("nonexistent_file.md") is False

    def test_missing_file_on_disk_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        assert reload_prompt_file("product_summary.md") is False

    def test_reload_docgen_sections_reparses_registry(self, monkeypatch, tmp_path):
        lang_dir = tmp_path / "en"
        lang_dir.mkdir(parents=True)
        (lang_dir / "docgen_sections.md").write_text(
            '<section id="overview">\nNew overview contract.\n</section>\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        assert reload_prompt_file("docgen_sections.md", language="en") is True
        # Unwrapped prompt file: the constant holds the raw file text...
        assert "New overview contract." in prompts_mod.DOCGEN_SECTIONS_PROMPT
        # The registry is re-parsed in place...
        assert SECTION_PROMPTS["overview"] == "New overview contract."
        # ...and the OTHER canonical sections fall back to generic stubs.
        for sid in ("architecture", "datamodel"):
            assert sid in SECTION_PROMPTS and SECTION_PROMPTS[sid]

    def test_reload_docgen_subpages_reparses_registry(self, monkeypatch, tmp_path):
        lang_dir = tmp_path / "en"
        lang_dir.mkdir(parents=True)
        (lang_dir / "docgen_subpages.md").write_text(
            '<subpage id="functional_item">\nNew functional contract.\n</subpage>\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        assert reload_prompt_file("docgen_subpages.md", language="en") is True
        assert "New functional contract." in prompts_mod.DOCGEN_SUBPAGES_PROMPT
        assert SUBPAGE_CONTRACTS["functional_item"] == "New functional contract."
        # The other canonical subpage types fall back to generic stubs.
        for sp_type in ("technical_item", "datamodel_item"):
            assert sp_type in SUBPAGE_CONTRACTS and SUBPAGE_CONTRACTS[sp_type]

    def test_reload_non_section_prompt(self, monkeypatch, tmp_path):
        lang_dir = tmp_path / "en"
        lang_dir.mkdir(parents=True)
        (lang_dir / "product_summary.md").write_text(
            "Summary prompt.", encoding="utf-8"
        )
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        assert reload_prompt_file("product_summary.md", language="en") is True
        # Wrapped generation prompts carry the language block first — note
        # LANGUAGE_INSTRUCTION itself starts with a leading newline.
        wrapped = prompts_mod.PRODUCT_SUMMARY_PROMPT
        assert wrapped.lstrip().startswith("<language>")
        assert "Summary prompt." in wrapped

    def test_reload_verification_guard(self, monkeypatch, tmp_path):
        lang_dir = tmp_path / "en"
        lang_dir.mkdir(parents=True)
        (lang_dir / "_verification_guard.md").write_text("Guard rules.", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        assert reload_prompt_file("_verification_guard.md", language="en") is True
        assert prompts_mod.VERIFICATION_GUARD == "Guard rules."

    def test_reload_mermaid_repair(self, monkeypatch, tmp_path):
        lang_dir = tmp_path / "en"
        lang_dir.mkdir(parents=True)
        (lang_dir / "mermaid_repair.md").write_text("Repair rules.", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        assert reload_prompt_file("mermaid_repair.md", language="en") is True
        assert prompts_mod.MERMAID_REPAIR_PROMPT == "Repair rules."

    def test_reload_requested_language_wins(self, monkeypatch, tmp_path):
        # Both language copies exist: reload(language=…) reads that copy.
        for lang, body in (("ru", "RU repair."), ("en", "EN repair.")):
            d = tmp_path / lang
            d.mkdir()
            (d / "mermaid_repair.md").write_text(body, encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        assert reload_prompt_file("mermaid_repair.md", language="ru") is True
        assert prompts_mod.MERMAID_REPAIR_PROMPT == "RU repair."
        assert reload_prompt_file("mermaid_repair.md", language="en") is True
        assert prompts_mod.MERMAID_REPAIR_PROMPT == "EN repair."

    def test_reload_missing_lang_copy_falls_back_to_en(self, monkeypatch, tmp_path):
        lang_dir = tmp_path / "en"
        lang_dir.mkdir(parents=True)
        (lang_dir / "_verification_guard.md").write_text("EN guard.", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        assert reload_prompt_file("_verification_guard.md", language="ru") is True
        assert prompts_mod.VERIFICATION_GUARD == "EN guard."

    def test_reload_read_error_returns_false(self, monkeypatch, tmp_path):
        lang_dir = tmp_path / "en"
        lang_dir.mkdir(parents=True)
        bad_file = lang_dir / "product_summary.md"
        bad_file.write_text("content", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        original_open = open

        def fake_open(path, *args, **kwargs):
            if str(path).endswith("product_summary.md"):
                raise IOError("read error")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        assert reload_prompt_file("product_summary.md", language="en") is False


# ---------------------------------------------------------------------------
# _parse_section_blocks
# ---------------------------------------------------------------------------

class TestParseSectionBlocks:
    def test_parses_blocks(self):
        text = (
            '# Title\n\n'
            '<section id="overview">\nOverview contract.\n</section>\n\n'
            '<section id="datamodel">\nDatamodel contract.\n</section>\n'
        )
        parsed = _parse_section_blocks(text)
        assert parsed == {
            "overview": "Overview contract.",
            "datamodel": "Datamodel contract.",
        }

    def test_empty_and_malformed(self):
        assert _parse_section_blocks("") == {}
        assert _parse_section_blocks(None) == {}
        # Unclosed block -> no match.
        assert _parse_section_blocks('<section id="x">body') == {}
        # Empty body -> skipped.
        assert _parse_section_blocks('<section id="x">\n</section>') == {}

    def test_duplicate_id_keeps_last(self):
        text = (
            '<section id="overview">\nfirst\n</section>\n'
            '<section id="overview">\nsecond\n</section>\n'
        )
        assert _parse_section_blocks(text) == {"overview": "second"}

    def test_single_quotes_supported(self):
        text = "<section id='qa'>\nbody\n</section>"
        assert _parse_section_blocks(text) == {"qa": "body"}

    def test_surrounding_prose_ignored(self):
        text = "Intro prose.\n<section id=\"cicd\">\nbody\n</section>\nOutro prose."
        assert _parse_section_blocks(text) == {"cicd": "body"}


# ---------------------------------------------------------------------------
# _parse_subpage_blocks
# ---------------------------------------------------------------------------

class TestParseSubpageBlocks:
    def test_parses_blocks(self):
        text = (
            '# Title\n\n'
            '<subpage id="functional_item">\nFunctional contract.\n</subpage>\n\n'
            '<subpage id="datamodel_item">\nDatamodel contract.\n</subpage>\n'
        )
        parsed = _parse_subpage_blocks(text)
        assert parsed == {
            "functional_item": "Functional contract.",
            "datamodel_item": "Datamodel contract.",
        }

    def test_empty_and_malformed(self):
        assert _parse_subpage_blocks("") == {}
        assert _parse_subpage_blocks(None) == {}
        assert _parse_subpage_blocks('<subpage id="x">body') == {}
        assert _parse_subpage_blocks('<subpage id="x">\n</subpage>') == {}

    def test_duplicate_id_keeps_last(self):
        text = (
            '<subpage id="technical_item">\nfirst\n</subpage>\n'
            '<subpage id="technical_item">\nsecond\n</subpage>\n'
        )
        assert _parse_subpage_blocks(text) == {"technical_item": "second"}

    def test_section_blocks_not_matched(self):
        # The subpage parser must not pick up <section> blocks.
        text = '<section id="functional">\nbody\n</section>'
        assert _parse_subpage_blocks(text) == {}


# ---------------------------------------------------------------------------
# SUBPAGE_CONTRACTS registry
# ---------------------------------------------------------------------------

class TestSubpageContracts:
    def test_all_subpage_types_present(self):
        for sp_type in ("functional_item", "technical_item", "datamodel_item"):
            assert sp_type in SUBPAGE_CONTRACTS, f"{sp_type} missing"

    def test_subpage_prompts_non_empty_and_raw(self):
        for sp_type, body in SUBPAGE_CONTRACTS.items():
            assert body, f"{sp_type} contract is empty"
            assert "<language>" not in body, f"{sp_type} unexpectedly wrapped"
            assert "<detail_level>" not in body, f"{sp_type} unexpectedly wrapped"

    def test_real_file_overrides_stubs(self):
        for sp_type, body in SUBPAGE_CONTRACTS.items():
            assert len(body) > 200, (
                f"{sp_type} contract looks like the generic stub ({len(body)} chars)"
            )

    def test_technical_item_has_kind_variants(self):
        body = SUBPAGE_CONTRACTS["technical_item"]
        for kind in ("endpoint", "job", "integration", "reference"):
            assert f'kind "{kind}"' in body, f"technical_item missing kind {kind}"


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

    def test_section_prompts_are_raw_not_wrapped(self):
        # The parsed contracts are RAW bodies (the language block is added by
        # the docgen pipeline when it renders the subagent system prompt).
        for section_id, prompt in SECTION_PROMPTS.items():
            assert "<language>" not in prompt, f"{section_id} unexpectedly wrapped"
            assert "<detail_level>" not in prompt, f"{section_id} unexpectedly wrapped"

    def test_real_file_overrides_stubs(self):
        # With the real docgen_sections.md on disk, every contract must come
        # from the file (long bodies), not the one-line generic stub.
        for section_id, prompt in SECTION_PROMPTS.items():
            assert len(prompt) > 200, (
                f"{section_id} contract looks like the generic stub ({len(prompt)} chars)"
            )


# ---------------------------------------------------------------------------
# Prompt BODY content validation
#
# The tests above assert on the loader's STRUCTURE. These tests assert on the
# prompt BODIES themselves: that each generation prompt carries the
# {placeholder} tokens the substitution paths (docgen scaffolding,
# expert _build_prompt, deep research, database RE) actually fill in, that no
# prompt is truncated to a stub (min length), and that the language
# instruction renders for every supported language.
# ---------------------------------------------------------------------------

class TestPromptContentValidation:
    # Required placeholders per prompt filename, as actually substituted by
    # api.docgen.codebase (str.replace slots), api.expert.prompt._build_prompt
    # and the deep research / database flows.
    REQUIRED_PLACEHOLDERS = {
        "docgen_router.md": ["{repo_brief}", "{sections_list}"],
        "docgen_orchestrator.md": [
            "{repo_name}", "{sections_list}", "{reused_sections}",
        ],
        "docgen_agent_system.md": ["{language_name}"],
        "docgen_agent_section.md": [
            "{repo_url}", "{repo_name}", "{section_id}", "{section_title}",
            "{repo_brief}", "{sections_list}", "{section_hints}",
            "{siblings_list}", "{section_instruction}",
        ],
        "docgen_decomposer.md": [
            "{repo_brief}", "{sections_list}", "{section_hints}",
        ],
        "expert_agent_system.md": ["{product_name}", "{language_name}"],
        "expert_agent_doc.md": ["{product_name}", "{language_name}"],
        "deep_research_planner.md": [
            "{query}", "{product_name}", "{language_name}",
        ],
        "deep_research_researcher.md": [
            "{plan}", "{product_name}", "{language_name}",
        ],
        "deep_research_synthesizer.md": [
            "{query}", "{product_name}", "{language_name}",
        ],
        "database_doc.md": [
            "{database_name}", "{dsn_masked}", "{schema_dump}", "{skeleton}",
            "{language_name}",
        ],
    }

    def _raw_body(self, filename: str, language: str = "en") -> str:
        """Read the raw, unwrapped body from refs/prompts/<language>/<filename>."""
        path = os.path.join(PROMPTS_DIR, language, filename)
        assert os.path.exists(path), f"prompt file missing: {filename}"
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def test_prompts_contain_required_placeholders(self):
        for filename, placeholders in self.REQUIRED_PLACEHOLDERS.items():
            body = self._raw_body(filename)
            for placeholder in placeholders:
                assert placeholder in body, (
                    f"{filename} missing required placeholder {placeholder}"
                )

    def test_docgen_agent_section_has_no_previous_content(self):
        # The previous_content slot was removed with the subagent redesign:
        # subagents never see sibling output (isolated contexts).
        body = self._raw_body("docgen_agent_section.md")
        assert "{previous_content}" not in body

    def test_docgen_prompts_meet_min_length(self):
        min_length = 100
        for filename in self.REQUIRED_PLACEHOLDERS:
            body = self._raw_body(filename)
            assert len(body) >= min_length, (
                f"{filename} is only {len(body)} chars (expected >= {min_length})"
            )

    def test_docgen_sections_has_all_section_blocks(self):
        body = self._raw_body("docgen_sections.md")
        for section in WIKI_SECTIONS:
            sid = section["id"]
            assert f'<section id="{sid}">' in body, f"{sid} block missing"
        # The LLD contract is gone for good.
        assert '<section id="lld">' not in body

    def test_docgen_subpages_has_all_subpage_blocks(self):
        body = self._raw_body("docgen_subpages.md")
        for sp_type in ("functional_item", "technical_item", "datamodel_item"):
            assert f'<subpage id="{sp_type}">' in body, f"{sp_type} block missing"

    def test_verification_guard_non_empty_and_has_provenance_rules(self):
        assert VERIFICATION_GUARD, "VERIFICATION_GUARD loaded empty"
        assert len(VERIFICATION_GUARD) >= 100
        guard_lower = VERIFICATION_GUARD.lower()
        assert "провенанс" in guard_lower or "provenance" in guard_lower
        assert "выдумыв" in guard_lower or "invent" in guard_lower

    def test_language_instruction_renders_for_all_supported_languages(self):
        from api.config import configs
        supported = configs.get("lang_config", {}).get("supported_languages", {})
        assert supported, "lang_config.supported_languages is empty"
        for lang_code in supported:
            assert lang_code in LANGUAGE_NAMES, (
                f"language {lang_code!r} in lang.json has no LANGUAGE_NAMES entry"
            )
            rendered = LANGUAGE_INSTRUCTION.format(language_name=LANGUAGE_NAMES[lang_code])
            assert "{language_name}" not in rendered
            assert LANGUAGE_NAMES[lang_code] in rendered

    def test_wrap_prompt_renders_for_all_supported_languages(self):
        from api.config import configs
        supported = configs.get("lang_config", {}).get("supported_languages", {})
        assert supported
        for lang_code in supported:
            wrapped = _wrap_prompt("BODY", lang_code)
            assert wrapped.endswith("BODY")
            assert DETAIL_LEVEL_COMPREHENSIVE in wrapped
            assert "{language_name}" not in wrapped


# ---------------------------------------------------------------------------
# PROMPT_FILES registry
# ---------------------------------------------------------------------------

class TestPromptFiles:
    @staticmethod
    def _disk_files(language: str) -> set:
        directory = os.path.join(PROMPTS_DIR, language)
        assert os.path.isdir(directory), f"missing language dir: {directory}"
        return {
            name for name in os.listdir(directory)
            if name.endswith(".md") and name != "README.md"
        }

    def test_registry_matches_files_on_disk_per_language(self):
        # THE inventory invariant: registry keys == the *.md files in EACH
        # language dir (refs/prompts/en, refs/prompts/ru — the parent only
        # keeps README.md). Adding/deleting a prompt file in either language
        # without updating the registry (or vice versa) fails here.
        for language in PROMPT_LANGUAGES:
            assert set(PROMPT_FILES) == self._disk_files(language), language

    def test_ru_and_en_inventories_match(self):
        # Translation invariant: every English prompt ships a Russian copy
        # and vice versa (the admin panel falls back to en when a copy is
        # missing, but the shipped tree is complete).
        assert self._disk_files("en") == self._disk_files("ru")

    def test_contains_docgen_pipeline_prompts(self):
        for filename in (
            "docgen_sections.md",
            "docgen_subpages.md",
            "docgen_decomposer.md",
            "docgen_agent_system.md",
            "docgen_agent_section.md",
            "docgen_router.md",
            "docgen_orchestrator.md",
            "docgen_judge.md",
        ):
            assert filename in PROMPT_FILES

    def test_contains_expert_prompts(self):
        assert "expert_agent_system.md" in PROMPT_FILES
        assert "expert_agent_doc.md" in PROMPT_FILES

    def test_contains_deep_research_prompts(self):
        assert "deep_research_planner.md" in PROMPT_FILES
        assert "deep_research_researcher.md" in PROMPT_FILES
        assert "deep_research_synthesizer.md" in PROMPT_FILES

    def test_dead_entries_removed(self):
        # Per-section prompts were consolidated into docgen_sections.md; the
        # deep_research_*_iteration prompts were replaced by planner/
        # researcher/synthesizer.
        for section in WIKI_SECTIONS:
            assert f"{section['id']}.md" not in PROMPT_FILES
        for dead in (
            "structure.md",
            "compact_generation.md",
            "documentation_doc.md",
            "testcase_doc.md",
            "knowledge_graph_extraction.md",
            "deep_research_first_iteration.md",
            "deep_research_intermediate_iteration.md",
            "deep_research_final_iteration.md",
        ):
            assert dead not in PROMPT_FILES

    def test_all_values_are_strings(self):
        for key, val in PROMPT_FILES.items():
            assert isinstance(val, str)
            assert val  # non-empty


# ---------------------------------------------------------------------------
# get_generation_language + prompts_dir / _resolve_prompt_path
# ---------------------------------------------------------------------------

class TestGenerationLanguage:
    def test_defaults_to_lang_json_default(self, isolated_db):
        # No admin override stored in the fresh settings store: the lang.json
        # default (ru) wins.
        assert get_generation_language() == "ru"

    def test_admin_setting_overrides(self, isolated_db):
        from api.config.settings import set_setting

        set_setting(prompts_mod.GENERATION_LANGUAGE_SETTING, "en")
        assert get_generation_language() == "en"

    def test_invalid_admin_setting_falls_back(self, isolated_db):
        from api.config.settings import set_setting

        set_setting(prompts_mod.GENERATION_LANGUAGE_SETTING, "klingon")
        assert get_generation_language() == "ru"


class TestPromptsDirHelpers:
    def test_prompts_dir_explicit_language(self):
        assert prompts_dir("en") == os.path.join(PROMPTS_DIR, "en")
        assert prompts_dir("ru") == os.path.join(PROMPTS_DIR, "ru")

    def test_prompts_dir_default_uses_active_language(self):
        active = get_generation_language()
        assert prompts_dir(None) == os.path.join(PROMPTS_DIR, active)
        # Unknown codes normalize to the active language too.
        assert prompts_dir("klingon") == os.path.join(PROMPTS_DIR, active)

    def test_resolve_prompt_path_lang_dir_first_en_fallback(self, monkeypatch, tmp_path):
        en_dir = tmp_path / "en"
        en_dir.mkdir(parents=True)
        (en_dir / "product_summary.md").write_text("EN body.", encoding="utf-8")
        monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", str(tmp_path))
        # ru copy missing -> the en copy path is returned...
        assert _resolve_prompt_path("product_summary.md", "ru") == str(
            en_dir / "product_summary.md"
        )
        # ...and when nothing exists anywhere, the primary (language-dir)
        # path is returned so callers can log a meaningful target.
        assert _resolve_prompt_path("ghost.md", "ru") == str(tmp_path / "ru" / "ghost.md")


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

    def test_docgen_constants_are_strings(self):
        for attr in (
            "DOCGEN_SECTIONS_PROMPT",
            "DOCGEN_SUBPAGES_PROMPT",
            "DOCGEN_DECOMPOSER_PROMPT",
            "DOCGEN_AGENT_SYSTEM_PROMPT",
            "DOCGEN_AGENT_SECTION_PROMPT",
            "DOCGEN_ROUTER_PROMPT",
            "DOCGEN_ORCHESTRATOR_PROMPT",
            "DEEP_RESEARCH_PLANNER_PROMPT",
            "DEEP_RESEARCH_RESEARCHER_PROMPT",
            "DEEP_RESEARCH_SYNTHESIZER_PROMPT",
            "DATABASE_DOC_PROMPT",
            "MERMAID_REPAIR_PROMPT",
            "VERIFICATION_GUARD",
        ):
            assert isinstance(getattr(prompts_mod, attr), str), attr

    def test_dead_constants_removed(self):
        # The per-section WIKI_* constants, the dead deep-research iteration
        # constants and the doc/testcase prompts are GONE.
        for attr in (
            "WIKI_OVERVIEW_PROMPT", "WIKI_ARCHITECTURE_PROMPT",
            "WIKI_FUNCTIONAL_PROMPT", "WIKI_TECHNICAL_PROMPT",
            "WIKI_CICD_PROMPT", "WIKI_LLD_PROMPT", "WIKI_DATAMODEL_PROMPT",
            "WIKI_STRUCTURE_PROMPT", "WIKI_COMPACT_GENERATION_PROMPT",
            "DEEP_RESEARCH_FIRST_ITERATION_PROMPT",
            "DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT",
            "DEEP_RESEARCH_FINAL_ITERATION_PROMPT",
            "DOCUMENTATION_DOC_PROMPT", "TESTCASE_DOC_PROMPT",
        ):
            assert not hasattr(prompts_mod, attr), f"dead constant still present: {attr}"
