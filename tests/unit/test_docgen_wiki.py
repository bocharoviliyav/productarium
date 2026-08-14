"""Unit tests for api.docgen.wiki (WikiGenerator + context).

Covers: WikiSectionType enum, WikiGenerator init, set_context, _format_prompt
variable substitution (str.replace), _get_prompt_for_section, generate_section
(no LLM -> returns prompt), generate_all_sections, get_section_prompt_for_frontend
dispatch, create_wiki_section_context.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.docgen.wiki import (
    WikiGenerator,
    WikiSectionContext,
    WikiSectionType,
    create_wiki_section_context,
)


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def sample_context():
    return WikiSectionContext(
        repo_url="https://github.com/owner/myapp",
        repo_name="myapp",
        repo_type="github",
        primary_language="Python",
        file_count=150,
        main_directories=["src", "tests", "docs"],
        project_structure="src/\n  main.py\ntests/\n  test_main.py",
        main_files=["main.py", "app.py", "config.py"],
        tech_stack={"language": "Python", "framework": "FastAPI"},
        config_files=["config.json", "settings.py"],
        cicd_files=[".github/workflows/ci.yml"],
        docker_files=["Dockerfile", "docker-compose.yml"],
        api_endpoints=[{"path": "/api/users", "method": "GET"}],
        databases=["PostgreSQL"],
        entities=["User", "Post"],
        modules=["auth", "users", "posts"],
    )


@pytest.fixture
def generator(sample_context):
    gen = WikiGenerator(model="qwen/test-model", language="ru")
    gen.set_context(sample_context)
    return gen


# ============================================================================
# WikiSectionType enum
# ============================================================================
class TestWikiSectionType:
    def test_section_order(self):
        assert len(WikiGenerator.SECTION_ORDER) == 7
        assert WikiGenerator.SECTION_ORDER[0] == WikiSectionType.OVERVIEW
        assert WikiGenerator.SECTION_ORDER[-1] == WikiSectionType.DATAMODEL

    def test_section_values(self):
        assert WikiSectionType.OVERVIEW.value == "overview"
        assert WikiSectionType.ARCHITECTURE.value == "architecture"
        assert WikiSectionType.FUNCTIONAL.value == "functional"
        assert WikiSectionType.TECHNICAL.value == "technical"
        assert WikiSectionType.CICD.value == "cicd"
        assert WikiSectionType.LLD.value == "lld"
        assert WikiSectionType.DATAMODEL.value == "datamodel"

    def test_section_names(self):
        names = WikiGenerator.SECTION_NAMES
        assert names[WikiSectionType.OVERVIEW] == "Общая информация"
        assert names[WikiSectionType.ARCHITECTURE] == "Системная архитектура"
        assert names[WikiSectionType.CICD] == "CI/CD"


# ============================================================================
# WikiGenerator init
# ============================================================================
class TestWikiGeneratorInit:
    def test_default_model(self):
        gen = WikiGenerator()
        assert gen.model == "qwen/qwen3.6-27b"
        assert gen.language == "ru"
        assert gen.generated_sections == {}

    def test_explicit_model(self):
        gen = WikiGenerator(model="custom/model", language="en")
        assert gen.model == "custom/model"
        assert gen.language == "en"

    def test_set_context(self, sample_context):
        gen = WikiGenerator()
        gen.set_context(sample_context)
        assert gen.context is sample_context


# ============================================================================
# _get_prompt_for_section
# ============================================================================
class TestGetPromptForSection:
    def test_returns_template_for_known_section(self, generator):
        prompt = generator._get_prompt_for_section(WikiSectionType.OVERVIEW)
        assert isinstance(prompt, str)
        # The real prompt template should contain some content
        assert len(prompt) > 0

    def test_returns_empty_for_unknown_section(self):
        gen = WikiGenerator()
        # Pass a value that is not in the map
        result = gen._get_prompt_for_section("unknown_section")
        assert result == ""


# ============================================================================
# _format_prompt (variable substitution via str.replace)
# ============================================================================
class TestFormatPrompt:
    def test_overview_substitutes_common_vars(self, generator, sample_context):
        prompt = generator._format_prompt(WikiSectionType.OVERVIEW, sample_context)
        assert "https://github.com/owner/myapp" in prompt
        assert "myapp" in prompt
        assert "github" in prompt
        assert "Python" in prompt
        assert "150" in prompt
        assert "src" in prompt

    def test_architecture_substitutes_vars(self, generator, sample_context):
        prompt = generator._format_prompt(WikiSectionType.ARCHITECTURE, sample_context)
        assert "main.py" in prompt
        assert "app.py" in prompt

    def test_functional_substitutes_vars(self, generator, sample_context):
        prompt = generator._format_prompt(WikiSectionType.FUNCTIONAL, sample_context)
        assert "auth" in prompt
        assert "users" in prompt
        # api_endpoints rendered as JSON
        assert "/api/users" in prompt

    def test_technical_substitutes_vars(self, generator, sample_context):
        prompt = generator._format_prompt(WikiSectionType.TECHNICAL, sample_context)
        assert "config.json" in prompt
        assert "FastAPI" in prompt

    def test_cicd_substitutes_vars(self, generator, sample_context):
        prompt = generator._format_prompt(WikiSectionType.CICD, sample_context)
        assert ".github/workflows/ci.yml" in prompt
        assert "Dockerfile" in prompt

    def test_lld_substitutes_vars(self, generator, sample_context):
        prompt = generator._format_prompt(WikiSectionType.LLD, sample_context)
        assert "auth" in prompt
        assert "/api/users" in prompt

    def test_datamodel_substitutes_vars(self, generator, sample_context):
        prompt = generator._format_prompt(WikiSectionType.DATAMODEL, sample_context)
        assert "PostgreSQL" in prompt
        assert "User" in prompt

    def test_previous_content_included(self, generator, sample_context):
        generator.generated_sections = {"overview": "Previous overview content here."}
        prompt = generator._format_prompt(WikiSectionType.ARCHITECTURE, sample_context)
        assert "Previous overview content here." in prompt
        # SECTION_NAMES is keyed by WikiSectionType enum, but generated_sections
        # uses string values; .get(key, key) falls back to the string key.
        assert "overview" in prompt

    def test_previous_content_empty_when_no_sections(self, generator, sample_context):
        prompt = generator._format_prompt(WikiSectionType.OVERVIEW, sample_context)
        # Should not contain the literal {previous_content} placeholder
        assert "{previous_content}" not in prompt

    def test_preserves_mermaid_braces(self, generator, sample_context):
        """str.replace should preserve literal braces in Mermaid/JSON examples."""
        prompt = generator._format_prompt(WikiSectionType.OVERVIEW, sample_context)
        # The prompt should not have been broken by str.format on braces
        assert "{previous_content}" not in prompt


# ============================================================================
# generate_section
# ============================================================================
class TestGenerateSection:
    def test_no_context_returns_error(self):
        gen = WikiGenerator()
        success, content = gen.generate_section(WikiSectionType.OVERVIEW)
        assert success is False
        assert "Context not set" in content

    def test_no_llm_returns_prompt(self, generator):
        success, content = gen_section = generator.generate_section(WikiSectionType.OVERVIEW)
        assert success is True
        assert len(content) > 0
        assert "https://github.com/owner/myapp" in content

    def test_with_llm_generator(self, generator):
        class FakeLLM:
            def __call__(self, prompt_kwargs=None):
                return "Generated section content"
        success, content = generator.generate_section(WikiSectionType.OVERVIEW, FakeLLM())
        assert success is True
        assert content == "Generated section content"
        assert generator.generated_sections["overview"] == "Generated section content"

    def test_llm_raises_returns_error(self, generator):
        class FakeLLM:
            def __call__(self, prompt_kwargs=None):
                raise RuntimeError("LLM error")
        success, content = generator.generate_section(WikiSectionType.OVERVIEW, FakeLLM())
        assert success is False
        assert "Error generating section" in content


# ============================================================================
# generate_all_sections
# ============================================================================
class TestGenerateAllSections:
    def test_no_context_returns_empty(self):
        gen = WikiGenerator()
        result = gen.generate_all_sections()
        assert result == {}

    def test_generates_all_seven_sections(self, generator):
        # generate_section stores into generated_sections only when an LLM is
        # provided; without one it returns (True, prompt) but doesn't persist.
        class FakeLLM:
            def __call__(self, prompt_kwargs=None):
                return "section content"
        result = generator.generate_all_sections(llm_generator=FakeLLM())
        assert len(result) == 7
        for section_type in WikiGenerator.SECTION_ORDER:
            assert section_type.value in result

    def test_callback_invoked(self, generator):
        calls = []
        def cb(section_type, success, content):
            calls.append((section_type.value, success))
        generator.generate_all_sections(section_callback=cb)
        assert len(calls) == 7
        assert all(s for _, s in calls)

    def test_callback_error_swallowed(self, generator):
        def bad_cb(section_type, success, content):
            raise RuntimeError("callback error")
        # Should not raise
        generator.generate_all_sections(section_callback=bad_cb)


# ============================================================================
# get_section_prompt_for_frontend
# ============================================================================
class TestGetSectionPromptForFrontend:
    def test_no_context_returns_error(self):
        gen = WikiGenerator()
        result = gen.get_section_prompt_for_frontend(WikiSectionType.OVERVIEW)
        assert result == "Error: Context not set"

    def test_overview_dispatch(self, generator):
        result = generator.get_section_prompt_for_frontend(WikiSectionType.OVERVIEW)
        assert "myapp" in result

    def test_each_section_dispatches(self, generator):
        for section_type in WikiGenerator.SECTION_ORDER:
            result = generator.get_section_prompt_for_frontend(section_type)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_unknown_section_type(self, generator):
        result = generator.get_section_prompt_for_frontend("unknown")
        assert result == "Unknown section type"


# ============================================================================
# create_wiki_section_context
# ============================================================================
class TestCreateWikiSectionContext:
    def test_builds_context_from_analysis(self):
        file_analysis = {
            "main_directories": ["src", "tests"],
            "main_files": ["main.py", "app.py"],
            "tech_stack": {"language": "Python"},
            "config_files": ["config.json"],
            "cicd_files": [".github/workflows/ci.yml"],
            "docker_files": ["Dockerfile"],
            "api_endpoints": [{"path": "/api/users", "method": "GET"}],
            "databases": ["PostgreSQL"],
            "entities": ["User"],
            "modules": ["auth"],
            "primary_language": "Python",
            "file_count": 100,
        }
        ctx = create_wiki_section_context(
            repo_url="https://github.com/owner/myapp",
            repo_type="github",
            file_tree="src/\n  main.py",
            readme="# My App",
            file_analysis=file_analysis,
        )
        assert ctx.repo_url == "https://github.com/owner/myapp"
        assert ctx.repo_name == "myapp"
        assert ctx.repo_type == "github"
        assert ctx.primary_language == "Python"
        assert ctx.file_count == 100
        assert ctx.main_directories == ["src", "tests"]
        assert ctx.main_files == ["main.py", "app.py"]
        assert ctx.tech_stack == {"language": "Python"}
        assert ctx.config_files == ["config.json"]
        assert ctx.project_structure == "src/\n  main.py"

    def test_repo_name_from_url_no_slash(self):
        ctx = create_wiki_section_context(
            repo_url="localrepo",
            repo_type="local",
            file_tree="",
            readme="",
            file_analysis={},
        )
        assert ctx.repo_name == "localrepo"

    def test_defaults_for_missing_keys(self):
        ctx = create_wiki_section_context(
            repo_url="https://github.com/o/repo",
            repo_type="github",
            file_tree="",
            readme="",
            file_analysis={},
        )
        assert ctx.primary_language == "unknown"
        assert ctx.file_count == 0
        assert ctx.main_directories == []
        assert ctx.main_files == []


# ============================================================================
# Per-section prompt builders (internal)
# ============================================================================
class TestSectionBuilders:
    def test_build_overview_prompt(self, generator):
        result = generator._build_overview_prompt("", "", None)
        assert "myapp" in result

    def test_build_architecture_prompt(self, generator):
        result = generator._build_architecture_prompt("", "", None)
        assert isinstance(result, str)

    def test_build_functional_prompt(self, generator):
        result = generator._build_functional_prompt("", "", None)
        assert isinstance(result, str)

    def test_build_technical_prompt(self, generator):
        result = generator._build_technical_prompt("", "", None)
        assert isinstance(result, str)

    def test_build_cicd_prompt(self, generator):
        result = generator._build_cicd_prompt("", "", None)
        assert isinstance(result, str)

    def test_build_lld_prompt(self, generator):
        result = generator._build_lld_prompt("", "", None)
        assert isinstance(result, str)

    def test_build_datamodel_prompt(self, generator):
        result = generator._build_datamodel_prompt("", "", None)
        assert isinstance(result, str)
