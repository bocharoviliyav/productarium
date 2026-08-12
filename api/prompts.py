"""Module containing all prompts used in the DeepWiki project.

Prompt BODIES live externally in ``refs/prompts/*.md`` and are loaded at module
import time by :func:`load_prompt_file`. This module keeps only the abstract,
code-level scaffolding: the canonical wiki section list, language/detail-level
helpers, the loader, and the :data:`SECTION_PROMPTS` registry that maps section
ids to the loaded templates.

Each prompt constant below is a short fallback used only if the corresponding
``refs/prompts/<name>.md`` file is missing. Do NOT inline prompt bodies here —
edit the matching ``refs/prompts/<name>.md`` file instead.
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# WIKI SECTION DEFINITIONS
# ============================================================================

# Canonical 7-section wiki structure
WIKI_SECTIONS = [
    {"id": "overview", "title_ru": "Общая информация", "title_en": "Overview"},
    {"id": "architecture", "title_ru": "Системная архитектура", "title_en": "System Architecture"},
    {"id": "functional", "title_ru": "Функциональное описание", "title_en": "Functional Description"},
    {"id": "technical", "title_ru": "Технические детали", "title_en": "Technical Details"},
    {"id": "cicd", "title_ru": "CI/CD", "title_en": "CI/CD"},
    {"id": "lld", "title_ru": "LLD (Low Level Design)", "title_en": "LLD (Low Level Design)"},
    {"id": "datamodel", "title_ru": "Модель данных", "title_en": "Data Model"},
]


def get_section_title(section_id: str, language: str = "ru") -> str:
    """Get localized section title by ID."""
    for s in WIKI_SECTIONS:
        if s["id"] == section_id:
            return s["title_ru"] if language == "ru" else s["title_en"]
    return section_id


LANGUAGE_INSTRUCTION = """\n<language>
IMPORTANT: You MUST write your ENTIRE response in {language_name}.
Technical terms, file names, code identifiers, and API endpoints should remain in English.
All other text (descriptions, explanations, headings, table content) MUST be in {language_name}.
</language>\n"""

DETAIL_LEVEL_COMPREHENSIVE = """\n<detail_level>
Mode: COMPREHENSIVE — provide maximum detail.
- Include ALL Mermaid diagrams (C4, sequence, ER, flowchart)
- Include code examples with file paths
- Provide detailed tables with all fields
- Write thorough descriptions for every section
- Priority: quality and completeness over brevity
</detail_level>\n"""

DETAIL_LEVEL_CONCISE = """\n<detail_level>
Mode: CONCISE — provide a brief summary.
- Include only the most important Mermaid diagrams (1-2 per section)
- Omit code examples, keep only references to files
- Use shorter tables with key fields only
- Write 2-3 sentence descriptions per subsection
- Priority: brevity and clarity over exhaustive detail
</detail_level>\n"""

LANGUAGE_NAMES = {
    "ru": "Russian (Русский)",
    "en": "English",
    "ja": "Japanese (日本語)",
    "zh": "Chinese (中文)",
    "es": "Spanish (Español)",
    "kr": "Korean (한국어)",
    "vi": "Vietnamese (Tiếng Việt)",
    "pt-br": "Brazilian Portuguese (Português Brasileiro)",
    "fr": "French (Français)",
}


def wrap_prompt(prompt: str, language: str = "ru", comprehensive: bool = True) -> str:
    """Wrap any wiki section prompt with language and detail-level instructions."""
    language_name = LANGUAGE_NAMES.get(language, language)
    lang_block = LANGUAGE_INSTRUCTION.format(language_name=language_name)
    detail_block = DETAIL_LEVEL_COMPREHENSIVE if comprehensive else DETAIL_LEVEL_CONCISE
    return lang_block + detail_block + prompt


# Forward declaration; populated by the dynamic-load block below.
SECTION_PROMPTS: Dict[str, str] = {}

# ============================================================================
# PROMPT TEMPLATE FALLBACKS
# ----------------------------------------------------------------------------
# The real prompt bodies are stored externally in refs/prompts/*.md and loaded
# by load_prompt_file() at the bottom of this module. These constants are tiny
# fallbacks used only when a refs file is missing. They MUST be defined before
# the load calls (which reference them as fallbacks).
# ============================================================================

WIKI_OVERVIEW_PROMPT = ""
WIKI_ARCHITECTURE_PROMPT = ""
WIKI_FUNCTIONAL_PROMPT = ""
WIKI_TECHNICAL_PROMPT = ""
WIKI_CICD_PROMPT = ""
WIKI_LLD_PROMPT = ""
WIKI_DATAMODEL_PROMPT = ""
WIKI_STRUCTURE_PROMPT = ""
WIKI_COMPACT_GENERATION_PROMPT = ""
DEEP_RESEARCH_FIRST_ITERATION_PROMPT = ""
DEEP_RESEARCH_FINAL_ITERATION_PROMPT = ""
DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT = ""
MERMAID_REPAIR_PROMPT = ""
# Unified verification guard (anti-hallucination + citation + no-line-numbers
# rules). Appended to every generation prompt via VERIFICATION_GUARD. Loaded
# from refs/prompts/_verification_guard.md; hot-reloadable via the admin panel.
VERIFICATION_GUARD = ""

# ============================================================================
# DYNAMIC PROMPT LOADING FROM REFS
# ============================================================================

PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "refs",
    "prompts",
)


def load_prompt_file(filename: str, fallback: str) -> str:
    """Load a prompt template from ``refs/prompts/<filename>``.

    Returns the file content (stripped) if it exists, otherwise ``fallback``.
    This is the single source of truth for prompt bodies — code references the
    loaded constants rather than inline strings.
    """
    try:
        path = os.path.join(PROMPTS_DIR, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as e:
        logger.warning(f"Error loading prompt from {filename}: {e}")
    return fallback


# Registry mapping filename -> module attribute name for all prompts loaded
# from ``refs/prompts/*.md``. Used by both the load block below and
# ``reload_prompt_file`` so an edit via the admin panel can hot-reload a
# single prompt without restarting the process.
#
# Keys are filenames (relative to ``refs/prompts/``); values are the module
# attribute name that holds the loaded prompt text.
#
# Note: expert_agent_system.md and expert_agent_doc.md are consumed by
# ``api.expert.prompt`` (which loads them via ``load_prompt_file`` into its
# own ``EXPERT_SYSTEM_PROMPT`` / ``EXPERT_DOC_PROMPT`` constants). They are
# included here so ``reload_prompt_file`` can also refresh them via
# ``importlib.reload(api.expert.prompt)`` (best-effort, optional).
PROMPT_FILES: Dict[str, str] = {
    "overview.md": "WIKI_OVERVIEW_PROMPT",
    "architecture.md": "WIKI_ARCHITECTURE_PROMPT",
    "functional.md": "WIKI_FUNCTIONAL_PROMPT",
    "technical.md": "WIKI_TECHNICAL_PROMPT",
    "cicd.md": "WIKI_CICD_PROMPT",
    "lld.md": "WIKI_LLD_PROMPT",
    "datamodel.md": "WIKI_DATAMODEL_PROMPT",
    "structure.md": "WIKI_STRUCTURE_PROMPT",
    "compact_generation.md": "WIKI_COMPACT_GENERATION_PROMPT",
    "deep_research_first_iteration.md": "DEEP_RESEARCH_FIRST_ITERATION_PROMPT",
    "deep_research_final_iteration.md": "DEEP_RESEARCH_FINAL_ITERATION_PROMPT",
    "deep_research_intermediate_iteration.md": "DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT",
    "expert_agent_system.md": "EXPERT_SYSTEM_PROMPT",
    "expert_agent_doc.md": "EXPERT_DOC_PROMPT",
    "product_summary.md": "PRODUCT_SUMMARY_PROMPT",
    "openapi_doc.md": "OPENAPI_DOC_PROMPT",
    "asyncapi_doc.md": "ASYNCAPI_DOC_PROMPT",
    "testcase_doc.md": "TESTCASE_DOC_PROMPT",
    "documentation_doc.md": "DOCUMENTATION_DOC_PROMPT",
    "mermaid_repair.md": "MERMAID_REPAIR_PROMPT",
    "_verification_guard.md": "VERIFICATION_GUARD",
}


def reload_prompt_file(filename: str) -> bool:
    """Re-read a prompt file from disk and update the in-memory constant.

    Looks up ``filename`` in :data:`PROMPT_FILES` to find the module attribute
    that holds the loaded text, reads the file fresh from ``refs/prompts/``,
    and updates both the module-level constant and the :data:`SECTION_PROMPTS`
    entry (if the prompt is a wiki section). Returns True on success, False if
    the file is unknown or missing.
    """
    attr_name = PROMPT_FILES.get(filename)
    if not attr_name:
        logger.warning("reload_prompt_file: unknown prompt file %r", filename)
        return False
    path = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(path):
        logger.warning("reload_prompt_file: file not found %r", path)
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        logger.warning("reload_prompt_file: error reading %r: %s", filename, e)
        return False
    # Update the module-level constant.
    globals()[attr_name] = content
    # SECTION_PROMPTS maps section_id -> prompt text. Reverse-map the section
    # ids so a section prompt reload stays consistent.
    _section_id_by_attr = {
        "WIKI_OVERVIEW_PROMPT": "overview",
        "WIKI_ARCHITECTURE_PROMPT": "architecture",
        "WIKI_FUNCTIONAL_PROMPT": "functional",
        "WIKI_TECHNICAL_PROMPT": "technical",
        "WIKI_CICD_PROMPT": "cicd",
        "WIKI_LLD_PROMPT": "lld",
        "WIKI_DATAMODEL_PROMPT": "datamodel",
    }
    section_id = _section_id_by_attr.get(attr_name)
    if section_id is not None:
        SECTION_PROMPTS[section_id] = content
    # Expert agent prompts live in api.expert.prompt as module-level constants.
    # ``api.expert.chat`` reads them via ``api.expert.prompt.<CONST>`` at call
    # time (not captured at import), so reloading the prompt module updates the
    # module object the chat path references and the next call picks up the new
    # text. Best-effort refresh.
    if filename in ("expert_agent_system.md", "expert_agent_doc.md"):
        try:
            import importlib
            import api.expert.prompt as _ep  # type: ignore
            importlib.reload(_ep)
        except Exception as e:  # pragma: no cover - optional best-effort
            logger.warning("reload_prompt_file: could not reload api.expert.prompt: %s", e)
    # The verification guard is consumed by name from this module by every
    # generation path (docgen, expert_agent, wiki_generator). It has
    # no SECTION_PROMPTS entry; the globals() update above is sufficient for
    # newly-built prompts to pick it up. (Already-built long-lived generators
    # re-read it lazily.)
    logger.info("reload_prompt_file: refreshed %r -> %s", filename, attr_name)
    return True


# Load every external template. refs/prompts/*.md is the source of truth; the
# fallbacks above are used only if a file is missing.
WIKI_OVERVIEW_PROMPT = load_prompt_file("overview.md", WIKI_OVERVIEW_PROMPT)
WIKI_ARCHITECTURE_PROMPT = load_prompt_file("architecture.md", WIKI_ARCHITECTURE_PROMPT)
WIKI_FUNCTIONAL_PROMPT = load_prompt_file("functional.md", WIKI_FUNCTIONAL_PROMPT)
WIKI_TECHNICAL_PROMPT = load_prompt_file("technical.md", WIKI_TECHNICAL_PROMPT)
WIKI_CICD_PROMPT = load_prompt_file("cicd.md", WIKI_CICD_PROMPT)
WIKI_LLD_PROMPT = load_prompt_file("lld.md", WIKI_LLD_PROMPT)
WIKI_DATAMODEL_PROMPT = load_prompt_file("datamodel.md", WIKI_DATAMODEL_PROMPT)
WIKI_STRUCTURE_PROMPT = load_prompt_file("structure.md", WIKI_STRUCTURE_PROMPT)
WIKI_COMPACT_GENERATION_PROMPT = load_prompt_file(
    "compact_generation.md", WIKI_COMPACT_GENERATION_PROMPT
)

DEEP_RESEARCH_FIRST_ITERATION_PROMPT = load_prompt_file(
    "deep_research_first_iteration.md", DEEP_RESEARCH_FIRST_ITERATION_PROMPT
)
DEEP_RESEARCH_FINAL_ITERATION_PROMPT = load_prompt_file(
    "deep_research_final_iteration.md", DEEP_RESEARCH_FINAL_ITERATION_PROMPT
)
DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT = load_prompt_file(
    "deep_research_intermediate_iteration.md", DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT
)

MERMAID_REPAIR_PROMPT = load_prompt_file("mermaid_repair.md", MERMAID_REPAIR_PROMPT)
VERIFICATION_GUARD = load_prompt_file("_verification_guard.md", VERIFICATION_GUARD)

# ============================================================================
# SECTION_PROMPTS registry (maps section_id -> loaded prompt template)
# ============================================================================

SECTION_PROMPTS = {
    "overview": WIKI_OVERVIEW_PROMPT,
    "architecture": WIKI_ARCHITECTURE_PROMPT,
    "functional": WIKI_FUNCTIONAL_PROMPT,
    "technical": WIKI_TECHNICAL_PROMPT,
    "cicd": WIKI_CICD_PROMPT,
    "lld": WIKI_LLD_PROMPT,
    "datamodel": WIKI_DATAMODEL_PROMPT,
}
