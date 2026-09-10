"""Module containing all prompts used in the DeepWiki project.

Prompt BODIES live externally in ``refs/prompts/<lang>/<name>.md`` (one file
per prompt per language: ``ru`` and ``en``) and are loaded at module import
time by :func:`load_prompt_file`. This module keeps only the abstract,
code-level scaffolding: the canonical wiki section list, language/detail-level
helpers, the loader, and the :data:`SECTION_PROMPTS` registry that maps section
ids to the parsed section contracts.

The seven wiki section contracts live in ONE file —
``refs/prompts/<lang>/docgen_sections.md`` — as ``<section id="...">...</section>``
blocks parsed by :func:`_parse_section_blocks` into :data:`SECTION_PROMPTS`.

The ACTIVE language is resolved by :func:`get_generation_language`: the
``generation.language`` admin setting (``ru`` | ``en``) wins, falling back to
the ``lang.json`` default. Switching it via the admin panel hot-reloads every
prompt through :func:`reload_all_prompt_files` — no restart needed.

Each prompt constant below is a short fallback used only if the corresponding
file is missing. Do NOT inline prompt bodies here — edit the matching
``refs/prompts/<lang>/<name>.md`` file instead.

Inventory invariant: the keys of :data:`PROMPT_FILES` are exactly the
``*.md`` files in each of ``refs/prompts/ru/`` and ``refs/prompts/en/``
(``refs/prompts/README.md`` is documentation, not a prompt). A test enforces
it — adding or deleting a prompt file without updating the registry (or
vice versa) fails the suite.
"""

import logging
import os
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# WIKI SECTION DEFINITIONS
# ============================================================================

# Canonical 7-section wiki structure. ``functional``, ``technical`` and
# ``datamodel`` are PARENT pages: each is decomposed (by the decomposer
# prompt) into SUBPAGES with contracts from ``docgen_subpages.md``.
WIKI_SECTIONS = [
    {"id": "overview", "title_ru": "Общая информация", "title_en": "Overview"},
    {"id": "architecture", "title_ru": "Системная архитектура", "title_en": "System Architecture"},
    {"id": "functional", "title_ru": "Функциональное описание", "title_en": "Functional Description"},
    {"id": "technical", "title_ru": "Технические детали", "title_en": "Technical Details"},
    {"id": "cicd", "title_ru": "CI/CD и SRE", "title_en": "CI/CD & SRE"},
    {"id": "qa", "title_ru": "QA (Тестирование)", "title_en": "QA (Testing)"},
    {"id": "datamodel", "title_ru": "Модель данных", "title_en": "Data Model"},
]

WIKI_SECTION_IDS = [s["id"] for s in WIKI_SECTIONS]

# Sections that decompose into subpages (parent + children pages), mapped to
# their subpage contract id in ``refs/prompts/docgen_subpages.md``.
SUBPAGE_SECTION_IDS = frozenset({"functional", "technical", "datamodel"})
SUBPAGE_TYPE_BY_SECTION = {
    "functional": "functional_item",
    "technical": "technical_item",
    "datamodel": "datamodel_item",
}


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

LANGUAGE_NAMES = {
    "ru": "Russian (Русский)",
    "en": "English",
}

# Prompts that should NOT be wrapped with language/detail-level instructions:
# - Expert + deep research prompts carry a {language_name} placeholder that is
#   substituted per-request, so a fixed language block would conflict.
# - Utility prompts (verification guard, mermaid repair) are not standalone
#   generation prompts and have their own output contracts.
# - Docgen prompts (sections contracts, router JSON, orchestrator dispatch,
#   agent system/section, judge) define their own output contracts and
#   {language_name}/{slot} substitution in api.docgen.codebase.
_UNWRAPPED_PROMPTS = frozenset({
    "expert_agent_system.md",
    "expert_agent_doc.md",
    "_verification_guard.md",
    "mermaid_repair.md",
    # Wave D docgen prompts: the agent system prompt carries a {language_name}
    # placeholder substituted per-request; the section contracts / task block /
    # router (JSON output) / orchestrator (dispatch report) have their own
    # contracts and must not get the generic language+detail prefix.
    "docgen_sections.md",
    "docgen_subpages.md",
    "docgen_decomposer.md",
    "docgen_agent_system.md",
    "docgen_agent_section.md",
    "docgen_router.md",
    "docgen_orchestrator.md",
    "docgen_judge.md",
    "spec_agent_system.md",
    "spec_enrich_task.md",
    # Wave E: deep research planner/researcher/synthesizer carry a
    # {language_name} placeholder substituted per-request.
    "deep_research_planner.md",
    "deep_research_researcher.md",
    "deep_research_synthesizer.md",
    # Wave E review #4: the database doc prompt carries {language_name}
    # substituted per-request from the generate body (was pinned to the
    # config default language and the request field was dropped). The
    # DB RE restructure adds the strict-JSON batch prompts (tables /
    # category objects) with the same per-request substitution.
    "database_doc.md",
    "database_tables.md",
    "database_categories.md",
    "database_relations.md",
})


def _default_language() -> str:
    """Resolve the default output language from lang.json (default 'ru')."""
    try:
        from api.config import configs
        return configs.get("lang_config", {}).get("default", "ru")
    except Exception:
        return "ru"


#: Supported prompt/generation languages (one ``refs/prompts/<lang>/`` dir each).
PROMPT_LANGUAGES = ("ru", "en")

#: Admin settings-store key that overrides the generation language (ru | en).
GENERATION_LANGUAGE_SETTING = "generation.language"


def _normalize_language(language: Optional[str]) -> str:
    """Map any input to a supported language code (default: active language)."""
    if language in PROMPT_LANGUAGES:
        return language  # type: ignore[return-value]
    return get_generation_language()


def get_generation_language() -> str:
    """Resolve the active prompt/generation language.

    Precedence: the ``generation.language`` admin setting (set in the admin
    panel's prompts section) > ``lang.json`` default. The settings store is
    non-fatal (DB down / import time -> falls through to the lang.json value).
    """
    try:
        from api.config.settings import get_setting

        stored = (get_setting(GENERATION_LANGUAGE_SETTING) or "").strip().lower()
        if stored in PROMPT_LANGUAGES:
            return stored
    except Exception:  # pragma: no cover - settings store is non-fatal
        pass
    fallback = _default_language().strip().lower()
    return fallback if fallback in PROMPT_LANGUAGES else "ru"


def _wrap_prompt(prompt: str, language: str = "ru") -> str:
    """Wrap a generation prompt with language and detail-level instructions."""
    language_name = LANGUAGE_NAMES.get(language, language)
    lang_block = LANGUAGE_INSTRUCTION.format(language_name=language_name)
    detail_block = DETAIL_LEVEL_COMPREHENSIVE
    return lang_block + detail_block + prompt


def _maybe_wrap(filename: str, content: str) -> str:
    """Apply language/detail wrapping to generation prompts.

    Skips expert/deep-research prompts (they carry {language_name} for
    per-request substitution) and utility prompts (verification guard,
    mermaid repair) which have their own output contracts.
    """
    if filename in _UNWRAPPED_PROMPTS or "{language_name}" in content:
        return content
    return _wrap_prompt(content, get_generation_language())


# Forward declarations; populated by the dynamic-load block below.
SECTION_PROMPTS: Dict[str, str] = {}
SUBPAGE_CONTRACTS: Dict[str, str] = {}

DEEP_RESEARCH_PLANNER_PROMPT = ""
DEEP_RESEARCH_RESEARCHER_PROMPT = ""
DEEP_RESEARCH_SYNTHESIZER_PROMPT = ""
DATABASE_DOC_PROMPT = ""
DATABASE_TABLES_PROMPT = ""
DATABASE_CATEGORIES_PROMPT = ""
DATABASE_RELATIONS_PROMPT = ""
MERMAID_REPAIR_PROMPT = ""
# Docgen pipeline (Wave F): consolidated section contracts + router +
# orchestrator. DOCGEN_SECTIONS_PROMPT is the RAW parsed file;
# SECTION_PROMPTS below holds the per-section bodies. DOCGEN_SUBPAGES_PROMPT
# holds the subpage contracts (parsed into SUBPAGE_CONTRACTS);
# DOCGEN_DECOMPOSER_PROMPT plans the subpage units as strict JSON.
DOCGEN_SECTIONS_PROMPT = ""
DOCGEN_SUBPAGES_PROMPT = ""
DOCGEN_DECOMPOSER_PROMPT = ""
DOCGEN_ROUTER_PROMPT = ""
DOCGEN_ORCHESTRATOR_PROMPT = ""
# Unified verification guard (anti-hallucination + citation + no-line-numbers
# rules). Appended to every generation prompt via VERIFICATION_GUARD. Loaded
# from refs/prompts/_verification_guard.md; hot-reloadable via the admin panel.
VERIFICATION_GUARD = ""

PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "refs",
    "prompts",
)


def prompts_dir(language: Optional[str] = None) -> str:
    """Directory holding the prompt files for ``language`` (default: active).

    ``refs/prompts/ru`` / ``refs/prompts/en`` — one file per
    :data:`PROMPT_FILES` key in each. ``PROMPTS_DIR`` itself only keeps
    ``README.md`` (documentation, never editable via the admin panel).
    """
    lang = _normalize_language(language)
    return os.path.join(PROMPTS_DIR, lang)


def _resolve_prompt_path(filename: str, language: Optional[str] = None) -> str:
    """First existing path for ``filename``: the language dir, then ``en``.

    Returns the language-dir path even when it does not exist, so callers can
    log a meaningful "missing" target and fall back to their fallback text.
    """
    lang = _normalize_language(language)
    primary = os.path.join(PROMPTS_DIR, lang, filename)
    if os.path.exists(primary) or lang == "en":
        return primary
    fallback = os.path.join(PROMPTS_DIR, "en", filename)
    if os.path.exists(fallback):
        return fallback
    return primary

# Minimal fallback used only when docgen_sections.md is missing: one short
# block per canonical section id so the pipeline can still run.
_SECTIONS_FALLBACK = "\n\n".join(
    f'<section id="{sid}">\n'
    f"Write the wiki section '{sid}' for the repository. Explore the "
    f"repository with the tools, ground every claim in files you read, cite "
    f"sources as `path`, and finish with ONLY the section Markdown.\n"
    f"</section>"
    for sid in WIKI_SECTION_IDS
)

# Minimal fallback used only when docgen_subpages.md is missing: one short
# block per canonical subpage type.
_SUBPAGES_FALLBACK = "\n\n".join(
    f'<subpage id="{sp_type}">\n'
    f"Write the subpage for the unit named in your task. Explore the "
    f"repository with the tools, ground every claim in files you read, cite "
    f"sources as `path`, include the diagrams the task asks for, and finish "
    f"with ONLY the subpage Markdown.\n"
    f"</subpage>"
    for sp_type in ("functional_item", "technical_item", "datamodel_item")
)

# Minimal fallback used only when docgen_decomposer.md is missing: keeps the
# strict-JSON contract usable so the pipeline can still decompose.
_DECOMPOSER_FALLBACK = (
    "You are a documentation planner. Given the repository brief, plan the "
    "subpages of the functional, technical and datamodel wiki sections.\n\n"
    "Respond with ONLY a JSON object (no prose, no code fence):\n"
    '{"functional": [{"slug": str, "title": str, "focus": str}], '
    '"technical": [{"slug": str, "title": str, "focus": str, '
    '"kind": "endpoint|job|integration|reference"}], '
    '"datamodel": [{"slug": str, "title": str, "focus": str, '
    '"kind": "layer"}]}\n\n'
    "At most 10 functional, 12 technical and 6 datamodel units. Ground every "
    "unit in the brief; prefer fewer, broader units when unsure."
)

# Matches <section id="..."> ... </section> blocks in docgen_sections.md.
_SECTION_BLOCK_RE = re.compile(
    r"<section\s+id=[\"'](?P<id>[a-z0-9_-]+)[\"']\s*>\n?(?P<body>.*?)</section>",
    re.DOTALL,
)

# Matches <subpage id="..."> ... </subpage> blocks in docgen_subpages.md.
_SUBPAGE_BLOCK_RE = re.compile(
    r"<subpage\s+id=[\"'](?P<id>[a-z0-9_-]+)[\"']\s*>\n?(?P<body>.*?)</subpage>",
    re.DOTALL,
)


def _parse_section_blocks(text: str) -> Dict[str, str]:
    """Parse ``<section id="...">...</section>`` blocks into a dict.

    Returns section_id -> stripped block body. Malformed/unknown ids are
    skipped silently (a warning is logged by the caller when a canonical
    section ends up missing); duplicate ids keep the LAST occurrence.
    """
    out: Dict[str, str] = {}
    if not text:
        return out
    for match in _SECTION_BLOCK_RE.finditer(text):
        sid = match.group("id").strip()
        body = (match.group("body") or "").strip()
        if sid and body:
            out[sid] = body
    return out


def _parse_subpage_blocks(text: str) -> Dict[str, str]:
    """Parse ``<subpage id="...">...</subpage>`` blocks into a dict.

    Same contract as :func:`_parse_section_blocks`, for the subpage contract
    file (``docgen_subpages.md``). Returns subpage_type -> stripped body;
    duplicate ids keep the LAST occurrence.
    """
    out: Dict[str, str] = {}
    if not text:
        return out
    for match in _SUBPAGE_BLOCK_RE.finditer(text):
        sp_type = match.group("id").strip()
        body = (match.group("body") or "").strip()
        if sp_type and body:
            out[sp_type] = body
    return out


def load_prompt_file(
    filename: str, fallback: str, language: Optional[str] = None
) -> str:
    """Load a prompt template from ``refs/prompts/<lang>/<filename>``.

    ``language`` defaults to the active generation language (see
    :func:`get_generation_language`); when the file is missing there, the
    English copy is used before ``fallback``. Returns the file content
    (stripped + wrapped); generation prompts are automatically prepended with
    language and detail-level instructions via ``_maybe_wrap``.
    """
    content = fallback
    try:
        path = _resolve_prompt_path(filename, language)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
    except Exception as e:
        logger.warning(f"Error loading prompt from {filename}: {e}")
    return _maybe_wrap(filename, content)


# Registry mapping filename -> module attribute name for all prompts loaded
# from ``refs/prompts/*.md``. Used by both the load block below and
# ``reload_prompt_file`` so an edit via the admin panel can hot-reload a
# single prompt without restarting the process.
#
# Keys are filenames (relative to ``refs/prompts/``); values are the module
# attribute name that holds the loaded prompt text.
#
# Inventory invariant: keys == the *.md files in refs/prompts/ minus
# README.md (enforced by tests).
#
# Note: expert_agent_system.md and expert_agent_doc.md are consumed by
# ``api.expert.prompt`` (which loads them via ``load_prompt_file`` into its
# own ``EXPERT_SYSTEM_PROMPT`` / ``EXPERT_DOC_PROMPT`` constants). They are
# included here so ``reload_prompt_file`` can also refresh them via
# ``importlib.reload(api.expert.prompt)`` (best-effort, optional).
PROMPT_FILES: Dict[str, str] = {
    "expert_agent_system.md": "EXPERT_SYSTEM_PROMPT",
    "expert_agent_doc.md": "EXPERT_DOC_PROMPT",
    "product_summary.md": "PRODUCT_SUMMARY_PROMPT",
    "openapi_doc.md": "OPENAPI_DOC_PROMPT",
    "asyncapi_doc.md": "ASYNCAPI_DOC_PROMPT",
    "mermaid_repair.md": "MERMAID_REPAIR_PROMPT",
    "_verification_guard.md": "VERIFICATION_GUARD",
    # Wave D (docgen verification pipeline).
    "docgen_judge.md": "DOCGEN_JUDGE_PROMPT",
    "spec_agent_system.md": "SPEC_AGENT_SYSTEM_PROMPT",
    "spec_enrich_task.md": "SPEC_ENRICH_TASK_PROMPT",
    # Wave E (database reverse-engineering + LangGraph deep research). The
    # tables/categories prompts are consumed via load_prompt_file() at call
    # time by api.docgen.database; loaded here for admin hot-reload + visibility.
    "database_doc.md": "DATABASE_DOC_PROMPT",
    "database_tables.md": "DATABASE_TABLES_PROMPT",
    "database_categories.md": "DATABASE_CATEGORIES_PROMPT",
    "database_relations.md": "DATABASE_RELATIONS_PROMPT",
    "deep_research_planner.md": "DEEP_RESEARCH_PLANNER_PROMPT",
    "deep_research_researcher.md": "DEEP_RESEARCH_RESEARCHER_PROMPT",
    "deep_research_synthesizer.md": "DEEP_RESEARCH_SYNTHESIZER_PROMPT",
    # Wave F (docgen subagent pipeline: consolidated section contracts,
    # routing hints router, deepagents orchestrator, section writer) and the
    # units restructure (subpage contracts + the decomposition planner).
    "docgen_sections.md": "DOCGEN_SECTIONS_PROMPT",
    "docgen_subpages.md": "DOCGEN_SUBPAGES_PROMPT",
    "docgen_decomposer.md": "DOCGEN_DECOMPOSER_PROMPT",
    "docgen_agent_system.md": "DOCGEN_AGENT_SYSTEM_PROMPT",
    "docgen_agent_section.md": "DOCGEN_AGENT_SECTION_PROMPT",
    "docgen_router.md": "DOCGEN_ROUTER_PROMPT",
    "docgen_orchestrator.md": "DOCGEN_ORCHESTRATOR_PROMPT",
}


def _load_sections_registry() -> None:
    """(Re)parse DOCGEN_SECTIONS_PROMPT into the SECTION_PROMPTS dict.

    Missing canonical sections are filled with a generic per-section stub so
    the pipeline can always run; a warning makes the gap visible.
    """
    parsed = _parse_section_blocks(globals().get("DOCGEN_SECTIONS_PROMPT", ""))
    SECTION_PROMPTS.clear()
    SECTION_PROMPTS.update(parsed)
    for sid in WIKI_SECTION_IDS:
        if sid not in SECTION_PROMPTS:
            logger.warning(
                "docgen_sections.md: section %r missing or empty; using generic stub", sid,
            )
            SECTION_PROMPTS[sid] = (
                f"Write the wiki section '{sid}' for the repository. Explore "
                f"the repository with the tools, ground every claim in files "
                f"you read, cite sources as `path`, and finish with ONLY the "
                f"section Markdown."
            )


def _load_subpages_registry() -> None:
    """(Re)parse DOCGEN_SUBPAGES_PROMPT into the SUBPAGE_CONTRACTS dict.

    Missing canonical subpage types are filled with a generic stub so the
    units pipeline can always run; a warning makes the gap visible.
    """
    parsed = _parse_subpage_blocks(globals().get("DOCGEN_SUBPAGES_PROMPT", ""))
    SUBPAGE_CONTRACTS.clear()
    SUBPAGE_CONTRACTS.update(parsed)
    for sp_type in ("functional_item", "technical_item", "datamodel_item"):
        if sp_type not in SUBPAGE_CONTRACTS:
            logger.warning(
                "docgen_subpages.md: subpage %r missing or empty; using generic stub",
                sp_type,
            )
            SUBPAGE_CONTRACTS[sp_type] = (
                "Write the subpage for the unit named in your task. Explore "
                "the repository with the tools, ground every claim in files "
                "you read, cite sources as `path`, and finish with ONLY the "
                "subpage Markdown."
            )


def reload_prompt_file(filename: str, language: Optional[str] = None) -> bool:
    """Re-read a prompt file from disk and update the in-memory constant.

    Looks up ``filename`` in :data:`PROMPT_FILES` to find the module attribute
    that holds the loaded text, reads the file fresh from the active
    language's dir (or ``language`` when given; English copy as fallback),
    and updates the module-level constant. When the file is
    ``docgen_sections.md`` the :data:`SECTION_PROMPTS` registry is re-parsed
    as well. Returns True on success, False if the file is unknown or missing.
    """
    attr_name = PROMPT_FILES.get(filename)
    if not attr_name:
        logger.warning("reload_prompt_file: unknown prompt file %r", filename)
        return False
    path = _resolve_prompt_path(filename, language)
    if not os.path.exists(path):
        logger.warning("reload_prompt_file: file not found %r", path)
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        logger.warning("reload_prompt_file: error reading %r: %s", filename, e)
        return False
    content = _maybe_wrap(filename, content)
    # Update the module-level constant.
    globals()[attr_name] = content
    # The consolidated section contracts file feeds the SECTION_PROMPTS
    # registry (section_id -> parsed contract body); the subpage contracts
    # file feeds SUBPAGE_CONTRACTS the same way.
    if filename == "docgen_sections.md":
        _load_sections_registry()
    if filename == "docgen_subpages.md":
        _load_subpages_registry()

    if filename in ("expert_agent_system.md", "expert_agent_doc.md"):
        try:
            import importlib

            import api.expert.prompt as _ep  # type: ignore
            importlib.reload(_ep)
        except Exception as e:  # pragma: no cover - optional best-effort
            logger.warning("reload_prompt_file: could not reload api.expert.prompt: %s", e)

    logger.info("reload_prompt_file: refreshed %r -> %s", filename, attr_name)
    return True


def reload_all_prompt_files(language: Optional[str] = None) -> None:
    """Reload EVERY prompt file (used when the generation language changes).

    Iterates :data:`PROMPT_FILES` and routes each through
    :func:`reload_prompt_file`, which also re-parses the section/subpage
    registries and refreshes ``api.expert.prompt``. Files missing on disk keep
    their current in-memory value (last good copy), so a partial translation
    set never blanks a prompt.
    """
    reloaded = 0
    for filename in PROMPT_FILES:
        if reload_prompt_file(filename, language=language):
            reloaded += 1
    logger.info(
        "reload_all_prompt_files: refreshed %d/%d prompts (language=%r)",
        reloaded,
        len(PROMPT_FILES),
        _normalize_language(language),
    )


# Load every external template. refs/prompts/<lang>/*.md is the source of
# truth; the fallbacks above are used only if a file is missing. Calls-time
# loaders (spec/summary/expert) read the file fresh with the active language.
DOCGEN_SECTIONS_PROMPT = load_prompt_file("docgen_sections.md", _SECTIONS_FALLBACK)
DOCGEN_SUBPAGES_PROMPT = load_prompt_file("docgen_subpages.md", _SUBPAGES_FALLBACK)
DOCGEN_DECOMPOSER_PROMPT = load_prompt_file("docgen_decomposer.md", _DECOMPOSER_FALLBACK)
DOCGEN_AGENT_SYSTEM_PROMPT = load_prompt_file("docgen_agent_system.md", "")
DOCGEN_AGENT_SECTION_PROMPT = load_prompt_file("docgen_agent_section.md", "")
DOCGEN_ROUTER_PROMPT = load_prompt_file("docgen_router.md", "")
DOCGEN_ORCHESTRATOR_PROMPT = load_prompt_file("docgen_orchestrator.md", "")

# Wave E: consumed via load_prompt_file() at call time by api.expert.deep_research
# and api.docgen.database; loaded here for admin hot-reload + visibility.
DEEP_RESEARCH_PLANNER_PROMPT = load_prompt_file(
    "deep_research_planner.md", DEEP_RESEARCH_PLANNER_PROMPT
)
DEEP_RESEARCH_RESEARCHER_PROMPT = load_prompt_file(
    "deep_research_researcher.md", DEEP_RESEARCH_RESEARCHER_PROMPT
)
DEEP_RESEARCH_SYNTHESIZER_PROMPT = load_prompt_file(
    "deep_research_synthesizer.md", DEEP_RESEARCH_SYNTHESIZER_PROMPT
)
DATABASE_DOC_PROMPT = load_prompt_file("database_doc.md", DATABASE_DOC_PROMPT)
DATABASE_TABLES_PROMPT = load_prompt_file("database_tables.md", DATABASE_TABLES_PROMPT)
DATABASE_CATEGORIES_PROMPT = load_prompt_file(
    "database_categories.md", DATABASE_CATEGORIES_PROMPT
)
DATABASE_RELATIONS_PROMPT = load_prompt_file(
    "database_relations.md", DATABASE_RELATIONS_PROMPT
)

MERMAID_REPAIR_PROMPT = load_prompt_file("mermaid_repair.md", MERMAID_REPAIR_PROMPT)
VERIFICATION_GUARD = load_prompt_file("_verification_guard.md", VERIFICATION_GUARD)

# ============================================================================
# SECTION_PROMPTS registry (maps section_id -> parsed section contract body)
# and SUBPAGE_CONTRACTS registry (subpage_type -> parsed contract body)
# ============================================================================
_load_sections_registry()
_load_subpages_registry()
