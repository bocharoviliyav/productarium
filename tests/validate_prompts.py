#!/usr/bin/env python3
"""Lightweight validator for the Productarium prompt catalog (refs/prompts/).

Standalone: requires no third-party packages, no network, no model calls.
Run:  python tests/validate_prompts.py

Checks per prompt file:
  1. exists and is non-empty;
  2. contains exactly the expected runtime placeholders (contract preservation);
  3. no stray `{word}` tokens outside the expected set (typo guard);
  4. fenced code blocks (```) are balanced;
  5. no UTF-8 replacement chars (broken Cyrillic);
  6. structural checks for the runtime-parsed contract files
     (docgen_sections.md / docgen_subpages.md block ids).

Every prompt body is substituted with str.replace (never str.format — bodies
may carry literal JSON/Mermaid braces), so the validator only tracks the
lowercase `{snake_case}` tokens the pipeline actually fills. The exact sets
below were verified against the consumer code (api.docgen.codebase,
api.docgen.verification, api.docgen.spec, api.docgen.database,
api.expert.deep_research, api.expert.prompt).

Exit code 0 = all good, 1 = at least one failure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "refs" / "prompts"

# --- Expected placeholder contract (verified against consumer code) ----------
# Value = set of placeholder names that MUST appear as literal {name} tokens;
# tokens outside the set are reported (typo guard).
REPLACE_CONTRACT: dict[str, set[str]] = {
    # Docgen pipeline (api.docgen.codebase; judge in api.docgen.verification).
    "docgen_router.md": {"repo_brief", "sections_list"},
    "docgen_orchestrator.md": {"repo_name", "sections_list", "reused_sections"},
    "docgen_agent_system.md": {"language_name"},
    "docgen_agent_section.md": {
        "repo_url", "repo_name", "section_id", "section_title",
        "repo_brief", "sections_list", "section_hints",
        "siblings_list", "section_instruction",
    },
    "docgen_decomposer.md": {"repo_brief", "sections_list", "section_hints"},
    "docgen_judge.md": {"draft_section", "source_evidence"},
    "docgen_subpages.md": {"item_title", "item_focus", "item_kind", "siblings_list"},
    # Expert agent (api.expert.prompt) + deep research (api.expert.deep_research).
    "expert_agent_system.md": {"language_name", "product_name"},
    "expert_agent_doc.md": {"language_name", "product_name"},
    "deep_research_planner.md": {
        "query", "product_name", "language_name", "history", "findings",
    },
    "deep_research_researcher.md": {
        "plan", "product_name", "language_name", "iteration", "max_iterations",
    },
    "deep_research_synthesizer.md": {
        "query", "product_name", "language_name", "history", "findings",
    },
    # Spec enrichment (api.docgen.spec).
    "spec_agent_system.md": {"language_name", "spec_kind"},
    "spec_enrich_task.md": {"artifact_name", "content", "skeleton", "spec_kind"},
    # Database reverse-engineering (api.docgen.database).
    "database_doc.md": {
        "database_name", "dsn_masked", "schema_dump", "skeleton", "language_name",
    },
    # Misc generation.
    "product_summary.md": {"content", "product_name"},
    "openapi_doc.md": {"artifact_name", "content", "previous_content", "repo_name"},
    "asyncapi_doc.md": {"artifact_name", "content", "previous_content", "repo_name"},
    # Utilities.
    "mermaid_repair.md": {"broken_diagram", "error"},
}

# No runtime placeholders at all.
NO_PLACEHOLDER: set[str] = {"_verification_guard.md"}

# Raw contract file parsed into <section> blocks at runtime: no placeholder is
# substituted inside the bodies (the only {language_name} occurrence is a prose
# mention — the language is substituted upstream in docgen_agent_system.md).
REPLACE_LOOSE: dict[str, set[str]] = {
    "docgen_sections.md": set(),
}

# --- Structural checks for the runtime-parsed contract files -----------------
# Must mirror api.prompts.WIKI_SECTION_IDS (the seven canonical sections).
CANONICAL_SECTION_IDS = (
    "overview", "architecture", "functional", "technical", "cicd", "qa", "datamodel",
)
CANONICAL_SUBPAGE_IDS = ("functional_item", "technical_item", "datamodel_item")

# Same id charset as api.prompts._SECTION_BLOCK_RE / _SUBPAGE_BLOCK_RE.
SECTION_ID_RE = re.compile(r"<section\s+id=[\"']([a-z0-9_-]+)[\"']")
SUBPAGE_ID_RE = re.compile(r"<subpage\s+id=[\"']([a-z0-9_-]+)[\"']")

# Docs, not templates.
IGNORE = {"README.md"}

TOKEN_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


class Result:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checked = 0

    def fail(self, fname: str, msg: str) -> None:
        self.errors.append(f"{fname}: {msg}")


def check_fences(text: str) -> bool:
    # Only lines whose stripped content STARTS with ``` are real fence
    # delimiters; inline ``` mentions inside prose do not toggle a block.
    fence_lines = sum(1 for ln in text.splitlines() if ln.lstrip().startswith("```"))
    return fence_lines % 2 == 0


def check_section_blocks(fname: str, text: str, r: Result) -> None:
    """docgen_sections.md: exactly the canonical section blocks, nothing else."""
    ids = SECTION_ID_RE.findall(text)
    unknown = sorted(set(ids) - set(CANONICAL_SECTION_IDS))
    missing = sorted(set(CANONICAL_SECTION_IDS) - set(ids))
    if missing:
        r.fail(fname, f"missing <section> blocks: {missing}")
    if unknown:
        # The runtime parser silently DROPS unknown ids (the section then runs
        # on the generic stub) — so a typo'd id must fail loudly here.
        r.fail(fname, f"unknown <section> blocks (dropped at runtime): {unknown}")
    if "lld" in ids:
        r.fail(fname, 'legacy <section id="lld"> block still present')


def check_subpage_blocks(fname: str, text: str, r: Result) -> None:
    """docgen_subpages.md: exactly the canonical subpage-type blocks."""
    ids = SUBPAGE_ID_RE.findall(text)
    unknown = sorted(set(ids) - set(CANONICAL_SUBPAGE_IDS))
    missing = sorted(set(CANONICAL_SUBPAGE_IDS) - set(ids))
    if missing:
        r.fail(fname, f"missing <subpage> blocks: {missing}")
    if unknown:
        r.fail(fname, f"unknown <subpage> blocks (dropped at runtime): {unknown}")


def validate() -> Result:
    r = Result()
    if not PROMPTS_DIR.is_dir():
        r.errors.append(f"prompts dir missing: {PROMPTS_DIR}")
        return r

    present = {p.name for p in PROMPTS_DIR.glob("*.md")}
    expected = (set(REPLACE_CONTRACT) | NO_PLACEHOLDER
                | set(REPLACE_LOOSE) | IGNORE)
    for missing in sorted(expected - present):
        r.fail(missing, "expected prompt file is missing")

    for fname in sorted(present):
        if fname in IGNORE:
            continue
        r.checked += 1
        path = PROMPTS_DIR / fname
        text = path.read_text(encoding="utf-8")

        if not text.strip():
            r.fail(fname, "file is empty")
            continue
        if "�" in text:
            r.fail(fname, "contains UTF-8 replacement char (broken encoding)")
        if not check_fences(text):
            r.fail(fname, "unbalanced ``` fenced blocks")

        found = set(TOKEN_RE.findall(text))

        if fname in NO_PLACEHOLDER:
            if found:
                r.fail(fname, f"unexpected placeholders present: {sorted(found)}")
        elif fname in REPLACE_CONTRACT:
            exp = REPLACE_CONTRACT[fname]
            missing = exp - found
            extra = found - exp
            if missing:
                r.fail(fname, f"missing placeholders: {sorted(missing)}")
            if extra:
                r.fail(fname, f"unexpected placeholder tokens: {sorted(extra)}")
        elif fname in REPLACE_LOOSE:
            exp = REPLACE_LOOSE[fname]
            missing = exp - found
            if missing:
                r.fail(fname, f"missing expected placeholders: {sorted(missing)}")
        else:
            r.fail(fname, "not in any known contract group (add to validator)")

        if fname == "docgen_sections.md":
            check_section_blocks(fname, text, r)
        elif fname == "docgen_subpages.md":
            check_subpage_blocks(fname, text, r)

    return r


def main() -> int:
    r = validate()
    if r.errors:
        print(f"FAIL: {len(r.errors)} problem(s) across prompts "
            f"({r.checked} files checked)")
        for e in r.errors:
            print(f"  - {e}")
        return 1
    print(f"OK: {r.checked} prompt files validated, no problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
