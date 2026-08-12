#!/usr/bin/env python3
"""Lightweight validator for the Productarium prompt catalog (refs/prompts/).

Standalone: requires no third-party packages, no network, no model calls.
Run:  python tests/validate_prompts.py

Checks per prompt file:
  1. exists and is non-empty;
  2. contains exactly the expected runtime placeholders (contract preservation);
  3. no stray `{word}` tokens outside the expected set (typo guard);
  4. str.format-group prompts render with dummy values (catches loose braces);
  5. fenced code blocks (```), are balanced;
  6. no UTF-8 replacement chars (broken Cyrillic);
  7. JSON examples in structure.md / knowledge_graph_extraction.md parse.

Exit code 0 = all good, 1 = at least one failure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "refs" / "prompts"

# --- Expected placeholder contract (verified against consumer code) ----------
# Value = set of placeholder names that MUST appear as literal {name} tokens.
REPLACE_CONTRACT: dict[str, set[str]] = {
    "overview.md": {"file_count", "main_directories", "primary_language",
                    "repo_name", "repo_type", "repo_url"},
    "architecture.md": {"main_files", "previous_content", "project_structure",
                        "repo_name", "repo_url"},
    "functional.md": {"api_endpoints", "app_type", "main_modules",
                    "previous_content", "repo_name", "repo_url"},
    "technical.md": {"config_files", "previous_content", "repo_name",
                    "repo_url", "tech_stack"},
    "cicd.md": {"cicd_files", "config_files", "docker_files",
                "previous_content", "repo_name", "repo_url"},
    "lld.md": {"api_endpoints", "components", "modules", "previous_content",
            "repo_name", "repo_url"},
    "datamodel.md": {"databases", "db_config", "entities", "previous_content",
                    "repo_name", "repo_url"},
    "structure.md": {"file_count", "main_directories", "primary_language",
                    "repo_name", "repo_type", "repo_url"},
    "compact_generation.md": {"project_structure", "repo_name", "repo_url",
                            "tech_stack"},
    "expert_agent_system.md": {"language_name", "product_name"},
    "expert_agent_doc.md": {"language_name", "product_name"},
    "product_summary.md": {"content", "product_name"},
    "documentation_doc.md": {"artifact_name", "content"},
    "openapi_doc.md": {"artifact_name", "content", "previous_content", "repo_name"},
    "asyncapi_doc.md": {"artifact_name", "content", "previous_content", "repo_name"},
    "testcase_doc.md": {"artifact_name", "content", "previous_content", "repo_name"},
    "mermaid_repair.md": {"broken_diagram", "error"},
}

# str.format-group: braces are structural — any stray brace breaks rendering.
FORMAT_CONTRACT: dict[str, set[str]] = {
    "deep_research_first_iteration.md": {"language_name", "repo_name",
                                        "repo_type", "repo_url"},
    "deep_research_intermediate_iteration.md": {"language_name", "repo_name",
                                                "repo_type", "repo_url",
                                                "research_iteration"},
    "deep_research_final_iteration.md": {"language_name", "repo_name",
                                        "repo_type", "repo_url"},
}

# No runtime placeholders at all.
NO_PLACEHOLDER: set[str] = {"_verification_guard.md"}

# recommended-next; str.replace with literal JSON braces allowed.
REPLACE_LOOSE: dict[str, set[str]] = {
    "knowledge_graph_extraction.md": {"product_name", "repo_name",
                                    "artifact_name", "content"},
}

# Files that contain a ```json block which must parse.
JSON_EXAMPLE_FILES = {"structure.md", "knowledge_graph_extraction.md"}

# Docs, not templates.
IGNORE = {"README.md"}

TOKEN_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


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


def check_json_blocks(text: str) -> list[str]:
    problems: list[str] = []
    blocks = JSON_BLOCK_RE.findall(text)
    if not blocks:
        problems.append("no ```json block found (expected at least one)")
        return problems
    for i, block in enumerate(blocks):
        try:
            json.loads(block)
        except json.JSONDecodeError as e:
            problems.append(f"json block #{i + 1} invalid: {e}")
    return problems


def dummy(names: set[str]) -> dict[str, str]:
    return {n: "X" for n in names}


def validate() -> Result:
    r = Result()
    if not PROMPTS_DIR.is_dir():
        r.errors.append(f"prompts dir missing: {PROMPTS_DIR}")
        return r

    present = {p.name for p in PROMPTS_DIR.glob("*.md")}
    expected = (set(REPLACE_CONTRACT) | set(FORMAT_CONTRACT) | NO_PLACEHOLDER
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
        elif fname in FORMAT_CONTRACT:
            exp = FORMAT_CONTRACT[fname]
            missing = exp - found
            extra = found - exp
            if missing:
                r.fail(fname, f"missing placeholders: {sorted(missing)}")
            if extra:
                r.fail(fname, f"unexpected placeholder tokens: {sorted(extra)}")
            try:
                text.format(**dummy(exp))
            except (KeyError, IndexError, ValueError) as e:
                r.fail(fname, f"str.format render failed (stray brace?): {e!r}")
        elif fname in REPLACE_LOOSE:
            exp = REPLACE_LOOSE[fname]
            missing = exp - found
            if missing:
                r.fail(fname, f"missing expected placeholders: {sorted(missing)}")
        else:
            r.fail(fname, "not in any known contract group (add to validator)")

        if fname in JSON_EXAMPLE_FILES:
            for prob in check_json_blocks(text):
                r.fail(fname, prob)

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
