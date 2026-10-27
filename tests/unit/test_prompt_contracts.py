"""Unit tests for the SPEC-015 deterministic unit-output contract.

Covers:
- ``UNIT_PROMPT_FILES`` registry: every member exists in BOTH language dirs.
- Contract composition: ``load_prompt_file`` composes the unified output
  contract (``_unit_output_contract.md`` from the same language dir) onto
  the END of every unit prompt (ru + en); non-unit prompts stay untouched.
- Trip-wire: a unit-SHAPED file in the prompt registry but outside
  ``UNIT_PROMPT_FILES`` ships without the contract — the check fails (a
  new unit prompt cannot silently skip the contract).
- Gate: truncated garbage (open fence / abrupt ending) fails
  ``validate_page``, the retry feedback names the truncation, and a clean
  second attempt passes.
- Watchdog: ``run_with_unit_budget`` cancels a stalled unit (TimeoutError)
  while a quick sibling unit completes.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import api.docgen._common as c
from api.prompts import (
    PROMPT_FILES,
    PROMPT_LANGUAGES,
    PROMPTS_DIR,
    UNIT_PROMPT_FILES,
    load_prompt_file,
)


#: Filename stems that mark a prompt as a docgen UNIT prompt (its output is
#: consumed as page Markdown / strict JSON by a unit agent). Any *.md in the
#: registry starting with one of these MUST be a UNIT_PROMPT_FILES member.
UNIT_STEM_PREFIXES = (
    "docgen_agent_section",
    "docgen_sections",
    "docgen_subpages",
    "database_entity",
    "database_tables",
    "database_categories",
    "database_relations",
)


def _unregistered_unit_prompts(filenames) -> list:
    """Unit-shaped prompt files missing from UNIT_PROMPT_FILES (trip-wire)."""
    return sorted(
        name for name in filenames
        if name.endswith(".md")
        and any(name.startswith(prefix) for prefix in UNIT_STEM_PREFIXES)
        and name not in UNIT_PROMPT_FILES
    )


# ============================================================================
# Unit contract registry (SPEC-015)
# ============================================================================
class TestUnitContractRegistry:
    def test_every_unit_prompt_exists_in_both_languages(self):
        for lang in PROMPT_LANGUAGES:
            for name in UNIT_PROMPT_FILES:
                assert os.path.isfile(
                    os.path.join(PROMPTS_DIR, lang, name)
                ), f"{lang}/{name} missing"

    def test_contract_composed_onto_every_unit_prompt(self):
        for lang in PROMPT_LANGUAGES:
            contract = load_prompt_file("_unit_output_contract.md", "", language=lang)
            assert contract, f"{lang}: contract file empty"
            for name in UNIT_PROMPT_FILES:
                composed = load_prompt_file(name, "", language=lang)
                assert composed.endswith(contract), f"{lang}/{name}: contract not last"

    def test_non_unit_prompt_untouched_by_contract(self):
        contract = load_prompt_file("_unit_output_contract.md", "", language="en")
        # The router answers strict JSON — not a unit prompt; the contract
        # must not leak into it.
        router = load_prompt_file("docgen_router.md", "", language="en")
        assert not router.endswith(contract)

    def test_no_registered_unit_prompt_without_contract(self):
        # THE trip-wire: a unit-shaped file in the prompt registry but
        # outside UNIT_PROMPT_FILES ships without the output contract.
        assert _unregistered_unit_prompts(PROMPT_FILES) == []

    def test_trip_wire_catches_a_new_unit_file(self):
        # Mechanism demo: a hypothetical database_entity_v2.md added to the
        # registry (as any new prompt must be) is reported.
        files = set(PROMPT_FILES) | {"database_entity_v2.md"}
        assert _unregistered_unit_prompts(files) == ["database_entity_v2.md"]


# ============================================================================
# Gate: truncation detection drives the retry (SPEC-015)
# ============================================================================
class TestGateTruncation:
    def _valid(self) -> str:
        return "# Обзор\n\n- пункт один\n- пункт два\n\n" + "Текст раздела. " * 30

    def _truncated(self) -> str:
        return (
            "# Обзор\n\n- пункт\n\n" + "Текст раздела. " * 30
            + "\n```mermaid\nflowchart TD\n  A --> B"
        )

    def test_validate_page_flags_truncation(self):
        assert any("обрезан" in v for v in c.validate_page("Обзор", self._truncated()))
        abrupt = "# Обзор\n\n- пункт\n\n" + "Текст раздела. " * 40 + "и обрыв на полусл"
        assert any("обрезан" in v for v in c.validate_page("Обзор", abrupt))
        assert c.validate_page("Обзор", self._valid()) == []

    def test_truncated_first_attempt_retries_then_recovers(self):
        seen = []

        async def run_once(feedback):
            seen.append(feedback)
            if len(seen) == 1:
                return self._truncated()
            assert "<quality_gate>" in feedback
            assert "обрезан" in feedback
            return self._valid()

        text, degraded, attempts = asyncio.run(
            c.run_with_page_gate(run_once, title="Обзор")
        )
        assert degraded is False and attempts == 2
        assert text == self._valid()


# ============================================================================
# Per-unit wall-clock watchdog (SPEC-015)
# ============================================================================
class TestUnitBudgetWatchdog:
    def test_stalled_unit_cancelled_quick_unit_completes(self):
        async def stalled():
            await asyncio.sleep(999)

        async def quick():
            return "done"

        async def main():
            return await asyncio.gather(
                c.run_with_unit_budget(stalled(), seconds=0.05, label="u1"),
                c.run_with_unit_budget(quick(), seconds=5, label="u2"),
                return_exceptions=True,
            )

        stalled_result, quick_result = asyncio.run(main())
        assert isinstance(stalled_result, asyncio.TimeoutError)
        assert quick_result == "done"
