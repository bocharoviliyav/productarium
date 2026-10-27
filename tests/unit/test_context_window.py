"""Unit tests for the SPEC-005 context-window resolution helpers.

Covers ``api.docgen._common.halve_budget_on_overflow`` (the shared
overflow-halving retry loop with the 8192 floor) and its wiring into the
codebase docgen single-call path.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.docgen.codebase as cb
from api.docgen._common import (
    _OVERFLOW_BUDGET_FLOOR_TOKENS,
    _is_context_overflow,
    halve_budget_on_overflow,
)


class TestIsContextOverflow:
    def test_matches_known_markers(self):
        assert _is_context_overflow(ValueError("maximum context length is 4096 tokens"))
        assert _is_context_overflow(RuntimeError("Error code: 400 - 'context_length_exceeded'"))
        assert _is_context_overflow(Exception("prompt is too long"))
        assert _is_context_overflow(Exception("Too many tokens"))

    def test_rejects_other_errors(self):
        assert not _is_context_overflow(RuntimeError("connection error"))
        assert not _is_context_overflow(ValueError("rate limited"))

    def test_codebase_reexport_is_the_shared_helper(self):
        assert cb._is_context_overflow is _is_context_overflow


class TestHalveBudgetOnOverflow:
    def test_halves_on_overflow(self):
        budgets = []

        async def call(budget):
            budgets.append(budget)
            if len(budgets) == 1:
                raise RuntimeError("Error code: 400 - 'context_length_exceeded'")
            return "ok"

        assert asyncio.run(halve_budget_on_overflow(call, 32_768)) == "ok"
        assert budgets == [32_768, 16_384]

    def test_succeeds_without_overflow_untouched(self):
        budgets = []

        async def call(budget):
            budgets.append(budget)
            return "ok"

        assert asyncio.run(halve_budget_on_overflow(call, 32_768)) == "ok"
        assert budgets == [32_768]

    def test_reraises_non_overflow_immediately(self):
        calls = []

        async def call(budget):
            calls.append(budget)
            raise RuntimeError("connection error")

        with pytest.raises(RuntimeError, match="connection error"):
            asyncio.run(halve_budget_on_overflow(call, 32_768))
        assert calls == [32_768]  # no retry for non-overflow errors

    def test_stops_at_floor_and_raises(self):
        budgets = []

        async def call(budget):
            budgets.append(budget)
            raise RuntimeError("Error code: 400 - 'context_length_exceeded'")

        with pytest.raises(RuntimeError, match="context_length_exceeded"):
            asyncio.run(halve_budget_on_overflow(call, 16_384))
        # 16384 -> 8192 (the floor) -> cannot halve further -> raise
        assert budgets == [16_384, 8_192]

    def test_floor_budget_never_retries(self):
        calls = []

        async def call(budget):
            calls.append(budget)
            raise RuntimeError("message too long")

        with pytest.raises(RuntimeError, match="message too long"):
            asyncio.run(halve_budget_on_overflow(call, _OVERFLOW_BUDGET_FLOOR_TOKENS))
        assert calls == [8_192]

    def test_warning_logged_on_each_halving(self, caplog):
        import logging

        async def call(budget):
            if budget > 8_192:
                raise RuntimeError("prompt is too long")
            return "ok"

        with caplog.at_level(logging.WARNING, logger="api.docgen._common"):
            assert asyncio.run(halve_budget_on_overflow(call, 32_768)) == "ok"
        assert any("halved prompt budget" in r.getMessage() for r in caplog.records)


class TestSingleCallWiring:
    """``_generate_section_single_call`` retries with a halved fit budget."""

    def test_overflow_then_success_via_helper(self, monkeypatch):
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 32_768)
        prompts = []

        class _LLM:
            async def generate(self, prompt):
                prompts.append(prompt)
                if len(prompts) == 1:
                    raise RuntimeError(
                        "Error code: 400 - 'context_length_exceeded'"
                    )
                return "recovered section text"

        result = asyncio.run(cb._generate_section_single_call("prompt", "", _LLM()))
        assert result == "recovered section text"
        assert len(prompts) == 2  # one retry after the halving

    def test_persistent_overflow_degrades_to_empty(self, monkeypatch):
        monkeypatch.setattr(cb, "_resolve_docgen_context_window", lambda: 8_192)

        class _LLM:
            async def generate(self, prompt):
                raise RuntimeError("maximum context length is 4096 tokens")

        result = asyncio.run(cb._generate_section_single_call("prompt", "", _LLM()))
        assert result == ""  # floor reached -> raise -> caught -> graceful ""
