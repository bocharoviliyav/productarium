#!/usr/bin/env python3
"""Unit tests for the Mermaid verifier + repair loop (api.formats.mermaid).

Runs under pytest (pytest.ini: testpaths=test). No live LLM/Node required for
the core logic tests: ``verify_diagram`` is monkeypatched to return canned
:class:`VerifyResult`s so the splice/queue/budget logic is deterministic. One
integration test exercises the real Node validator and is skipped when
``node`` or the frontend ``mermaid`` bundle is unavailable.
"""

from __future__ import annotations

import asyncio
import os
import shutil

import pytest


# --- Shared fixtures ---------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Keep verifier tunables deterministic across tests."""
    monkeypatch.setenv("MERMAID_VERIFY", "true")
    monkeypatch.setenv("MERMAID_MAX_REPAIR_ATTEMPTS", "3")
    monkeypatch.setenv("MERMAID_VERIFY_TIMEOUT", "10")
    monkeypatch.setenv("MERMAID_REPAIR_TIMEOUT", "120")
    # Force-reload the module so env changes take effect for module-level
    # tunables (they are read at import time).
    import importlib
    import api.formats.mermaid as mv
    importlib.reload(mv)
    yield


def _reload():
    import importlib
    import api.formats.mermaid as mv
    importlib.reload(mv)
    return mv


# ============================================================================
# Block extraction
# ============================================================================
class TestExtractMermaidBlocks:
    def test_no_blocks(self):
        mv = _reload()
        assert mv.extract_mermaid_blocks("# Title\n\nsome text\n") == []

    def test_single_block(self):
        mv = _reload()
        md = "intro\n\n```mermaid\nflowchart TD\n  A --> B\n```\n\noutro\n"
        blocks = mv.extract_mermaid_blocks(md)
        assert len(blocks) == 1
        b = blocks[0]
        assert b.index == 0
        assert "flowchart TD" in b.body
        assert b.raw.startswith("```mermaid")
        assert b.raw.endswith("```")
        # Offsets point at the raw block within the source.
        assert md[b.start:b.end] == b.raw

    def test_multiple_blocks_preserve_order_and_offsets(self):
        mv = _reload()
        md = (
            "```mermaid\nflowchart TD\n  A --> B\n```\n"
            "middle\n"
            "```mermaid\nsequenceDiagram\n  A->>B: hi\n```\n"
        )
        blocks = mv.extract_mermaid_blocks(md)
        assert len(blocks) == 2
        assert blocks[0].index == 0
        assert blocks[1].index == 1
        assert "flowchart" in blocks[0].body
        assert "sequenceDiagram" in blocks[1].body
        # Both offsets are valid slices of the source.
        for b in blocks:
            assert md[b.start:b.end] == b.raw

    def test_language_variant_fence(self):
        mv = _reload()
        # Tolerant of ```mermaid{.theme} style fences.
        md = "```mermaid{.theme}\nflowchart TD\n  A --> B\n```\n"
        blocks = mv.extract_mermaid_blocks(md)
        assert len(blocks) == 1
        assert "flowchart TD" in blocks[0].body

    def test_empty_string(self):
        mv = _reload()
        assert mv.extract_mermaid_blocks("") == []

    def test_non_mermaid_fences_ignored(self):
        mv = _reload()
        md = "```python\nprint('hi')\n```\n```js\nconsole.log(1)\n```\n"
        assert mv.extract_mermaid_blocks(md) == []


# ============================================================================
# verify_diagram infrastructure failure handling
# ============================================================================
class TestVerifyDiagramInfrastructure:
    def test_missing_node_returns_unverifiable(self, monkeypatch):
        mv = _reload()
        monkeypatch.setattr(mv.shutil, "which", lambda name: None)
        res = asyncio.run(mv.verify_diagram("flowchart TD\n  A --> B"))
        # Conservative: treat as unverifiable (ok=True) so generation is never
        # blocked, and do NOT mark the diagram broken.
        assert res.ok is True
        assert res.unverifiable is True

    def test_empty_body_is_ok(self, monkeypatch):
        mv = _reload()
        monkeypatch.setattr(mv.shutil, "which", lambda name: "/usr/bin/node")
        res = asyncio.run(mv.verify_diagram("   "))
        assert res.ok is True


# ============================================================================
# Repair prompt extraction from LLM responses
# ============================================================================
class TestExtractRepairedBody:
    def test_fenced_mermaid_block(self):
        mv = _reload()
        text = "Here is the fix:\n```mermaid\nflowchart TD\n  A --> B\n```\n"
        body = mv._extract_repaired_body(text)
        assert body is not None
        assert "flowchart TD" in body

    def test_bare_diagram_with_keyword(self):
        mv = _reload()
        text = "flowchart TD\n  A --> B"
        body = mv._extract_repaired_body(text)
        assert body is not None
        assert "flowchart TD" in body

    def test_unrelated_text_returns_none(self):
        mv = _reload()
        body = mv._extract_repaired_body("I cannot fix this diagram.")
        assert body is None

    def test_empty_returns_none(self):
        mv = _reload()
        assert mv._extract_repaired_body("") is None
        assert mv._extract_repaired_body(None) is None

    def test_fence_only_without_keyword(self):
        mv = _reload()
        # Stripped to bare text with no diagram keyword -> None.
        body = mv._extract_repaired_body("```\njust some prose\n```")
        assert body is None


# ============================================================================
# Normalization + hashing (budget dedup)
# ============================================================================
class TestBodyHashing:
    def test_trailing_whitespace_does_not_change_hash(self):
        mv = _reload()
        a = "flowchart TD\n  A --> B\n"
        b = "flowchart TD\n  A --> B   \n   "
        assert mv._body_hash(a) == mv._body_hash(b)

    def test_different_diagrams_have_different_hashes(self):
        mv = _reload()
        assert mv._body_hash("flowchart TD\n  A --> B") != mv._body_hash(
            "flowchart TD\n  A --> C"
        )


# ============================================================================
# run_repair_loop end-to-end (verifier + LLM mocked)
# ============================================================================
def _patch_verify(monkeypatch, results_by_body):
    """Patch verify_diagram to return canned results keyed by body string.

    ``results_by_body`` maps an exact body -> VerifyResult. A default
    ``VerifyResult(ok=True)`` is used for bodies not in the map.
    """
    mv = _reload()

    async def _fake_verify(body, timeout=None):
        return results_by_body.get(body, mv.VerifyResult(ok=True))

    monkeypatch.setattr(mv, "verify_diagram", _fake_verify)
    return mv


class TestRunRepairLoop:
    def test_valid_diagram_left_untouched(self, monkeypatch):
        mv = _patch_verify(monkeypatch, {})
        md = "```mermaid\nflowchart TD\n  A --> B\n```\n"
        patched, stats = asyncio.run(mv.run_repair_loop(md, llm=None))
        assert patched == md
        assert stats == {"verified": 1, "broken": 0, "unverifiable": 0,
                         "fixed": 0, "failed": 0}

    def test_broken_diagram_repaired_and_spliced(self, monkeypatch):
        mv = _reload()
        broken = "flowchart TD\n  A --> > B\n"
        results = {broken: mv.VerifyResult(ok=False, error="Parse error")}

        async def _fake_verify(body, timeout=None):
            return results.get(body, mv.VerifyResult(ok=True))

        monkeypatch.setattr(mv, "verify_diagram", _fake_verify)

        async def llm(prompt):
            return "```mermaid\nflowchart TD\n    A --> B\n```"

        md = "intro\n\n```mermaid\nflowchart TD\n  A --> > B\n```\n\noutro\n"
        patched, stats = asyncio.run(mv.run_repair_loop(md, llm))
        # The broken body was replaced with the fixed one.
        assert "A --> > B" not in patched
        assert "A --> B" in patched
        assert stats["broken"] == 1
        assert stats["fixed"] == 1
        assert stats["failed"] == 0

    def test_unverifiable_c4_left_untouched(self, monkeypatch):
        mv = _reload()
        c4_body = "C4Context\n  title Test\n  Person(user, \"User\")\n"
        results = {
            c4_body: mv.VerifyResult(
                ok=False, unverifiable=True, error="Bt.addHook is not a function"
            ),
        }

        async def _fake_verify(body, timeout=None):
            return results.get(body, mv.VerifyResult(ok=True))

        monkeypatch.setattr(mv, "verify_diagram", _fake_verify)

        async def llm(prompt):
            pytest.fail("LLM should not be called for an unverifiable diagram")

        md = f"```mermaid\n{c4_body}```\n"
        patched, stats = asyncio.run(mv.run_repair_loop(md, llm))
        assert patched == md  # unchanged
        assert stats["unverifiable"] == 1
        assert stats["broken"] == 0
        assert stats["fixed"] == 0

    def test_broken_still_broken_after_budget_gets_marker(self, monkeypatch):
        mv = _reload()
        broken = "flowchart TD\n  A --> > B\n"
        # Both the original AND the LLM's re-suggested body verify as broken so
        # the loop exhausts the budget and falls back to the original + marker.
        still_broken = "flowchart TD\n  X --> > Y\n"
        results = {
            broken: mv.VerifyResult(ok=False, error="Parse error"),
            still_broken: mv.VerifyResult(ok=False, error="Parse error"),
        }

        async def _fake_verify(body, timeout=None):
            # Normalize for lookup so trailing-newline differences don't matter.
            key = (body or "").strip() + "\n"
            return results.get(key, mv.VerifyResult(ok=False, error="Parse error"))

        monkeypatch.setattr(mv, "verify_diagram", _fake_verify)

        async def llm(prompt):
            # Always returns the same still-broken body.
            return f"```mermaid\n{still_broken}```"

        md = f"```mermaid\n{broken}```\n"
        patched, stats = asyncio.run(mv.run_repair_loop(md, llm))
        # Original broken block is kept in place...
        assert "A --> > B" in patched
        # ...and an error marker is appended.
        assert "Mermaid" in patched
        assert "не отрисовывается" in patched
        assert stats["failed"] >= 1
        assert stats["fixed"] == 0

    def test_no_llm_marks_broken_with_marker(self, monkeypatch):
        mv = _reload()
        broken = "flowchart TD\n  A --> > B\n"
        results = {broken: mv.VerifyResult(ok=False, error="Parse error")}

        async def _fake_verify(body, timeout=None):
            return results.get(body, mv.VerifyResult(ok=True))

        monkeypatch.setattr(mv, "verify_diagram", _fake_verify)
        md = f"```mermaid\n{broken}```\n"
        patched, stats = asyncio.run(mv.run_repair_loop(md, llm=None))
        assert "A --> > B" in patched
        assert "Mermaid" in patched
        assert stats["broken"] == 1
        assert stats["failed"] == 1

    def test_disabled_returns_unchanged(self, monkeypatch):
        monkeypatch.setenv("MERMAID_VERIFY", "false")
        mv = _reload()
        md = "```mermaid\nflowchart TD\n  A --> B\n```\n"
        patched, stats = asyncio.run(mv.run_repair_loop(md, llm=None))
        assert patched == md
        assert stats == {"verified": 0, "broken": 0, "unverifiable": 0,
                         "fixed": 0, "failed": 0}

    def test_progress_callback_invoked(self, monkeypatch):
        mv = _patch_verify(monkeypatch, {})
        md = "```mermaid\nflowchart TD\n  A --> B\n```\n"
        seen = []

        async def on_progress(stats):
            seen.append(dict(stats))

        asyncio.run(mv.run_repair_loop(md, llm=None, on_progress=on_progress))
        assert len(seen) == 1
        assert seen[0]["verified"] == 1

    def test_multiple_blocks_mixed_outcomes(self, monkeypatch):
        mv = _reload()
        good = "flowchart TD\n  A --> B\n"
        broken = "flowchart TD\n  A --> > B\n"
        c4 = "C4Context\n  title T\n"
        results = {
            broken: mv.VerifyResult(ok=False, error="Parse error"),
            c4: mv.VerifyResult(ok=False, unverifiable=True, error="addHook"),
        }

        async def _fake_verify(body, timeout=None):
            return results.get(body, mv.VerifyResult(ok=True))

        monkeypatch.setattr(mv, "verify_diagram", _fake_verify)

        async def llm(prompt):
            return "```mermaid\nflowchart TD\n    A --> B\n```"

        md = (
            f"```mermaid\n{good}```\n"
            f"```mermaid\n{broken}```\n"
            f"```mermaid\n{c4}```\n"
        )
        patched, stats = asyncio.run(mv.run_repair_loop(md, llm))
        assert stats["verified"] == 1
        assert stats["broken"] == 1
        assert stats["unverifiable"] == 1
        assert stats["fixed"] == 1
        # C4 block preserved verbatim.
        assert "C4Context" in patched

    def test_budget_shared_for_identical_bodies(self, monkeypatch):
        # Two identical broken blocks share ONE budget unit: with attempts=1
        # budget, both get repaired in a single LLM call each (they are
        # distinct block indices but the same hash). Here we set the budget to
        # 1 and confirm neither re-loops infinitely.
        monkeypatch.setenv("MERMAID_MAX_REPAIR_ATTEMPTS", "1")
        mv = _reload()
        broken = "flowchart TD\n  A --> > B\n"
        call_count = {"n": 0}

        async def _fake_verify(body, timeout=None):
            if body == broken:
                return mv.VerifyResult(ok=False, error="Parse error")
            return mv.VerifyResult(ok=True)

        monkeypatch.setattr(mv, "verify_diagram", _fake_verify)

        async def llm(prompt):
            call_count["n"] += 1
            # Returns a valid (different) body.
            return "```mermaid\nflowchart TD\n    A --> B\n```"

        md = f"```mermaid\n{broken}```\n```mermaid\n{broken}```\n"
        patched, stats = asyncio.run(mv.run_repair_loop(md, llm))
        # Both blocks were repaired (each got its own LLM call within budget).
        assert stats["fixed"] == 2
        assert "A --> > B" not in patched
        assert patched.count("A --> B") == 2


# ============================================================================
# Integration test against the real Node validator (skipped if unavailable)
# ============================================================================
def _node_and_mermaid_available():
    if not shutil.which("node"):
        return False
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "api", "_mermaid_validate.mjs",
    )
    if not os.path.isfile(script):
        return False
    mermaid_bundle = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "node_modules", "mermaid", "dist", "mermaid.esm.min.mjs",
    )
    return os.path.isfile(mermaid_bundle)


@pytest.mark.skipif(
    not _node_and_mermaid_available(),
    reason="node or frontend mermaid bundle not installed",
)
class TestRealNodeValidator:
    def test_valid_diagram_ok(self):
        import api.formats.mermaid as mv
        res = asyncio.run(mv.verify_diagram("flowchart TD\n  A --> B\n"))
        assert res.ok is True
        assert not res.unverifiable

    def test_broken_diagram_reports_error(self):
        import api.formats.mermaid as mv
        res = asyncio.run(mv.verify_diagram("flowchart TD\n  A --> > B\n"))
        assert res.ok is False
        assert res.error
        assert not res.unverifiable

    def test_c4_marked_unverifiable(self):
        import api.formats.mermaid as mv
        res = asyncio.run(
            mv.verify_diagram('C4Context\n  title Test\n  Person(user, "User")\n')
        )
        assert res.ok is False
        assert res.unverifiable is True

    def test_run_repair_loop_with_real_verifier(self):
        import api.formats.mermaid as mv

        async def llm(prompt):
            return "```mermaid\nflowchart TD\n    A --> B\n```"

        md = (
            "```mermaid\nflowchart TD\n  A --> B\n```\n"
            "```mermaid\nflowchart TD\n  A --> > B\n```\n"
            '```mermaid\nC4Context\n  title T\n  Person(u, "U")\n```\n'
        )
        patched, stats = asyncio.run(mv.run_repair_loop(md, llm))
        assert stats["verified"] >= 1
        assert stats["fixed"] == 1
        assert stats["unverifiable"] == 1
        assert "A --> > B" not in patched
        assert "C4Context" in patched
