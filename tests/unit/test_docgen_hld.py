"""Unit tests for ``api.docgen.hld`` (HLD generation pipeline).

Hermetic: fake LLM / repair loop / judge, checkpoint + indexing stubbed,
context window monkeypatched (no network, no DB).
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

import api.docgen.hld as hld_mod
from api.docgen._common import JobCancelledError
from api.docgen.verification import JudgeVerdict


class FakeCodebase:
    def __init__(self, name, docs):
        self.name = name
        self.generated_docs = docs
        self.id = f"cb_{name}"


class FakeProduct:
    def __init__(self, name="MyProduct", pid="prod_123", codebases=None):
        self.name = name
        self.id = pid
        self.codebases = (
            codebases
            if codebases is not None
            else [FakeCodebase("app", "app docs text")]
        )
        self.databases = []
        self.specs = []
        self.links = []


class FakeHld:
    def __init__(self):
        self.id = "hld_prod_123"
        self.generated_docs = None
        self.pages = None


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = []
        self.closed = False

    async def generate(self, prompt):
        self.calls.append(prompt)
        return self.text

    async def aclose(self):
        self.closed = True


_SECTION_BODY = (
    "Текст раздела про компоненты продукта.\n\n"
    "```mermaid\nflowchart TD\n  A[API] --> B[Core]\n```\n\n"
    "### Провенанс и проверка\n- Источник: codebase app\n- Уверенность: высокая"
)


def _patch_pipeline(monkeypatch, text=_SECTION_BODY):
    llm = FakeLLM(text)
    monkeypatch.setattr(hld_mod, "_safe_build_llm", lambda *a, **k: llm)
    monkeypatch.setattr(hld_mod, "_index_in_background", lambda *a, **k: None)
    checkpoints = []

    def _checkpoint(aid, model, md, pages):
        checkpoints.append((aid, md, pages))
        return True

    monkeypatch.setattr(hld_mod, "_checkpoint_partial_docs", _checkpoint)

    async def fake_ctx(**kw):
        return 32_768

    monkeypatch.setattr("api.utils.get_model_context_window_async", fake_ctx)
    monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "false")

    repairs = []

    async def fake_repair(md, llm_ref, **kw):
        repairs.append(md)
        return md, {"verified": 1, "broken": 0, "unverifiable": 0, "fixed": 0, "failed": 0}

    monkeypatch.setattr(hld_mod, "run_repair_loop", fake_repair)
    return llm, checkpoints, repairs


def _run(monkeypatch, entity=None, product=None, text=_SECTION_BODY, **kw):
    llm, checkpoints, repairs = _patch_pipeline(monkeypatch, text)
    entity = entity if entity is not None else FakeHld()
    product = product if product is not None else FakeProduct()
    docs = asyncio.run(
        hld_mod.generate_hld_docs(entity, product, **kw)
    )
    return llm, checkpoints, repairs, entity, docs


class TestGenerateHldDocs:
    def test_pages_structure_and_prompt(self, monkeypatch):
        progress_events = []
        llm, checkpoints, repairs, entity, docs = _run(
            monkeypatch, progress=lambda **f: progress_events.append(f)
        )
        assert set(entity.pages) == {
            f"hld_{s['id']}" for s in hld_mod.HLD_SECTIONS
        }
        overview = entity.pages["hld_overview"]
        assert overview["id"] == "hld_overview"
        assert overview["title"] == "Обзор продукта"
        assert overview["importance"] == "high"
        assert entity.pages["hld_architecture"]["importance"] == "medium"
        assert {"id", "title", "content", "parent", "importance",
                "relatedPages", "filePaths", "provenance"} <= set(overview)

        prov = overview["provenance"]
        assert prov["prompt_file"] == "hld_page.md"
        assert prov["regen"] == "generated"
        assert prov["section_id"] == "overview"
        assert prov["mermaid"]["verified"] == 1

        # The LLM provenance block is split out of the page text.
        assert "Провенанс" not in overview["content"]
        assert prov["report"].startswith("- Источник")

        assert "## Обзор продукта" in entity.generated_docs
        assert docs == entity.generated_docs

        assert len(llm.calls) == 6
        first = llm.calls[0]
        assert "MyProduct" in first
        assert "app docs text" in first
        assert "Назначение продукта" in first  # section instructions
        assert "Russian (Русский)" in first
        assert "{context}" not in first and "{section_title}" not in first

        assert len(repairs) == 6  # mermaid stage ran per section
        assert len(checkpoints) == 6
        assert len(checkpoints[-1][2]) == 6  # pages grow per checkpoint
        assert checkpoints[0][0] == "hld_prod_123"

        phases = [e.get("phase") for e in progress_events if e.get("phase")]
        assert phases[0] == "planning"
        assert "sections" in phases and "verifying" in phases

    def test_english_titles(self, monkeypatch):
        _, _, _, entity, _ = _run(monkeypatch, language="en")
        assert entity.pages["hld_architecture"]["title"] == "Architecture"

    def test_secret_masking(self, monkeypatch):
        text = "Ключ: ghp_abcdefghijklmnopqrst\n\n" + _SECTION_BODY
        _, _, _, entity, _ = _run(monkeypatch, text=text)
        page = entity.pages["hld_overview"]
        assert "ghp_" not in page["content"]
        assert page["provenance"]["secrets_masked"] == 1

    def test_empty_context_raises(self, monkeypatch):
        _patch_pipeline(monkeypatch)
        with pytest.raises(ValueError, match="нет исходных данных"):
            asyncio.run(
                hld_mod.generate_hld_docs(FakeHld(), FakeProduct(codebases=[]))
            )

    def test_llm_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr(hld_mod, "_safe_build_llm", lambda *a, **k: None)
        with pytest.raises(ValueError, match="LLM"):
            asyncio.run(
                hld_mod.generate_hld_docs(FakeHld(), FakeProduct())
            )

    def test_cancel_mid_run(self, monkeypatch):
        llm, checkpoints, _ = _patch_pipeline(monkeypatch)
        state = {"n": 0}

        def should_cancel():
            state["n"] += 1
            return state["n"] >= 3  # cancel at the start of section 3

        entity = FakeHld()
        with pytest.raises(JobCancelledError):
            asyncio.run(
                hld_mod.generate_hld_docs(
                    entity, FakeProduct(), should_cancel=should_cancel
                )
            )
        assert len(checkpoints) == 2  # two sections checkpointed before Stop
        assert entity.pages is None  # nothing persisted
        assert llm.closed is True  # client released even on cancel

    def test_judge_verdict_recorded(self, monkeypatch):
        _patch_pipeline(monkeypatch)
        monkeypatch.setenv("DOCGEN_JUDGE_ENABLED", "true")

        async def fake_judge(section_id, draft, evidence, *, model=None):
            assert "app docs text" in evidence  # ground truth = context
            return JudgeVerdict(verdict="inconsistent", issues=["fabricated table"])

        monkeypatch.setattr(hld_mod, "judge_section", fake_judge)
        entity = FakeHld()
        asyncio.run(hld_mod.generate_hld_docs(entity, FakeProduct()))
        assert entity.pages["hld_overview"]["provenance"]["judge"] == {
            "verdict": "inconsistent",
            "issues": ["fabricated table"],
        }

    def test_verify_flags_kept_on_identical_regen(self, monkeypatch):
        _, _, _, entity, _ = _run(monkeypatch)
        verified = {
            **entity.pages["hld_overview"],
            "verified": True, "verified_by": "u1",
            "verified_at": "2026-01-01T00:00:00",
        }
        entity.pages = {**entity.pages, "hld_overview": verified}
        _, _, _, entity, _ = _run(monkeypatch, entity=entity)
        assert entity.pages["hld_overview"].get("verified") is True

    def test_verify_flags_reset_on_changed_content(self, monkeypatch):
        _, _, _, entity, _ = _run(monkeypatch)
        verified = {
            **entity.pages["hld_overview"],
            "verified": True, "verified_by": "u1",
            "verified_at": "2026-01-01T00:00:00",
        }
        entity.pages = {**entity.pages, "hld_overview": verified}
        other = _SECTION_BODY.replace("Текст раздела", "Другой текст")
        _, _, _, entity, _ = _run(monkeypatch, entity=entity, text=other)
        assert "verified" not in entity.pages["hld_overview"]


class TestHldLightPayload:
    def test_strip_page_content_hld(self):
        from api.repositories.product_repo import strip_page_content
        from api.schemas import Hld, Product

        product = Product(
            id="p", name="P",
            hld=Hld(
                id="hld_p", generated_docs="big blob",
                pages={
                    "hld_overview": {
                        "id": "hld_overview", "title": "Обзор",
                        "content": "body", "provenance": {"x": 1},
                        "importance": "high",
                    },
                },
            ),
        )
        light = strip_page_content(product)
        assert light.hld.generated_docs is None
        page = light.hld.pages["hld_overview"]
        assert "content" not in page and "provenance" not in page
        assert page["title"] == "Обзор" and page["importance"] == "high"

    def test_pageless_hld_keeps_blob(self):
        from api.repositories.product_repo import strip_page_content
        from api.schemas import Hld, Product

        light = strip_page_content(
            Product(id="p", name="P",
                    hld=Hld(id="hld_p", generated_docs="blob", pages={}))
        )
        assert light.hld.generated_docs == "blob"
