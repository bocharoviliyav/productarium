#!/usr/bin/env python3
"""Unit tests for the product-scoped expert agent (api.expert) and its
router (api.routers.expert) — Wave 2 scope F.

Runs under pytest (pytest.ini: testpaths=test). No live Ollama / cognee / RLM /
Postgres required: the LLM, cognee recall, and fast-rlm are mocked. The
fallback-artifact-docs test uses an isolated SQLite DB.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any, AsyncIterator, Dict, List

import pytest


# --- Shared fixtures ---------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Isolated SQLite env + temp dirs so tests never touch real services."""
    db_file = tmp_path / "expert_test.db"
    monkeypatch.setenv("DB_PROVIDER", "sqlite")
    monkeypatch.setenv("DB_HOST", str(tmp_path))
    monkeypatch.setenv("DB_NAME", str(db_file))
    monkeypatch.setenv("DB_USERNAME", "")
    monkeypatch.setenv("DB_PASSWORD", "")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    # Auth disabled by default so router tests work without a session cookie.
    # (api.auth snapshots AUTH_PROVIDER at import, so tests that need it patch
    #  api.auth.deps.AUTH_PROVIDER directly.)
    monkeypatch.setenv("AUTH_PROVIDER", "none")
    yield


def _sqlite_db():
    """Reload api.db under the SQLite env and create tables. Returns the module."""
    import api.db as db
    importlib.reload(db)
    db.init_db()
    return db


# --- Fakes -------------------------------------------------------------------
class _FakeLLM:
    """Minimal stand-in for _ExpertLLM used by _safe_build_llm."""

    def __init__(self, text: str = "FAKE ANSWER", chunks: List[str] | None = None):
        self._text = text
        self._chunks = chunks if chunks is not None else [text]

    async def generate(self, prompt: str) -> str:
        return self._text

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        for c in self._chunks:
            yield c


def _patch_llm(monkeypatch, text: str = "FAKE ANSWER", chunks: List[str] | None = None):
    """Patch api.expert.generate._safe_build_llm to return a _FakeLLM.

    ``_generate_answer`` (in ``api.expert.generate``) looks up
    ``_safe_build_llm`` in ``generate``'s globals, so the patch must target
    that submodule (not the ``api.expert`` package).
    """
    fake = _FakeLLM(text=text, chunks=chunks)

    def _builder(*args, **kwargs):
        return fake

    monkeypatch.setattr("api.expert.generate._safe_build_llm", _builder)
    return fake


def _patch_cognee(monkeypatch, payload: str):
    """Patch api.cognee_manager.query_cognee to return ``payload``."""

    async def _fake_query(query: str, dataset_name: str, top_k: int = 20) -> str:
        return payload

    monkeypatch.setattr("api.cognee_manager.query_cognee", _fake_query)


def _patch_rlm(monkeypatch, result: str, success: bool = True):
    """Patch api.rlm.runner.run_rlm_task to return a fixed result dict."""

    async def _fake_rlm(query: str, model_name: str | None = None) -> Dict[str, Any]:
        return {"results": result, "usage": {}, "success": success}

    monkeypatch.setattr("api.rlm.runner.run_rlm_task", _fake_rlm)


# ============================================================================
# Imports & prompt loading
# ============================================================================
class TestImportsAndPrompts:
    def test_modules_import_cleanly(self):
        import api.expert  # noqa: F401
        import api.routers.expert  # noqa: F401

    def test_router_is_apirouter_with_products_prefix(self):
        from fastapi import APIRouter
        import api.routers.expert as mod

        assert isinstance(mod.router, APIRouter)
        assert mod.router.prefix == "/api/products"
        # Both endpoints exist on the router (paths include the router prefix).
        paths = {r.path for r in mod.router.routes}
        assert "/api/products/{product_id}/ask" in paths
        assert "/api/products/{product_id}/ask/doc" in paths

    def test_prompts_loaded_from_refs(self):
        import api.expert as ea
        # The .md files exist and are loaded (non-empty).
        assert ea.EXPERT_SYSTEM_PROMPT, "expert_agent_system.md not loaded"
        assert ea.EXPERT_DOC_PROMPT, "expert_agent_doc.md not loaded"
        # Placeholders are present in the bodies (substituted at runtime).
        assert "{product_name}" in ea.EXPERT_SYSTEM_PROMPT
        assert "{language_name}" in ea.EXPERT_SYSTEM_PROMPT
        assert "{product_name}" in ea.EXPERT_DOC_PROMPT
        assert "{language_name}" in ea.EXPERT_DOC_PROMPT

    def test_public_api_surface(self):
        import api.expert as ea
        for name in ("run_expert_chat", "run_expert_doc"):
            assert hasattr(ea, name)


# ============================================================================
# Helpers
# ============================================================================
class TestHelpers:
    def test_safe_replace_substitutes_and_leaves_unmatched(self):
        import api.expert as ea
        out = ea._safe_replace("a={x} b={y} c={z}", {"x": "1", "y": None})
        assert out == "a=1 b= c={z}"

    def test_safe_replace_empty_template(self):
        import api.expert as ea
        assert ea._safe_replace("", {"x": "1"}) == ""

    def test_clean_llm_text_strips_fences(self):
        import api.expert as ea
        assert ea._clean_llm_text("```markdown\n# T\nbody\n```") == "# T\nbody"
        assert ea._clean_llm_text("  plain  ") == "plain"
        assert ea._clean_llm_text(None) == ""
        assert ea._clean_llm_text("") == ""

    def test_chunk_text_keeps_short_lines_and_splits_long(self):
        import api.expert as ea
        pieces = ea._chunk_text("hi\n" + ("a " * 200), size=20)
        assert pieces[0] == "hi\n"
        # Long line is split into multiple pieces each <= ~20 chars.
        assert len(pieces) > 2
        for p in pieces[1:]:
            assert len(p) <= 21  # word + space tolerance

    def test_format_history_renders_pairs(self):
        import api.expert as ea
        out = ea._format_history([
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "s"},
        ])
        assert "<user>q1</user>" in out
        assert "<assistant>a1</assistant>" in out
        assert "<system>s</system>" in out

    def test_format_history_skips_empty(self):
        import api.expert as ea
        assert ea._format_history([]) == ""
        assert ea._format_history([{"role": "user", "content": ""}]) == ""

    def test_build_prompt_includes_knowledge_and_query(self):
        import api.expert as ea
        prompt = ea._build_prompt(
            ea.EXPERT_SYSTEM_PROMPT, "Acme", "KNOWLEDGE HERE", "<user>hi</user>", "QUESTION?"
        )
        assert "Acme" in prompt
        assert "<product_knowledge>" in prompt
        assert "KNOWLEDGE HERE" in prompt
        assert "<conversation_history>" in prompt
        assert "<user>hi</user>" in prompt
        assert "<query>\nQUESTION?\n</query>" in prompt

    def test_build_prompt_uses_note_when_no_knowledge(self):
        import api.expert as ea
        prompt = ea._build_prompt(ea.EXPERT_DOC_PROMPT, "Acme", "", "", "Q")
        assert "<note>" in prompt
        # No knowledge BLOCK was added. The template body mentions
        # <product_knowledge> in prose, so check for the closing block tag
        # (only present when an actual block is emitted) and for the absence of
        # a populated block.
        assert "</product_knowledge>" not in prompt
        assert "<product_knowledge>\n" not in prompt


# ============================================================================
# Knowledge retrieval (cognee + fallback)
# ============================================================================
class TestKnowledgeRetrieval:
    def test_retrieve_uses_cognee_when_available(self, monkeypatch):
        import api.expert as ea
        _patch_cognee(monkeypatch, "COGNEE CONTEXT")
        out = asyncio.run(ea._retrieve_product_knowledge("prod_1", "q"))
        assert out == "COGNEE CONTEXT"

    def test_retrieve_falls_back_to_artifact_docs(self, monkeypatch):
        import api.expert as ea
        db = _sqlite_db()
        from api.models import ArtifactORM, ProductORM

        with db.SessionLocal() as session:
            session.add(ProductORM(id="prod_1", name="Acme"))
            session.flush()
            session.add(
                ArtifactORM(
                    id="art_1",
                    product_id="prod_1",
                    name="svc",
                    type="codebase",
                    generated_docs="# Svc\nthe docs",
                    pages={"page_overview": {"id": "page_overview", "title": "Overview",
                                             "content": "PAGE CONTENT"}},
                )
            )
            session.commit()

        # cognee empty -> fallback
        _patch_cognee(monkeypatch, "")
        out = asyncio.run(ea._retrieve_product_knowledge("prod_1", "q"))
        assert "the docs" in out
        assert "PAGE CONTENT" in out
        assert "Acme" not in out  # name comes from artifact, not product

    def test_fallback_returns_empty_for_missing_product(self, monkeypatch):
        import api.expert as ea
        db = _sqlite_db()
        _patch_cognee(monkeypatch, "")
        out = asyncio.run(ea._retrieve_product_knowledge("does_not_exist", "q"))
        assert out == ""

    def test_fallback_artifact_docs_non_fatal_on_db_error(self, monkeypatch):
        import api.expert as ea
        # Force SessionLocal to raise by pointing it at a broken callable.
        import api.db as dbmod

        class _BrokenSession:
            def __enter__(self):
                raise RuntimeError("boom")

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(dbmod, "SessionLocal", lambda: _BrokenSession())
        # _fallback_artifact_docs imports SessionLocal lazily from api.db, so
        # patching the attr on the module is enough.
        assert ea._fallback_artifact_docs("prod_x") == ""


# ============================================================================
# run_expert_doc
# ============================================================================
class TestRunExpertDoc:
    def test_returns_markdown_from_llm(self, monkeypatch):
        import api.expert as ea
        _patch_cognee(monkeypatch, "CTX")
        _patch_llm(monkeypatch, text="# Generated Doc\n\nbody text")
        out = asyncio.run(ea.run_expert_doc("prod_1", "summarize the service"))
        assert out.startswith("# Generated Doc")
        assert "body text" in out

    def test_returns_placeholder_when_llm_empty(self, monkeypatch):
        import api.expert as ea
        _patch_cognee(monkeypatch, "CTX")
        _patch_llm(monkeypatch, text="")
        out = asyncio.run(ea.run_expert_doc("prod_1", "q"))
        assert "No content was generated" in out
        assert "prod_1" in out

    def test_rlm_used_for_long_context(self, monkeypatch):
        import api.expert as ea
        import api.settings_store as ss
        monkeypatch.setattr(ss, "get_rlm_mode", lambda task: "auto")
        # Big knowledge -> prompt >= RLM_MIN_CHARS -> RLM path.
        _patch_cognee(monkeypatch, "K" * (ea.RLM_MIN_CHARS + 5000))
        _patch_rlm(monkeypatch, "RLM DOC RESULT")
        # Standard LLM would return this if (incorrectly) used.
        _patch_llm(monkeypatch, text="STANDARD DOC")
        out = asyncio.run(ea.run_expert_doc("prod_1", "deep synthesis question"))
        assert out == "RLM DOC RESULT"


# ============================================================================
# run_expert_chat
# ============================================================================
class TestRunExpertChat:
    def test_collect_returns_full_answer(self, monkeypatch):
        import api.expert as ea
        _patch_cognee(monkeypatch, "CTX")
        _patch_llm(monkeypatch, text="# Answer\nthe body")
        out = asyncio.run(
            ea.run_expert_chat("prod_1", "what is it?", stream=False)
        )
        assert out == "# Answer\nthe body"

    def test_stream_yields_chunks(self, monkeypatch):
        import api.expert as ea
        _patch_cognee(monkeypatch, "CTX")
        _patch_llm(monkeypatch, text="FULL", chunks=["Hel", "lo", " world"])

        async def _collect():
            chunks = []
            async for c in ea.run_expert_chat("prod_1", "hi", stream=True):
                chunks.append(c)
            return chunks

        assert asyncio.run(_collect()) == ["Hel", "lo", " world"]

    def test_stream_uses_rlm_for_long_context(self, monkeypatch):
        import api.expert as ea
        import api.settings_store as ss
        monkeypatch.setattr(ss, "get_rlm_mode", lambda task: "auto")
        _patch_cognee(monkeypatch, "K" * (ea.RLM_MIN_CHARS + 5000))
        _patch_rlm(monkeypatch, "RLM CHUNKED ANSWER")
        _patch_llm(monkeypatch, text="STANDARD", chunks=["SHOULD", "NOT", "HAPPEN"])

        async def _collect():
            out = []
            async for c in ea.run_expert_chat("prod_1", "deep q", stream=True):
                out.append(c)
            return "".join(out)

        assert asyncio.run(_collect()) == "RLM CHUNKED ANSWER"

    def test_stream_rlm_empty_falls_back_to_llm(self, monkeypatch):
        import api.expert as ea
        _patch_cognee(monkeypatch, "K" * (ea.RLM_MIN_CHARS + 5000))
        # RLM fails -> standard LLM stream is used.
        _patch_rlm(monkeypatch, "", success=False)
        _patch_llm(monkeypatch, text="FALLBACK", chunks=["FALL", "BACK"])

        async def _collect():
            out = []
            async for c in ea.run_expert_chat("prod_1", "deep q", stream=True):
                out.append(c)
            return out

        assert asyncio.run(_collect()) == ["FALL", "BACK"]

    def test_collect_no_llm_returns_empty(self, monkeypatch):
        import api.expert as ea
        _patch_cognee(monkeypatch, "CTX")
        # _safe_build_llm returns None -> empty answer.
        monkeypatch.setattr("api.expert.generate._safe_build_llm", lambda *a, **k: None)
        out = asyncio.run(ea.run_expert_chat("prod_1", "q", stream=False))
        assert out == ""

    def test_messages_history_included_in_prompt(self, monkeypatch):
        # ``_run_expert_chat_collect`` lives in ``api.expert.chat`` and looks up
        # ``_generate_answer`` / ``_retrieve_product_knowledge`` /
        # ``_product_name_by_id`` in ``chat``'s globals (not the package's), so
        # the patches must target that use-site submodule.
        import api.expert.chat as chat
        captured: dict = {}

        async def _fake_generate_answer(prompt, provider, model, base_url, api_key, use_rlm):
            captured["prompt"] = prompt
            return "ok"

        monkeypatch.setattr(chat, "_generate_answer", _fake_generate_answer)
        monkeypatch.setattr(chat, "_retrieve_product_knowledge", _async_value("CTX"))
        monkeypatch.setattr(chat, "_product_name_by_id", lambda pid: "Acme")
        asyncio.run(
            chat._run_expert_chat_collect(
                "prod_1", "current q", [{"role": "user", "content": "prior"}], None, None
            )
        )
        assert "<conversation_history>" in captured["prompt"]
        assert "<user>prior</user>" in captured["prompt"]
        assert "<query>\ncurrent q\n</query>" in captured["prompt"]


def _async_value(value):
    async def _ret(*args, **kwargs):
        return value
    return _ret


# ============================================================================
# Router (api.routers.expert) via FastAPI TestClient
# ============================================================================
class TestExpertRouter:
    @pytest.fixture
    def app_and_client(self, monkeypatch):
        # Auth disabled -> get_current_user returns the system user (no cookie).
        import api.auth.deps as deps
        monkeypatch.setattr(deps, "AUTH_PROVIDER", "none")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import api.routers.expert as expert_router

        app = FastAPI()
        app.include_router(expert_router.router)
        return app, TestClient(app)

    def test_ask_streams_sse(self, app_and_client, monkeypatch):
        _, client = app_and_client
        import api.routers.expert as expert_router

        def _fake_chat(product_id, query, messages, model, stream=True, use_rlm=None, **kwargs):
            async def gen():
                yield "Hello"
                yield " world"
            return gen()

        monkeypatch.setattr(expert_router, "run_expert_chat", _fake_chat)
        resp = client.post("/api/products/prod_1/ask", json={"query": "hi"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert 'data: {"content": "Hello"}' in body
        assert 'data: {"content": " world"}' in body
        assert "data: [DONE]" in body

    def test_ask_doc_returns_markdown_file(self, app_and_client, monkeypatch):
        _, client = app_and_client
        import api.routers.expert as expert_router

        async def _fake_doc(product_id, query, model=None, use_rlm=None, **kwargs):
            return "# Title\n\ndoc body"

        monkeypatch.setattr(expert_router, "run_expert_doc", _fake_doc)
        resp = client.post(
            "/api/products/prod_1/ask/doc", json={"query": "write the doc"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        cd = resp.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert "productarium_prod_1_expert.md" in cd
        assert "# Title" in resp.text
        assert "doc body" in resp.text

    def test_ask_doc_empty_query_400(self, app_and_client):
        _, client = app_and_client
        resp = client.post("/api/products/prod_1/ask/doc", json={"query": "   "})
        assert resp.status_code == 400

    def test_ask_empty_query_400(self, app_and_client):
        _, client = app_and_client
        resp = client.post("/api/products/prod_1/ask", json={"query": ""})
        assert resp.status_code == 400

    def test_ask_requires_auth_when_not_none(self, app_and_client, monkeypatch):
        app, client = app_and_client
        import api.auth.deps as deps
        # Flip auth to 'local' with no session cookie -> 401.
        monkeypatch.setattr(deps, "AUTH_PROVIDER", "local")
        resp = client.post("/api/products/prod_1/ask", json={"query": "hi"})
        assert resp.status_code == 401

    def test_ask_doc_requires_auth_when_not_none(self, app_and_client, monkeypatch):
        _, client = app_and_client
        import api.auth.deps as deps
        monkeypatch.setattr(deps, "AUTH_PROVIDER", "local")
        resp = client.post("/api/products/prod_1/ask/doc", json={"query": "hi"})
        assert resp.status_code == 401

    def test_safe_filename_sanitizes(self):
        import api.routers.expert as expert_router
        assert expert_router._safe_filename("prod_1") == "productarium_prod_1_expert.md"
        # Unsafe characters are replaced with underscores.
        assert expert_router._safe_filename("prod a/b") == "productarium_prod_a_b_expert.md"
