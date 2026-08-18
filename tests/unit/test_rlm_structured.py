"""Unit tests for the RLM structured path (``run_rlm_structured_sync``) and the
RLM retrieval tools (``api.rlm.tools``).

Covers:
- ``run_rlm_structured_sync``: dict query passed through, tools/env_variables/
  session_dir/session_id forwarded to ``run()``, per-scenario budgets applied,
  graceful fallback to flat-string ``run_rlm_task_sync`` when ``run()`` rejects
  the new kwargs (TypeError), and the unavailable path.
- ``api.rlm.tools``: ``build_expert_tools`` / ``build_docgen_tools`` return the
  expected self-contained callables; ``resolve_env_variables`` carries
  PRODUCT_ID / CODEBASE_ID / RLM_API_BASE.
- ``get_rlm_session_dir``: creates + returns a per-scope directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

import api.rlm.runner as runner
import api.rlm.tools as rlm_tools


# --------------------------------------------------------------------------- #
# Shared fixture (mirrors test_rlm_runner.py's rlm_deps so the structured path
# has the same stubbed admin config / context window / RLMConfig).
# --------------------------------------------------------------------------- #
@pytest.fixture
def rlm_deps(monkeypatch):
    monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", True)

    config = MagicMock()
    config.max_prompt_tokens = 200000
    config.max_completion_tokens = 50000
    config.api_timeout_ms = 3600000
    config_class = MagicMock()
    config_class.default.return_value = config
    monkeypatch.setattr(runner, "RLMConfig", config_class)

    captured: dict = {}

    def _default_run(query, config=None, verbose=False, **kwargs):
        captured["query"] = query
        captured["kwargs"] = kwargs
        return {"results": "ok", "usage": {}}

    run_mock = MagicMock(side_effect=_default_run)
    monkeypatch.setattr(runner, "run", run_mock)

    admin_cfg = {
        "model": "test-model",
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-needed",
    }
    # These patches mirror test_rlm_runner.py, but are best-effort: when
    # test_expert_generate.py is collected in the same run it installs a stub
    # ``api.config`` module (no ``settings``/``ssl`` submodules), which makes
    # the string-form monkeypatch.setattr and the cognee import raise during
    # fixture setup. _resolve_rlm_config already imports these defensively
    # inside try/except, so skipping the patch here doesn't affect what the
    # structured-path tests assert on (query/tools/env/session/budgets).
    try:
        monkeypatch.setattr(
            "api.config.settings.get_model_for_task",
            lambda task: dict(admin_cfg),
        )
    except Exception:
        pass
    try:
        monkeypatch.setattr(
            "api.utils.get_model_context_window",
            lambda **kw: 8192,
        )
    except Exception:
        pass
    try:
        import api.cognee._runtime as rt
        monkeypatch.setattr(rt, "_host_to_v1", lambda h: h if h.endswith("/v1") else h.rstrip("/") + "/v1")
        monkeypatch.setattr(rt, "_strip_provider_prefix", lambda m: m)
    except Exception:
        pass

    deps = SimpleNamespace(config=config, captured=captured, run_mock=run_mock, admin_cfg=admin_cfg)

    def set_run(fn):
        run_mock.side_effect = fn

    deps.set_run = set_run
    return deps


# --------------------------------------------------------------------------- #
# run_rlm_structured_sync
# --------------------------------------------------------------------------- #
class TestRunRlmStructured:
    def test_unavailable_returns_failure(self, monkeypatch):
        monkeypatch.setattr(runner, "_FAST_RLM_AVAILABLE", False)
        result = runner.run_rlm_structured_sync({"task": "x"}, "model")
        assert result["success"] is False
        assert "not available" in result["results"]

    def test_dict_query_forwarded(self, rlm_deps):
        query = {"task": "do thing", "product": "p1", "query": "q"}
        result = runner.run_rlm_structured_sync(query, "test-model", task="expert")
        assert result["success"] is True
        # The dict query must be passed through to run() verbatim.
        assert rlm_deps.captured["query"] == query

    def test_tools_env_session_forwarded(self, rlm_deps):
        def _capture(query, config=None, verbose=False, **kwargs):
            rlm_deps.captured["tools"] = kwargs.get("tools")
            rlm_deps.captured["env_variables"] = kwargs.get("env_variables")
            rlm_deps.captured["session_dir"] = kwargs.get("session_dir")
            rlm_deps.captured["session_id"] = kwargs.get("session_id")
            return {"results": "ok", "usage": {}}
        rlm_deps.set_run(_capture)

        def _tool():
            pass

        runner.run_rlm_structured_sync(
            {"task": "x"},
            "test-model",
            tools=[_tool],
            env_variables={"PRODUCT_ID": "p1"},
            session_dir="/tmp/sess",
            session_id="p1",
            task="expert",
        )
        assert rlm_deps.captured["tools"] == [_tool]
        assert rlm_deps.captured["env_variables"] == {"PRODUCT_ID": "p1"}
        assert rlm_deps.captured["session_dir"] == "/tmp/sess"
        assert rlm_deps.captured["session_id"] == "p1"

    def test_expert_budgets_applied(self, rlm_deps):
        def _capture(query, config=None, verbose=False, **kwargs):
            rlm_deps.captured["max_depth"] = config.max_depth
            rlm_deps.captured["max_calls"] = config.max_calls_per_subagent
            return {"results": "ok", "usage": {}}
        rlm_deps.set_run(_capture)

        runner.run_rlm_structured_sync({"task": "x"}, "test-model", task="expert")
        assert rlm_deps.captured["max_depth"] == runner.RLM_EXPERT_MAX_DEPTH
        assert rlm_deps.captured["max_calls"] == runner.RLM_EXPERT_MAX_CALLS

    def test_docgen_budgets_applied(self, rlm_deps):
        def _capture(query, config=None, verbose=False, **kwargs):
            rlm_deps.captured["max_depth"] = config.max_depth
            rlm_deps.captured["max_calls"] = config.max_calls_per_subagent
            return {"results": "ok", "usage": {}}
        rlm_deps.set_run(_capture)

        runner.run_rlm_structured_sync({"task": "x"}, "test-model", task="docgen")
        assert rlm_deps.captured["max_depth"] == runner.RLM_DOCGEN_MAX_DEPTH
        assert rlm_deps.captured["max_calls"] == runner.RLM_DOCGEN_MAX_CALLS

    def test_explicit_overrides_win(self, rlm_deps):
        def _capture(query, config=None, verbose=False, **kwargs):
            rlm_deps.captured["max_depth"] = config.max_depth
            rlm_deps.captured["max_calls"] = config.max_calls_per_subagent
            return {"results": "ok", "usage": {}}
        rlm_deps.set_run(_capture)

        runner.run_rlm_structured_sync(
            {"task": "x"}, "test-model",
            max_depth=7, max_calls_per_subagent=99, task="expert",
        )
        assert rlm_deps.captured["max_depth"] == 7
        assert rlm_deps.captured["max_calls"] == 99

    def test_typeerror_falls_back_to_flat_string(self, rlm_deps, monkeypatch):
        """When run() rejects a structured kwarg (older fast-rlm), fall back to
        the flat-string run_rlm_task path instead of crashing."""
        flat_calls: list = []

        def _raise_then_flat(query, config=None, verbose=False, **kwargs):
            if kwargs:
                raise TypeError("run() got an unexpected keyword argument 'tools'")
            flat_calls.append(query)
            return {"results": "flat ok", "usage": {}}
        rlm_deps.set_run(_raise_then_flat)

        result = runner.run_rlm_structured_sync(
            {"task": "x"}, "test-model", tools=[lambda: None], task="expert",
        )
        assert result["success"] is True
        assert result["results"] == "flat ok"
        # The fallback stringifies the dict query.
        assert len(flat_calls) == 1
        assert isinstance(flat_calls[0], str)

    def test_run_exception_returns_failure(self, rlm_deps):
        def _boom(query, config=None, verbose=False, **kwargs):
            raise RuntimeError("connection error")
        rlm_deps.set_run(_boom)

        result = runner.run_rlm_structured_sync({"task": "x"}, "test-model")
        assert result["success"] is False
        assert "Failed to execute structured RLM task" in result["results"]


# --------------------------------------------------------------------------- #
# Async wrapper
# --------------------------------------------------------------------------- #
class TestRunRlmStructuredAsync:
    @pytest.mark.asyncio
    async def test_async_delegates_to_sync(self, rlm_deps):
        rlm_deps.set_run(lambda query, config=None, verbose=False, **kw: {"results": "async ok", "usage": {}})
        result = await runner.run_rlm_structured({"task": "x"}, "test-model", task="expert")
        assert result["success"] is True
        assert result["results"] == "async ok"


# --------------------------------------------------------------------------- #
# get_rlm_session_dir
# --------------------------------------------------------------------------- #
class TestGetRlmSessionDir:
    def test_creates_and_returns_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        d = runner.get_rlm_session_dir("expert_prod_123")
        assert d.endswith("rlm_sessions/expert_prod_123")
        assert os.path.isdir(d)

    def test_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        d1 = runner.get_rlm_session_dir("docgen_cb_1")
        d2 = runner.get_rlm_session_dir("docgen_cb_1")
        assert d1 == d2


# --------------------------------------------------------------------------- #
# Tools assembly (api.rlm.tools)
# --------------------------------------------------------------------------- #
class TestToolsAssembly:
    def test_build_expert_tools_returns_callables(self):
        tools = rlm_tools.build_expert_tools()
        assert len(tools) == 8
        for t in tools:
            assert callable(t)

    def test_build_docgen_tools_returns_callables(self):
        tools = rlm_tools.build_docgen_tools()
        assert len(tools) == 7
        for t in tools:
            assert callable(t)

    def test_expert_tools_include_search_knowledge(self):
        names = [t.__name__ for t in rlm_tools.build_expert_tools()]
        assert "search_knowledge" in names
        assert "read_codebase_file" in names
        assert "get_specs" in names

    def test_docgen_tools_include_read_file(self):
        names = [t.__name__ for t in rlm_tools.build_docgen_tools()]
        assert "read_codebase_file" in names
        assert "list_codebase_files" in names
        assert "search_code" in names

    def test_resolve_env_variables_carries_ids(self, monkeypatch):
        monkeypatch.setenv("PORT", "8001")
        env = rlm_tools.resolve_env_variables("prod_1", "cb_1")
        assert env["PRODUCT_ID"] == "prod_1"
        assert env["CODEBASE_ID"] == "cb_1"
        assert "RLM_API_BASE" in env
        assert "8001" in env["RLM_API_BASE"]

    def test_resolve_env_variables_no_codebase(self, monkeypatch):
        monkeypatch.setenv("PORT", "9000")
        env = rlm_tools.resolve_env_variables("prod_1")
        assert env["PRODUCT_ID"] == "prod_1"
        assert "CODEBASE_ID" not in env

    def test_resolve_env_variables_override_base(self, monkeypatch):
        monkeypatch.setenv("RLM_API_BASE", "http://my-host:9999")
        env = rlm_tools.resolve_env_variables("prod_1", "cb_1")
        assert env["RLM_API_BASE"] == "http://my-host:9999"


# --------------------------------------------------------------------------- #
# Tool functions are self-contained (no closure over call-site state). They
# cannot run in the host (they import pyodide.http inside the body), but we can
# verify they degrade gracefully (return [] / "") when pyodide is absent.
# --------------------------------------------------------------------------- #
class TestToolsDefensive:
    def test_search_knowledge_no_env_returns_empty(self, monkeypatch):
        # No RLM_API_BASE / PRODUCT_ID in env -> tool returns "" without raising.
        monkeypatch.delenv("RLM_API_BASE", raising=False)
        monkeypatch.delenv("PRODUCT_ID", raising=False)
        assert rlm_tools.search_knowledge("q") == ""

    def test_list_codebase_files_no_env_returns_empty(self, monkeypatch):
        monkeypatch.delenv("RLM_API_BASE", raising=False)
        monkeypatch.delenv("CODEBASE_ID", raising=False)
        assert rlm_tools.list_codebase_files() == []

    def test_read_codebase_file_no_env_returns_empty(self, monkeypatch):
        monkeypatch.delenv("RLM_API_BASE", raising=False)
        monkeypatch.delenv("CODEBASE_ID", raising=False)
        assert rlm_tools.read_codebase_file("src/main.py") == ""

    def test_search_code_no_env_returns_empty(self, monkeypatch):
        monkeypatch.delenv("RLM_API_BASE", raising=False)
        monkeypatch.delenv("CODEBASE_ID", raising=False)
        assert rlm_tools.search_code("foo") == []

    def test_get_specs_no_env_returns_empty(self, monkeypatch):
        monkeypatch.delenv("RLM_API_BASE", raising=False)
        monkeypatch.delenv("PRODUCT_ID", raising=False)
        assert rlm_tools.get_specs() == []

    def test_get_links_no_env_returns_empty(self, monkeypatch):
        monkeypatch.delenv("RLM_API_BASE", raising=False)
        monkeypatch.delenv("PRODUCT_ID", raising=False)
        assert rlm_tools.get_links() == []

    def test_get_knowledge_nodes_no_env_returns_empty(self, monkeypatch):
        monkeypatch.delenv("RLM_API_BASE", raising=False)
        monkeypatch.delenv("PRODUCT_ID", raising=False)
        assert rlm_tools.get_knowledge_nodes() == []

    def test_get_codebases_no_env_returns_empty(self, monkeypatch):
        monkeypatch.delenv("RLM_API_BASE", raising=False)
        monkeypatch.delenv("PRODUCT_ID", raising=False)
        assert rlm_tools.get_codebases() == []
