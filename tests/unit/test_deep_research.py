"""Unit tests for ``api.expert.deep_research`` (Wave E) + the ``deep_research``
flag on the expert /ask router.

Hermetic: every LLM/agent seam is monkeypatched (``_invoke_chat`` /
``_run_research_agent`` / ``_gather_research_tools`` / graph build), so no
LangGraph react agent, no MCP, no LLM server is needed. Node-level tests use
``_FakeChat`` (a plain async ``ainvoke``) through the real ``_invoke_chat``.

Covers:
- helpers: ``_DECISION_RE``, ``_history_block``, ``_findings_block``, ``_render``
- nodes: planner (decision parse + strip, default CONTINUE, iteration hard
  stop, reasoning cap), ``_route_after_planner``, ``_run_straight_line``
- ``run_deep_research_stream`` (graph path + straight-line fallback):
  immediate SYNTHESIZE, one research round (tool events + findings flow into
  the next planner and synthesizer prompts), MAX_ITERATIONS hard stop,
  error event, empty-final fallback, history injection
- router: ``deep_research`` flag routes to the deep runner (session created,
  SSE frames, [DONE], transcript persisted), no flag → regular runner,
  unknown product streams statelessly.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import api.agents.expert as agents_expert_mod
import api.expert.deep_research as dr_mod
import api.routers.expert as expert_router_module
from api.expert.types import (
    EVENT_ANSWERING,
    EVENT_CONTENT,
    EVENT_ERROR,
    EVENT_REASONING,
    EVENT_STATUS,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    ExpertStreamEvent,
)
from api.models import ProductORM
from tests.conftest import build_test_client

QUERY = "How does Widget store data?"


# --------------------------------------------------------------------------- #
# Fakes & helpers
# --------------------------------------------------------------------------- #
class _FakeChat:
    """Minimal chat stand-in for node-level tests (real _invoke_chat path)."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.prompts: List[str] = []

    async def ainvoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _patch_flow(
    monkeypatch,
    *,
    chat_responses: List[Any],
    researcher_results: Optional[List[Tuple[str, List[Tuple[str, str]]]]] = None,
    tools: Optional[Tuple[str, ...]] = ("TOOL_A",),
    prompts: Optional[List[str]] = None,
    researcher_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Patch every deep-research seam; return a dict with the close recorder.

    ``chat_responses`` are consumed in call order by BOTH the planner and the
    synthesizer (an ``Exception`` item is raised instead of returned), so the
    caller lists exactly one item per expected ``_invoke_chat`` call — an
    IndexError means "unexpectedly many LLM calls".
    """
    researcher_results = list(researcher_results or [])
    closed: List[Any] = []

    async def fake_gather(product_id, session_factory=None):
        return list(tools or [])

    async def fake_invoke(chat, prompt):
        if prompts is not None:
            prompts.append(prompt)
        item = chat_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def fake_research(chat, tool_list, system_prompt, task):
        if researcher_calls is not None:
            researcher_calls.append(
                {
                    "tools": list(tool_list),
                    "system_prompt": system_prompt,
                    "task": task,
                }
            )
        return researcher_results.pop(0)

    async def fake_close(chat):
        closed.append(chat)

    def fake_name(product_id):
        return "Widget"

    monkeypatch.setattr(dr_mod, "_gather_research_tools", fake_gather)
    monkeypatch.setattr(dr_mod, "_invoke_chat", fake_invoke)
    monkeypatch.setattr(dr_mod, "_run_research_agent", fake_research)
    monkeypatch.setattr(dr_mod, "_close_model_client", fake_close)
    monkeypatch.setattr(agents_expert_mod, "_product_name_by_id", fake_name)
    return {"closed": closed}


def _collect(**kwargs) -> List[ExpertStreamEvent]:
    async def _run():
        return [
            event
            async for event in dr_mod.run_deep_research_stream(
                "prod_1", QUERY, chat=object(), **kwargs
            )
        ]

    return asyncio.run(_run())


def _statuses(events: List[ExpertStreamEvent]) -> List[str]:
    return [e.content for e in events if e.type == EVENT_STATUS]


def _answer(events: List[ExpertStreamEvent]) -> str:
    return "".join(e.content for e in events if e.type == EVENT_CONTENT)


def _tool_event(events: List[ExpertStreamEvent], etype: str) -> Optional[ExpertStreamEvent]:
    return next((e for e in events if e.type == etype), None)


# --------------------------------------------------------------------------- #
# _gather_research_tools: MCP tools never shadow built-ins (review #4)
# --------------------------------------------------------------------------- #
class TestGatherResearchToolsDedup:
    def test_mcp_shadowing_builtin_dropped(self, monkeypatch):
        import api.agents.tools as agents_tools_mod

        monkeypatch.setattr(
            agents_tools_mod,
            "build_expert_tools",
            lambda pid, session_factory=None: [
                SimpleNamespace(name="search_knowledge"),
            ],
        )

        async def fake_mcp(pid, session_factory=None):
            return [
                SimpleNamespace(name="search_knowledge"),  # shadows builtin
                SimpleNamespace(name="mcp_extra"),
            ]

        monkeypatch.setattr(agents_expert_mod, "_gather_mcp_tools", fake_mcp)
        out = asyncio.run(dr_mod._gather_research_tools("prod_1"))
        assert [getattr(t, "name", "") for t in out] == ["search_knowledge", "mcp_extra"]

    def test_mcp_failure_is_best_effort(self, monkeypatch):
        import api.agents.tools as agents_tools_mod

        monkeypatch.setattr(
            agents_tools_mod,
            "build_expert_tools",
            lambda pid, session_factory=None: [SimpleNamespace(name="TOOL_A")],
        )

        async def boom(pid, session_factory=None):
            raise RuntimeError("mcp down")

        monkeypatch.setattr(agents_expert_mod, "_gather_mcp_tools", boom)
        out = asyncio.run(dr_mod._gather_research_tools("prod_1"))
        assert [getattr(t, "name", "") for t in out] == ["TOOL_A"]


# --------------------------------------------------------------------------- #
# Deadline budget (review #4: wall-clock cap on one Deep Research turn)
# --------------------------------------------------------------------------- #
class TestDeepResearchDeadline:
    def test_planner_deadline_forces_synthesize_without_llm(self, monkeypatch):
        async def fail_invoke(chat, prompt):
            raise AssertionError("planner must not call the LLM past the deadline")

        monkeypatch.setattr(dr_mod, "_invoke_chat", fail_invoke)
        update = asyncio.run(
            dr_mod._planner_node({
                "iteration": 0,
                "findings": ["f1"],
                "deadline": time.monotonic() - 1.0,
            })
        )
        assert update["decision"] == "synthesize"
        assert update["iteration"] == 1
        assert update["events"][0] == (
            dr_mod.EVENT_STATUS, dr_mod.STATUS_PLANNING,
        )
        assert "budget" in update["plan"]

    def test_planner_without_deadline_uses_llm(self):
        # No deadline in state (defensive): the normal planner path runs.
        chat = _FakeChat(["plan body\nDECISION: SYNTHESIZE"])
        update = asyncio.run(
            dr_mod._planner_node({
                "iteration": 0,
                "findings": [],
                "query": "q",
                "chat": chat,
            })
        )
        assert update["decision"] == "synthesize"
        assert update["plan"] == "plan body"

    def test_researcher_deadline_aborts_and_records(self, monkeypatch):
        async def slow_research(chat, tools, system_prompt, task):
            await asyncio.sleep(30)
            return "never", []

        monkeypatch.setattr(dr_mod, "_run_research_agent", slow_research)
        update = asyncio.run(
            dr_mod._researcher_node({
                "iteration": 1,
                "plan": "p",
                "chat": object(),
                "tools": [],
                "findings": [],
                "deadline": time.monotonic() + 0.05,
            })
        )
        assert "time budget" in update["findings"][0]
        assert update["events"][0] == (
            dr_mod.EVENT_STATUS, dr_mod.STATUS_RESEARCHING,
        )

    def test_expired_budget_skips_research_entirely(self, monkeypatch):
        # Negative budget → deadline already in the past: the planner force-
        # synthesizes and the researcher (patched to fail loudly) never runs.
        monkeypatch.setattr(dr_mod, "DEEP_RESEARCH_TIMEOUT_SECONDS", -1.0)
        _patch_flow(monkeypatch, chat_responses=["FINAL ANSWER"])
        events = _collect()
        assert _answer(events) == "FINAL ANSWER"
        statuses = _statuses(events)
        assert dr_mod.STATUS_RESEARCHING not in statuses
        assert dr_mod.STATUS_PLANNING in statuses
        assert dr_mod.STATUS_SYNTHESIZING in statuses


# --------------------------------------------------------------------------- #
# Helpers: _DECISION_RE / _history_block / _findings_block / _render
# --------------------------------------------------------------------------- #
class TestDecisionRegex:
    def test_matches_continue_and_synthesize(self):
        for text, expected in (
            ("DECISION: CONTINUE", "CONTINUE"),
            ("DECISION: SYNTHESIZE", "SYNTHESIZE"),
        ):
            m = dr_mod._DECISION_RE.search(text)
            assert m is not None
            assert m.group(1) == expected

    def test_case_insensitive_and_trailing_whitespace(self):
        assert dr_mod._DECISION_RE.search("decision: synthesize  \n") is not None
        assert dr_mod._DECISION_RE.search("plan body\nDECISION: continue") is not None

    def test_no_match_with_trailing_text(self):
        assert dr_mod._DECISION_RE.search("DECISION: CONTINUE (maybe)") is None
        assert dr_mod._DECISION_RE.search("DECISION: CONTINUE\nmore text") is None

    def test_no_match_without_keyword(self):
        assert dr_mod._DECISION_RE.search("Let us continue researching.") is None


class TestHistoryBlock:
    def test_none_and_empty_are_placeholder(self):
        assert dr_mod._history_block(None) == "(none)"
        assert dr_mod._history_block([]) == "(none)"

    def test_renders_user_assistant_lines(self):
        block = dr_mod._history_block(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        )
        assert block == "user: hello\nassistant: hi there"

    def test_filters_unknown_roles_and_blank_content(self):
        block = dr_mod._history_block(
            [
                {"role": "system", "content": "sysprompt"},
                {"role": "user", "content": "   "},
                {"role": "user", "content": "kept"},
            ]
        )
        assert block == "user: kept"

    def test_keeps_only_last_ten_messages(self):
        history = [{"role": "user", "content": f"m{i}"} for i in range(15)]
        block = dr_mod._history_block(history)
        assert "m5" in block and "m14" in block
        assert "m4" not in block

    def test_truncates_to_cap_from_the_head(self):
        history = [
            {"role": "user", "content": "x" * (dr_mod.MAX_HISTORY_CHARS + 50)}
        ]
        block = dr_mod._history_block(history)
        assert block.startswith("...(truncated)\n")
        assert len(block) <= len("...(truncated)\n") + dr_mod.MAX_HISTORY_CHARS
        assert block.endswith("x")


class TestFindingsBlock:
    def test_empty_is_placeholder(self):
        assert dr_mod._findings_block([]) == "(no findings yet)"

    def test_numbered_findings(self):
        block = dr_mod._findings_block(["first", "second"])
        assert block == "[1] first\n\n[2] second"

    def test_keeps_only_last_max_findings_renumbered(self):
        findings = [f"f{i}" for i in range(1, dr_mod.MAX_FINDINGS + 3)]
        block = dr_mod._findings_block(findings)
        assert block.startswith(f"[1] f3")
        assert f"[{dr_mod.MAX_FINDINGS}] f{dr_mod.MAX_FINDINGS + 2}" in block
        assert "f1\n" not in block and "f2\n" not in block


class TestRenderAndTemplates:
    def test_render_replaces_only_known_placeholders(self):
        out = dr_mod._render("a {x} {y} {{literal}}", {"x": "1"})
        assert out == "a 1 {y} {{literal}}"

    def test_planner_template_loads_md_body(self):
        template = dr_mod._load_template(
            "deep_research_planner.md", dr_mod._PLANNER_FALLBACK
        )
        assert "<findings_so_far>" in template
        assert "{findings}" in template


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
class TestPlannerNode:
    def _state(self, chat, **overrides) -> Dict[str, Any]:
        state = {
            "query": QUERY,
            "product_name": "Widget",
            "history_block": "(none)",
            "chat": chat,
            "findings": [],
            "iteration": 0,
        }
        state.update(overrides)
        return state

    def test_parses_decision_and_strips_it_from_plan(self):
        chat = _FakeChat(["Plan: inspect tables\nDECISION: SYNTHESIZE"])
        update = asyncio.run(dr_mod._planner_node(self._state(chat)))
        assert update["iteration"] == 1
        assert update["decision"] == "synthesize"
        assert update["plan"] == "Plan: inspect tables"
        assert update["events"][0] == (EVENT_STATUS, dr_mod.STATUS_PLANNING)
        assert update["events"][1][0] == EVENT_REASONING
        assert update["events"][1][1] == "Plan: inspect tables"
        # Prompt carried the query, product name and the empty-findings block.
        assert QUERY in chat.prompts[0]
        assert "Widget" in chat.prompts[0]
        assert "(no findings yet)" in chat.prompts[0]

    def test_missing_decision_defaults_to_continue(self):
        chat = _FakeChat(["just a plan, no decision line"])
        update = asyncio.run(dr_mod._planner_node(self._state(chat)))
        assert update["decision"] == "continue"
        assert update["plan"] == "just a plan, no decision line"

    def test_iteration_hard_stop_forces_synthesize(self):
        chat = _FakeChat(["more\nDECISION: CONTINUE"])
        update = asyncio.run(
            dr_mod._planner_node(self._state(chat, iteration=dr_mod.MAX_ITERATIONS - 1))
        )
        assert update["iteration"] == dr_mod.MAX_ITERATIONS
        assert update["decision"] == "synthesize"

    def test_reasoning_event_capped_at_4000_chars(self):
        chat = _FakeChat([f"{'p' * 9000}\nDECISION: SYNTHESIZE"])
        update = asyncio.run(dr_mod._planner_node(self._state(chat)))
        assert update["events"][1][0] == EVENT_REASONING
        assert len(update["events"][1][1]) == 4000


class TestRouteAfterPlanner:
    def test_routes_by_decision(self):
        assert dr_mod._route_after_planner({"decision": "synthesize"}) == "synthesize"
        assert dr_mod._route_after_planner({"decision": "SYNTHESIZE"}) == "synthesize"
        assert dr_mod._route_after_planner({"decision": "continue"}) == "research"
        assert dr_mod._route_after_planner({}) == "research"


class TestRunStraightLine:
    def test_runs_plan_research_then_synthesize(self, monkeypatch):
        async def fake_research(chat, tools, system_prompt, task):
            return "F1", []

        monkeypatch.setattr(dr_mod, "_run_research_agent", fake_research)
        chat = _FakeChat(
            [
                "round one\nDECISION: CONTINUE",
                "enough\nDECISION: SYNTHESIZE",
                "FINAL",
            ]
        )
        state = {
            "query": QUERY,
            "product_name": "Widget",
            "history_block": "(none)",
            "chat": chat,
            "tools": [],
            "findings": [],
            "iteration": 0,
        }
        final = asyncio.run(dr_mod._run_straight_line(state))
        assert final["final"] == "FINAL"
        assert final["findings"] == ["F1"]
        assert final["iteration"] == 2
        # Event batches collected for SSE parity with the graph path.
        assert final["collected_events"][0] == (EVENT_STATUS, dr_mod.STATUS_PLANNING)
        assert final["collected_events"][-1] == (EVENT_STATUS, dr_mod.STATUS_SYNTHESIZING)


# --------------------------------------------------------------------------- #
# run_deep_research_stream (graph path + straight-line fallback)
# --------------------------------------------------------------------------- #
class TestRunDeepResearchStream:
    def test_immediate_synthesize_skips_researcher(self, monkeypatch):
        researcher_calls: List[Dict[str, Any]] = []
        prompts: List[str] = []
        rec = _patch_flow(
            monkeypatch,
            chat_responses=[
                "Findings look complete.\nDECISION: SYNTHESIZE",
                "# FINAL ANSWER",
            ],
            researcher_results=[],
            prompts=prompts,
            researcher_calls=researcher_calls,
        )
        events = _collect()
        # planner: status+reasoning, synthesizer: status, then answering+content
        assert [e.type for e in events] == [
            EVENT_STATUS, EVENT_REASONING, EVENT_STATUS, EVENT_STATUS, EVENT_CONTENT,
        ]
        assert _statuses(events) == [
            dr_mod.STATUS_PLANNING,
            dr_mod.STATUS_SYNTHESIZING,
            EVENT_ANSWERING,
        ]
        assert _answer(events) == "# FINAL ANSWER"
        assert researcher_calls == []  # researcher never ran
        # Decision line stripped from the reasoning event.
        assert events[1].content == "Findings look complete."
        # Planner prompt then synthesizer prompt (both through _invoke_chat).
        assert len(prompts) == 2
        assert "<findings_so_far>" in prompts[0]
        assert "(no findings yet)" in prompts[0]
        assert "<research_findings>" in prompts[1]
        # Model client closed exactly once.
        assert len(rec["closed"]) == 1

    def test_history_rendered_into_planner_prompt(self, monkeypatch):
        prompts: List[str] = []
        _patch_flow(
            monkeypatch,
            chat_responses=["go\nDECISION: SYNTHESIZE", "FINAL"],
            prompts=prompts,
        )
        _collect(history=[{"role": "user", "content": "earlier question"}])
        assert "user: earlier question" in prompts[0]

    def test_one_research_round_streams_tool_events_and_findings(
        self, monkeypatch
    ):
        prompts: List[str] = []
        researcher_calls: List[Dict[str, Any]] = []
        _patch_flow(
            monkeypatch,
            chat_responses=[
                "Plan: check users table\nDECISION: CONTINUE",
                "Enough data\nDECISION: SYNTHESIZE",
                "# FINAL",
            ],
            researcher_results=[
                (
                    "FOUND-1 users table has 3 rows",
                    [
                        (
                            EVENT_TOOL_CALL,
                            json.dumps(
                                {"name": "search_knowledge", "args": {"query": "users"}}
                            ),
                        ),
                        (
                            EVENT_TOOL_RESULT,
                            json.dumps(
                                {"name": "search_knowledge", "content": "row1"}
                            ),
                        ),
                    ],
                )
            ],
            prompts=prompts,
            researcher_calls=researcher_calls,
        )
        events = _collect()
        assert [e.type for e in events] == [
            EVENT_STATUS,        # planning (1)
            EVENT_REASONING,     # plan 1
            EVENT_STATUS,        # researching
            EVENT_TOOL_CALL,     # researcher tool events forwarded
            EVENT_TOOL_RESULT,
            EVENT_STATUS,        # planning (2)
            EVENT_REASONING,     # plan 2
            EVENT_STATUS,        # synthesizing
            EVENT_STATUS,        # answering
            EVENT_CONTENT,       # final answer
        ]
        assert _answer(events) == "# FINAL"
        # Tool frames are JSON payloads with the expected shape.
        call_event = _tool_event(events, EVENT_TOOL_CALL)
        assert call_event is not None
        assert json.loads(call_event.content) == {
            "name": "search_knowledge",
            "args": {"query": "users"},
        }
        result_event = _tool_event(events, EVENT_TOOL_RESULT)
        assert result_event is not None
        assert json.loads(result_event.content)["name"] == "search_knowledge"
        # Researcher got the gathered tools, the plan as its task and a
        # system prompt carrying the plan block.
        assert len(researcher_calls) == 1
        assert researcher_calls[0]["tools"] == ["TOOL_A"]
        assert researcher_calls[0]["task"] == "Plan: check users table"
        assert "<research_plan>" in researcher_calls[0]["system_prompt"]
        # Findings flow into the second planner prompt and the synthesizer.
        assert "[1] FOUND-1 users table has 3 rows" in prompts[1]
        assert "[1] FOUND-1 users table has 3 rows" in prompts[2]

    def test_max_iterations_hard_stop(self, monkeypatch):
        prompts: List[str] = []
        researcher_calls: List[Dict[str, Any]] = []
        _patch_flow(
            monkeypatch,
            chat_responses=[
                f"Plan round {i}\nDECISION: CONTINUE" for i in range(1, 6)
            ] + ["# FINAL"],
            researcher_results=[(f"FOUND-{i}", []) for i in range(1, 5)],
            prompts=prompts,
            researcher_calls=researcher_calls,
        )
        events = _collect()
        statuses = _statuses(events)
        assert statuses.count(dr_mod.STATUS_PLANNING) == dr_mod.MAX_ITERATIONS
        assert statuses.count(dr_mod.STATUS_RESEARCHING) == dr_mod.MAX_ITERATIONS - 1
        assert statuses.count(dr_mod.STATUS_SYNTHESIZING) == 1
        assert statuses[-1] == EVENT_ANSWERING
        assert _answer(events) == "# FINAL"
        # 5 planner prompts + 1 synthesizer prompt; 4 researcher rounds.
        assert len(prompts) == dr_mod.MAX_ITERATIONS + 1
        assert len(researcher_calls) == 4
        # Iteration budget visible to the researcher system prompts 1..4.
        # The prompt body is language-dependent (admin generation.language),
        # so match the rendered numbers, not the English wording.
        for i, call in enumerate(researcher_calls, start=1):
            assert re.search(
                rf"{i}\s+(?:of|из)\s+{dr_mod.MAX_ITERATIONS}",
                call["system_prompt"],
            )
        # Last planner prompt saw all four accumulated findings.
        assert "[4] FOUND-4" in prompts[dr_mod.MAX_ITERATIONS - 1]

    def test_planner_error_yields_error_event_and_closes_client(
        self, monkeypatch
    ):
        rec = _patch_flow(
            monkeypatch,
            chat_responses=[RuntimeError("boom")],
            researcher_results=[],
        )
        events = _collect()
        assert [e.type for e in events] == [EVENT_ERROR]
        assert events[0].content == "Deep research failed: boom"
        assert len(rec["closed"]) == 1

    def test_empty_final_uses_fallback_answer(self, monkeypatch):
        _patch_flow(
            monkeypatch,
            chat_responses=["enough\nDECISION: SYNTHESIZE", "   "],
            researcher_results=[],
        )
        events = _collect()
        assert _statuses(events) == [
            dr_mod.STATUS_PLANNING,
            dr_mod.STATUS_SYNTHESIZING,
            EVENT_ANSWERING,
        ]
        assert "produced no final answer" in _answer(events)

    def test_straight_line_fallback_without_graph(self, monkeypatch):
        researcher_calls: List[Dict[str, Any]] = []
        _patch_flow(
            monkeypatch,
            chat_responses=[
                "Findings look complete.\nDECISION: SYNTHESIZE",
                "# STRAIGHT FINAL",
            ],
            researcher_results=[],
            researcher_calls=researcher_calls,
        )
        monkeypatch.setattr(dr_mod, "_build_research_graph", lambda: None)
        events = _collect()
        assert _statuses(events) == [
            dr_mod.STATUS_PLANNING,
            dr_mod.STATUS_SYNTHESIZING,
            EVENT_ANSWERING,
        ]
        assert _answer(events) == "# STRAIGHT FINAL"
        assert researcher_calls == []

    def test_missing_decision_defaults_to_continue(self, monkeypatch):
        prompts: List[str] = []
        _patch_flow(
            monkeypatch,
            chat_responses=[
                "no explicit decision",
                "done\nDECISION: SYNTHESIZE",
                "FINAL",
            ],
            researcher_results=[("FOUND-1", [])],
            prompts=prompts,
        )
        events = _collect()
        assert _statuses(events).count(dr_mod.STATUS_RESEARCHING) == 1
        assert _answer(events) == "FINAL"
        assert "[1] FOUND-1" in prompts[1]


# --------------------------------------------------------------------------- #
# Router: POST /api/products/{id}/ask with deep_research
# --------------------------------------------------------------------------- #
def _seed_product(db_mod, pid: str = "prod_1"):
    with db_mod.SessionLocal() as s:
        s.add(ProductORM(id=pid, name="Widget"))
        s.commit()


def _make_client(db_mod, user):
    app, client = build_test_client(db_mod, [expert_router_module])
    app.dependency_overrides[expert_router_module.get_current_user] = lambda: user
    return app, client


class TestAskRouterDeepResearch:
    def test_flag_routes_to_deep_research_runner(
        self, isolated_db, admin_user, monkeypatch
    ):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db, admin_user)

        calls: List[Dict[str, Any]] = []

        async def fake_deep_runner(
            product_id,
            query,
            *,
            session_id=None,
            history=None,
            model=None,
            seed_history=False,
            **kwargs,
        ):
            calls.append(
                {
                    "product_id": product_id,
                    "query": query,
                    "session_id": session_id,
                    "history": history,
                    "model": model,
                    "seed_history": seed_history,
                }
            )
            yield ExpertStreamEvent(EVENT_STATUS, "planning")
            yield ExpertStreamEvent(EVENT_CONTENT, "ANSWER")

        async def regular_runner_must_not_run(*args, **kwargs):
            raise AssertionError("regular runner used for a deep-research turn")
            yield  # pragma: no cover - make it an async generator

        monkeypatch.setattr(
            expert_router_module, "run_deep_research_stream", fake_deep_runner
        )
        monkeypatch.setattr(
            expert_router_module, "run_agent_chat_stream", regular_runner_must_not_run
        )

        r = client.post(
            "/api/products/prod_1/ask",
            json={
                "query": QUERY,
                "messages": [{"role": "user", "content": "prior turn"}],
                "model": "m1",
                "deep_research": True,
            },
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        # New session announced (first frame + header) since product exists.
        assert r.headers.get("x-session-id")
        assert '"session_id"' in r.text
        # The deep-research statuses/frames pass through the SSE contract.
        assert '"planning"' in r.text
        assert '"ANSWER"' in r.text
        assert r.text.endswith("data: [DONE]\n\n")

        # Runner received the parsed request (positional id/query + kwargs).
        assert len(calls) == 1
        assert calls[0]["product_id"] == "prod_1"
        assert calls[0]["query"] == QUERY
        assert calls[0]["session_id"] == r.headers["x-session-id"]
        assert calls[0]["history"] == [{"role": "user", "content": "prior turn"}]
        assert calls[0]["model"] == "m1"
        assert calls[0]["seed_history"] is True

        # Transcript persisted from the streamed events.
        msgs = client.get(
            f"/api/products/prod_1/chat/sessions/{calls[0]['session_id']}/messages"
        )
        assert msgs.status_code == 200
        roles = [m["role"] for m in msgs.json()]
        assert roles == ["user", "assistant"]
        assert msgs.json()[1]["content"] == "ANSWER"

    def test_without_flag_uses_regular_runner(
        self, isolated_db, admin_user, monkeypatch
    ):
        _seed_product(isolated_db)
        app, client = _make_client(isolated_db, admin_user)

        async def deep_runner_must_not_run(*args, **kwargs):
            raise AssertionError("deep runner used without the flag")
            yield  # pragma: no cover - make it an async generator

        async def fake_regular_runner(
            product_id,
            query,
            *,
            session_id=None,
            history=None,
            model=None,
            seed_history=False,
            **kwargs,
        ):
            yield ExpertStreamEvent(EVENT_STATUS, "retrieving")
            yield ExpertStreamEvent(EVENT_CONTENT, "PLAIN")

        monkeypatch.setattr(
            expert_router_module, "run_deep_research_stream", deep_runner_must_not_run
        )
        monkeypatch.setattr(
            expert_router_module, "run_agent_chat_stream", fake_regular_runner
        )

        r = client.post("/api/products/prod_1/ask", json={"query": QUERY})
        assert r.status_code == 200
        assert '"PLAIN"' in r.text
        assert '"planning"' not in r.text
        assert r.text.endswith("data: [DONE]\n\n")

    def test_unknown_product_streams_statelessly(
        self, isolated_db, admin_user, monkeypatch
    ):
        app, client = _make_client(isolated_db, admin_user)
        calls: List[Dict[str, Any]] = []

        async def fake_deep_runner(
            product_id,
            query,
            *,
            session_id=None,
            history=None,
            model=None,
            seed_history=False,
            **kwargs,
        ):
            calls.append({"session_id": session_id, "seed_history": seed_history})
            yield ExpertStreamEvent(EVENT_CONTENT, "EPHEMERAL")

        monkeypatch.setattr(
            expert_router_module, "run_deep_research_stream", fake_deep_runner
        )
        r = client.post(
            "/api/products/ghost/ask", json={"query": QUERY, "deep_research": True}
        )
        assert r.status_code == 200
        # No session row → no session_id frame, no header, no persistence.
        assert '"session_id"' not in r.text
        assert not r.headers.get("x-session-id")
        assert '"EPHEMERAL"' in r.text
        assert r.text.endswith("data: [DONE]\n\n")
        assert calls[0]["session_id"] is None
        assert calls[0]["seed_history"] is False
