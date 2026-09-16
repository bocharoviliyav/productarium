"""Deep Research flow (Wave E): planner → researcher → synthesizer.

A LangGraph ``StateGraph`` answers a query with up to
``MAX_ITERATIONS`` (5) plan/research rounds, then synthesizes the final
answer from everything the researcher gathered:

- **planner** — plain LLM call (``deep_research_planner.md``): given the
  query, the conversation history and the findings so far, it produces the
  next research plan and ends its output with a machine-readable
  ``DECISION: CONTINUE`` or ``DECISION: SYNTHESIZE`` line that drives the
  conditional edge (continue researching vs. move to synthesis). The
  iteration budget is enforced here as a hard stop regardless of the
  decision.
- **researcher** — a ``create_react_agent`` sub-agent whose tools are the
  product-scoped knowledge tools (``api.agents.tools.build_expert_tools``:
  knowledge recall + artifact readers) plus the product's bound enabled MCP
  tools (best-effort). Tool calls/results are surfaced as
  ``tool_call`` / ``tool_result`` events (same frames as the regular expert
  agent) and the agent's final text becomes a "finding".
- **synthesizer** — plain LLM call (``deep_research_synthesizer.md``) over
  the query + all findings; its markdown answer is delivered as ``content``
  frames.

Stream contract (additive to the Wave-B SSE contract; the UI knows the
``planning`` / ``researching`` / ``synthesizing`` statuses and ignores
unknown frames):

    data: {"status": "planning"|"researching"|"synthesizing"|"answering"}
    data: {"reasoning": "<plan / iteration trace>"}
    data: {"tool_call": {...}} / {"tool_result": {...}}
    data: {"content": "<answer chunk>"}
    data: {"error": "<message>"}

If ``langgraph`` is unavailable the same node functions run straight-line,
so the module has no hard langgraph dependency (same pattern as the Wave-D
docgen flows). The one chat model (and its httpx client) is closed in the
``finally`` of the public generator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from api.agents.expert import (
    _build_expert_chat_model,
    _close_model_client,
    _summarize_tool_content,
)
from api.expert.prompt import _chunk_text
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
from api.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

#: Hard cap on plan/research rounds (planner DECISION cannot exceed it).
MAX_ITERATIONS = 5
#: Findings kept for the planner/synthesizer prompts (most recent win).
MAX_FINDINGS = 10
#: Char cap for one finding (protects the planner/synthesizer context).
MAX_FINDING_CHARS = 8_000
#: Char cap for the rendered history block injected into prompts.
MAX_HISTORY_CHARS = 4_000
#: Overall wall-clock budget for ONE Deep Research turn (DoS guard, review
#: #4: iteration caps bound the round count, not the wall time; a slow LLM
#: or MCP tool could occupy the request indefinitely). Env-overridable;
#: default 15 minutes. When exceeded the planner force-synthesizes and a
#: running researcher iteration is aborted with its findings kept.
def _env_float(name: str, default: float) -> float:
    """Import-time env parsing that never crashes on garbage values."""
    raw = (os.environ.get(name) or "").strip()
    try:
        return float(raw)
    except ValueError:
        if raw:
            logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


DEEP_RESEARCH_TIMEOUT_SECONDS = _env_float("DEEP_RESEARCH_TIMEOUT_SECONDS", 900.0)

#: Additive Deep Research status values (see the module docstring).
STATUS_PLANNING = "planning"
STATUS_RESEARCHING = "researching"
STATUS_SYNTHESIZING = "synthesizing"


# --------------------------------------------------------------------------- #
# Prompt templates (EN bodies in refs/prompts/*.md; str.replace, never .format)
# --------------------------------------------------------------------------- #
_PLANNER_FALLBACK = (
    "You are the planning step of a deep research assistant for the product "
    "`{product_name}`. The user query, the conversation so far and the "
    "findings gathered by previous research iterations are below.\n\n"
    "<query>\n{query}\n</query>\n\n"
    "<conversation_history>\n{history}\n</conversation_history>\n\n"
    "<findings_so_far>\n{findings}\n</findings_so_far>\n\n"
    "Decide how to proceed:\n"
    "1. If key aspects of the query are still unresearched, write a SHORT "
    "research plan (bullet list of concrete questions/instructions for the "
    "researcher) and end with the line: DECISION: CONTINUE\n"
    "2. If the findings are sufficient to answer, briefly state what will be "
    "synthesized and end with the line: DECISION: SYNTHESIZE\n"
    "Write {language_name}. The DECISION line must be the LAST line."
)

_RESEARCHER_FALLBACK = (
    "You are the research step of a deep research assistant for the product "
    "`{product_name}`. Follow the research plan below using your tools "
    "(knowledge recall over the product's indexed artifacts and any bound "
    "MCP tools). Call the tools you need; do not guess answers you can look "
    "up. Finish with a concise factual summary of what you found, with "
    "citations (file paths / artifact names / table names) where possible.\n\n"
    "<research_plan>\n{plan}\n</research_plan>\n\n"
    "This is research iteration {iteration} of {max_iterations}. "
    "Write {language_name}."
)

_SYNTHESIZER_FALLBACK = (
    "You are the synthesis step of a deep research assistant for the product "
    "`{product_name}`. Using EXCLUSIVELY the research findings below (never "
    "invent facts), write the final answer to the user's query as "
    "well-structured Markdown: lead with the direct answer, then supporting "
    "detail; cite sources (file paths / artifact names / table names) inline; "
    "state clearly when the findings do not cover something.\n\n"
    "<query>\n{query}\n</query>\n\n"
    "<conversation_history>\n{history}\n</conversation_history>\n\n"
    "<research_findings>\n{findings}\n</research_findings>\n\n"
    "Write {language_name}."
)

_LANGUAGE_INSTRUCTION = (
    "the same language as the user's query (keep code identifiers, file "
    "paths, and API names in English)"
)

_DECISION_RE = re.compile(
    r"DECISION:\s*(CONTINUE|SYNTHESIZE)\s*$", re.IGNORECASE
)


def _load_template(filename: str, fallback: str) -> str:
    from api.prompts import load_prompt_file

    return load_prompt_file(filename, fallback) or fallback


def _render(template: str, values: Dict[str, str]) -> str:
    """str.replace substitution (never .format — bodies carry literal braces)."""
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def _history_block(history: Optional[List[Dict[str, Any]]]) -> str:
    """Render ``[{role, content}]`` history as a small prompt block."""
    lines: List[str] = []
    for m in (history or [])[-10:]:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            lines.append(f"{role}: {content}")
    block = "\n".join(lines)
    if len(block) > MAX_HISTORY_CHARS:
        block = "...(truncated)\n" + block[-MAX_HISTORY_CHARS:]
    return block or "(none)"


def _findings_block(findings: List[str]) -> str:
    """Render the accumulated findings as a numbered prompt block."""
    if not findings:
        return "(no findings yet)"
    parts: List[str] = []
    for i, finding in enumerate(findings[-MAX_FINDINGS:], start=1):
        parts.append(f"[{i}] {finding}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Test seams (monkeypatch these; no real LLM/agent is built in unit tests)
# --------------------------------------------------------------------------- #
async def _gather_research_tools(
    product_id: str,
    session_factory: Optional[Any] = None,
) -> List[Any]:
    """Product knowledge tools + best-effort MCP tools for the researcher.

    Built-in tool names win on a name conflict (the same rule as
    ``build_expert_agent``, review #4: an MCP tool named like a built-in
    would shadow/confuse the react agent otherwise).
    """
    from api.agents.tools import build_expert_tools

    tools = list(build_expert_tools(product_id, session_factory=session_factory))
    taken_names = {getattr(t, "name", "") for t in tools}
    try:
        from api.agents.expert import _gather_http_integration_tools, _gather_mcp_tools

        extra = await _gather_mcp_tools(product_id, session_factory)
        extra = (extra or []) + _gather_http_integration_tools(product_id, session_factory)
        for tool in extra:
            name = getattr(tool, "name", "")
            if name and name in taken_names:
                logger.warning(
                    "deep research: external tool %r shadows a built-in tool; dropping",
                    name,
                )
                continue
            tools.append(tool)
    except Exception as e:  # pragma: no cover - best-effort by contract
        logger.debug("deep research: external tools unavailable: %s", e)
    return tools


async def _invoke_chat(chat: Any, prompt: str) -> str:
    """One non-streaming LLM call → cleaned text (single seam to patch)."""
    result = await chat.ainvoke(prompt)
    content = getattr(result, "content", result)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        content = "".join(parts)
    return str(content or "")


async def _run_research_agent(
    chat: Any,
    tools: List[Any],
    system_prompt: str,
    task: str,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Run the researcher react-agent; return (summary, tool_events).

    Tool events are ``(event_type, json_payload)`` tuples using the same
    shapes the SSE router emits for the regular expert agent.
    """
    from langchain_core.messages import HumanMessage

    from langgraph.prebuilt import create_react_agent

    from api.config.timeout import resolve_expert_recursion_limit

    agent = create_react_agent(model=chat, tools=tools, prompt=system_prompt)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=task)]},
        config={"recursion_limit": resolve_expert_recursion_limit()},
    )
    messages = (
        getattr(result, "messages", None)
        or (result.get("messages") if isinstance(result, dict) else None)
        or []
    )
    events: List[Tuple[str, str]] = []
    tool_names: Dict[str, str] = {}
    summary = ""
    for message in messages:
        mtype = getattr(message, "type", "")
        if mtype == "ai":
            calls = getattr(message, "tool_calls", None) or []
            for call in calls:
                name = str(call.get("name") or "tool")
                tool_names[str(call.get("id") or "")] = name
                try:
                    args = call.get("args") or {}
                except Exception:  # pragma: no cover - defensive
                    args = {}
                if not isinstance(args, dict):
                    args = {"input": str(args)}
                events.append(
                    (
                        EVENT_TOOL_CALL,
                        json.dumps({"name": name, "args": args}, ensure_ascii=False),
                    )
                )
            text = getattr(message, "content", "")
            if isinstance(text, list):
                text = "".join(
                    b if isinstance(b, str) else b.get("text", "")
                    for b in text
                    if isinstance(b, (str, dict))
                )
            if isinstance(text, str) and text.strip():
                summary = text
        elif mtype == "tool":
            name = (
                getattr(message, "name", None)
                or tool_names.get(getattr(message, "tool_call_id", "") or "")
                or "tool"
            )
            events.append(
                (
                    EVENT_TOOL_RESULT,
                    json.dumps(
                        {
                            "name": str(name),
                            "content": _summarize_tool_content(
                                getattr(message, "content", "")
                            ),
                        },
                        ensure_ascii=False,
                    ),
                )
            )
    return summary, events


# --------------------------------------------------------------------------- #
# Graph state + nodes
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import guard (same pattern as the docgen flows)
    from langgraph.graph import END, START, StateGraph
    from typing import TypedDict

    class _ResearchState(TypedDict, total=False):
        query: str
        product_name: str
        history_block: str
        language: str
        chat: Any
        tools: List[Any]
        events: List[Tuple[str, str]]
        findings: List[str]
        iteration: int
        plan: str
        decision: str
        final: str
        error: str
        deadline: float

    _LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - straight-line fallback below
    _LANGGRAPH_AVAILABLE = False

    class _ResearchState(dict):  # type: ignore[no-redef]
        """Plain dict state for the straight-line fallback."""

        pass


async def _planner_node(state: _ResearchState) -> Dict[str, Any]:
    """Plan the next research round and decide CONTINUE vs SYNTHESIZE."""
    iteration = int(state.get("iteration") or 0) + 1
    findings = list(state.get("findings") or [])
    deadline = state.get("deadline")
    if deadline is not None and time.monotonic() >= float(deadline):
        # Time budget exhausted (review #4): skip the LLM call entirely and
        # force synthesis of whatever findings exist so the turn still ends.
        plan = "(research time budget exhausted — synthesizing the findings collected so far)"
        return {
            "iteration": iteration,
            "plan": plan,
            "decision": "synthesize",
            "events": [
                (EVENT_STATUS, STATUS_PLANNING),
                (EVENT_REASONING, plan),
            ],
        }
    prompt = _render(
        _load_template("deep_research_planner.md", _PLANNER_FALLBACK),
        {
            "product_name": state.get("product_name") or "this product",
            "query": state.get("query") or "",
            "history": state.get("history_block") or "(none)",
            "findings": _findings_block(findings),
            "language_name": state.get("language") or _LANGUAGE_INSTRUCTION,
        },
    )
    raw = await _invoke_chat(state["chat"], prompt)
    match = _DECISION_RE.search((raw or "").strip())
    decision = "continue"
    if match:
        decision = match.group(1).lower()
        raw = (raw or "")[: match.start()].strip()
    if iteration >= MAX_ITERATIONS:
        decision = "synthesize"
    plan = (raw or "(no plan)").strip()
    events: List[Tuple[str, str]] = [
        (EVENT_STATUS, STATUS_PLANNING),
        (EVENT_REASONING, plan[:4000]),
    ]
    return {
        "iteration": iteration,
        "plan": plan,
        "decision": decision,
        "events": events,
    }


async def _researcher_node(state: _ResearchState) -> Dict[str, Any]:
    """Run the tool-using researcher agent for the current plan."""
    iteration = int(state.get("iteration") or 1)
    system_prompt = _render(
        _load_template("deep_research_researcher.md", _RESEARCHER_FALLBACK),
        {
            "product_name": state.get("product_name") or "this product",
            "plan": state.get("plan") or "(no plan)",
            "iteration": str(iteration),
            "max_iterations": str(MAX_ITERATIONS),
            "language_name": state.get("language") or _LANGUAGE_INSTRUCTION,
        },
    )
    deadline = state.get("deadline")
    task = _run_research_agent(
        state["chat"], state.get("tools") or [], system_prompt, state.get("plan") or ""
    )
    try:
        if deadline is not None:
            remaining = max(0.0, float(deadline) - time.monotonic())
            summary, tool_events = await asyncio.wait_for(task, timeout=remaining)
        else:  # pragma: no cover - deadline is always set by the public runner
            summary, tool_events = await task
    except asyncio.TimeoutError:
        # Budget exhausted mid-research (review #4): keep what was gathered,
        # record an explicit finding, and let the graph move on (the next
        # planner pass force-synthesizes).
        logger.warning(
            "deep research: researcher iteration %d exceeded the time budget",
            iteration,
        )
        summary = (
            f"(research iteration {iteration} exceeded the remaining time "
            "budget; continuing with the findings collected so far)"
        )
        tool_events = []
    events: List[Tuple[str, str]] = [(EVENT_STATUS, STATUS_RESEARCHING)] + tool_events
    finding = (summary or "(researcher produced no summary)").strip()[:MAX_FINDING_CHARS]
    findings = list(state.get("findings") or []) + [finding]
    return {"findings": findings[-MAX_FINDINGS:], "events": events}


async def _synthesizer_node(state: _ResearchState) -> Dict[str, Any]:
    """Synthesize the final markdown answer from the findings."""
    prompt = _render(
        _load_template("deep_research_synthesizer.md", _SYNTHESIZER_FALLBACK),
        {
            "product_name": state.get("product_name") or "this product",
            "query": state.get("query") or "",
            "history": state.get("history_block") or "(none)",
            "findings": _findings_block(list(state.get("findings") or [])),
            "language_name": state.get("language") or _LANGUAGE_INSTRUCTION,
        },
    )
    raw = await _invoke_chat(state["chat"], prompt)
    final = (raw or "").strip()
    return {
        "final": final,
        "events": [(EVENT_STATUS, STATUS_SYNTHESIZING)],
    }


def _route_after_planner(state: _ResearchState) -> str:
    """Conditional edge: keep researching or move to synthesis."""
    if (state.get("decision") or "continue").lower() == "synthesize":
        return "synthesize"
    return "research"


def _build_research_graph() -> Optional[Any]:
    """Compile the planner→researcher→synthesizer StateGraph (or None)."""
    if not _LANGGRAPH_AVAILABLE:  # pragma: no cover - env-dependent
        return None
    try:
        graph = StateGraph(_ResearchState)
        graph.add_node("plan", _planner_node)
        graph.add_node("research", _researcher_node)
        graph.add_node("synthesize", _synthesizer_node)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            _route_after_planner,
            {"research": "research", "synthesize": "synthesize"},
        )
        graph.add_edge("research", "plan")
        graph.add_edge("synthesize", END)
        return graph.compile()
    except Exception as e:  # pragma: no cover - defensive over graph build
        logger.warning("deep research: graph build failed (%s); straight-line", e)
        return None


# --------------------------------------------------------------------------- #
# Straight-line fallback (no langgraph)
# --------------------------------------------------------------------------- #
async def _run_straight_line(state: _ResearchState) -> Dict[str, Any]:
    """Run the same node functions sequentially (langgraph unavailable).

    Node event batches are collected under ``collected_events`` so the public
    generator can yield the same status/reasoning/tool frames the graph path
    streams (parity between both execution modes).
    """
    current: Dict[str, Any] = dict(state)
    collected: List[Tuple[str, str]] = []
    while True:
        update = await _planner_node(current)
        current.update(update)
        collected.extend(update.get("events") or [])
        if (current.get("decision") or "continue").lower() == "synthesize":
            break
        update = await _researcher_node(current)
        current.update(update)
        collected.extend(update.get("events") or [])
    update = await _synthesizer_node(current)
    current.update(update)
    collected.extend(update.get("events") or [])
    current["collected_events"] = collected
    return current


# --------------------------------------------------------------------------- #
# Public API (signature-compatible with run_agent_chat_stream)
# --------------------------------------------------------------------------- #
async def run_deep_research_stream(
    product_id: str,
    query: str,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    checkpointer: Optional[Any] = None,
    session_factory: Optional[Any] = None,
    seed_history: bool = False,
    chat: Optional[Any] = None,
) -> AsyncIterator[ExpertStreamEvent]:
    """Stream a Deep Research turn as typed events (≤ MAX_ITERATIONS rounds).

    Same generator contract as ``api.agents.expert.run_agent_chat_stream`` so
    the SSE router can substitute either runner; ``session_id`` /
    ``checkpointer`` are accepted for signature compatibility but unused
    (deep research turns are self-contained; the transcript is persisted by
    the router from the streamed events).
    """
    from api.agents.expert import _product_name_by_id

    chat = chat or _build_expert_chat_model(model)
    try:
        tools = await _gather_research_tools(product_id, session_factory)
        initial: Dict[str, Any] = {
            "query": query,
            "product_name": _product_name_by_id(product_id),
            "history_block": _history_block(history),
            "language": _LANGUAGE_INSTRUCTION,
            "chat": chat,
            "tools": tools,
            "findings": [],
            "iteration": 0,
            "deadline": time.monotonic() + DEEP_RESEARCH_TIMEOUT_SECONDS,
        }

        final_text = ""
        try:
            graph = _build_research_graph()
            if graph is not None:
                # updates mode: each yielded item is {node_name: node_update};
                # node updates carry the "events" batch and (synthesize) "final".
                async for update in graph.astream(
                    initial,
                    config={"recursion_limit": 2 * MAX_ITERATIONS + 6},
                    stream_mode="updates",
                ):
                    if not isinstance(update, dict):
                        continue
                    for node_state in update.values():
                        if not isinstance(node_state, dict):
                            continue
                        for event in node_state.get("events") or []:
                            yield ExpertStreamEvent(event[0], event[1])
                        if node_state.get("final"):
                            final_text = str(node_state["final"])
            else:
                final_state = await _run_straight_line(initial)
                for event in final_state.get("collected_events") or []:
                    yield ExpertStreamEvent(event[0], event[1])
                final_text = final_state.get("final") or ""
        except Exception as e:  # pragma: no cover - depends on live LLM/graph
            logger.error("deep research failed: %s", e, exc_info=True)
            yield ExpertStreamEvent(EVENT_ERROR, f"Deep research failed: {e}")
            return

        if not final_text:
            final_text = (
                "_(Deep research produced no final answer — the model may be "
                "unavailable or the findings empty.)_"
            )
        yield ExpertStreamEvent(EVENT_STATUS, EVENT_ANSWERING)
        for chunk in _chunk_text(final_text):
            yield ExpertStreamEvent(EVENT_CONTENT, chunk)
    finally:
        await _close_model_client(chat)


__all__ = [
    "MAX_ITERATIONS",
    "STATUS_PLANNING",
    "STATUS_RESEARCHING",
    "STATUS_SYNTHESIZING",
    "run_deep_research_stream",
]
