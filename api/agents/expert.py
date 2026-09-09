"""Expert agent on LangChain/LangGraph (Wave B).

Builds a product-scoped reasoning agent
(``langgraph.prebuilt.create_react_agent``) with:

- the model from :func:`api.llm.client.build_chat_model` (the single LLM
  factory: corporate TLS, timeouts, no-auth placeholder policy),
- product-scoped tools from :func:`api.agents.tools.build_expert_tools`
  (``product_id`` bound in a closure — never an LLM-controlled argument),
- MCP tools from the product's enabled bindings (:func:`api.mcp.manager.gather_mcp_agent_tools`,
  best-effort — appended after the built-in tools),
- a persistent checkpointer (:mod:`api.agents.runtime`) keyed by the chat
  session id, so conversation state survives across requests/processes.

Public API:
- :func:`run_agent_chat_stream` — async generator of ``ExpertStreamEvent``
  objects mapped from ``astream_events`` (status / reasoning / content /
  tool_call / tool_result), ready for the SSE router.
- :func:`run_agent_doc` — one-shot expert document generation over the agent.

The SSE contract emitted downstream (see ``api/routers/expert.py``):

    data: {"status": "retrieving"|"thinking"|"answering"}
    data: {"reasoning": "<model thoughts>"}
    data: {"content": "<answer chunk>"}
    data: {"tool_call": {"name": "<tool>", "args": {...}}}
    data: {"tool_result": {"name": "<tool>", "content": "<summary>"}}
    data: {"error": "<message>"}
    data: [DONE]
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from api.agents.tools import build_expert_tools
from api.expert.types import (
    EVENT_ANSWERING,
    EVENT_CONTENT,
    EVENT_REASONING,
    EVENT_RETRIEVING,
    EVENT_STATUS,
    EVENT_THINKING,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    ExpertStreamEvent,
)

logger = logging.getLogger(__name__)

#: How many messages of checkpointed history are replayed into the agent input.
MAX_HISTORY_MESSAGES = 20


def _resolve_expert_model_config(model: Optional[str]) -> Dict[str, Optional[str]]:
    """Resolve (model, base_url, api_key) for the expert task from admin config."""
    try:
        from api.config.abstraction import get_task_config

        cfg = get_task_config("expert") or {}
        resolved_model = model or cfg.get("model") or "qwen/qwen3.6-27b"
        return {
            "model": resolved_model,
            "base_url": cfg.get("base_url"),
            "api_key": cfg.get("api_key"),
        }
    except Exception as e:  # pragma: no cover - settings store is import-safe
        logger.debug("get_task_config(expert) failed; using defaults: %s", e)
        return {"model": model or "qwen/qwen3.6-27b", "base_url": None, "api_key": None}


def _build_expert_chat_model(model: Optional[str] = None) -> Any:
    """Build the expert task's ChatOpenAI via the central LLM factory."""
    from api.llm.client import build_chat_model

    cfg = _resolve_expert_model_config(model)
    return build_chat_model(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
    )


async def _close_model_client(chat: Any) -> None:
    """Close the per-request httpx client behind a chat model (no leaks).

    ``build_chat_model`` builds a fresh ``httpx.AsyncClient`` per call and the
    OpenAI SDK never closes a client it was handed, so the request owner
    must close it once the stream/invoke finishes.
    """
    client = getattr(chat, "http_async_client", None)
    if client is None:
        return
    try:
        await client.aclose()
    except Exception:  # pragma: no cover - defensive
        logger.debug("could not close the expert model httpx client", exc_info=True)


def build_expert_agent(
    product_id: str,
    model: Optional[str] = None,
    session_factory: Optional[Any] = None,
    checkpointer: Optional[Any] = None,
    chat: Optional[Any] = None,
    extra_tools: Optional[List[Any]] = None,
) -> Any:
    """Build the product-scoped expert agent (LangGraph create_react_agent).

    Args:
        product_id: Product every tool is scoped to (closure-bound).
        model: Optional model override; otherwise the admin ``models.expert``
            config resolves the model/base_url/api_key.
        session_factory: Optional SQLAlchemy session factory for the tools
            (tests); defaults to ``api.db.SessionLocal``.
        checkpointer: Optional LangGraph checkpointer (tests); defaults to the
            process-wide one from :func:`api.agents.runtime.get_checkpointer`.
        chat: Optional pre-built chat model. When given, no new model is
            built and the CALLER owns closing the model's httpx client
            (see ``_close_model_client``).
        extra_tools: Optional extra LangChain tools appended AFTER the
            built-in product tools (e.g. MCP tools from
            :func:`api.mcp.manager.gather_mcp_agent_tools`). Built-in tool
            names win on a name conflict.

    Returns:
        A compiled LangGraph react-agent graph. Invoke it with
        ``{"messages": [...]}`` and a ``config={"configurable": {"thread_id":
        session_id}}`` when a checkpointer is attached.
    """
    from langgraph.prebuilt import create_react_agent

    if chat is None:
        chat = _build_expert_chat_model(model)
    tools = build_expert_tools(product_id, session_factory=session_factory)
    if extra_tools:
        taken_names = {getattr(t, "name", "") for t in tools}
        for tool in extra_tools:
            name = getattr(tool, "name", "")
            if name and name in taken_names:
                logger.warning(
                    "expert agent: extra tool %r shadows a built-in tool; dropping",
                    name,
                )
                continue
            tools.append(tool)
    product_name = _product_name_by_id(product_id)

    from api.expert import prompt as _expert_prompt

    system_prompt = _build_agent_system_prompt(
        _expert_prompt.EXPERT_SYSTEM_PROMPT, product_name
    )
    return create_react_agent(
        model=chat,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
    )


async def _gather_mcp_tools(
    product_id: str,
    session_factory: Optional[Any] = None,
) -> List[Any]:
    """Best-effort MCP tools for the product (never raises, never hangs).

    Wraps :func:`api.mcp.manager.gather_mcp_agent_tools` — an MCP outage
    (DB down, dependency missing, dead server) can never break the agent.
    """
    try:
        from api.mcp.manager import gather_mcp_agent_tools

        return await gather_mcp_agent_tools(product_id, session_factory=session_factory)
    except Exception as e:
        logger.debug("mcp tools unavailable for product %s: %s", product_id, e)
        return []


def _gather_http_integration_tools(
    product_id: str,
    session_factory: Optional[Any] = None,
) -> List[Any]:
    """Best-effort HTTP-integration tools (never raises).

    Wraps :func:`api.integrations.http_tools.build_http_integration_tools` —
    every enabled integration registered in the admin panel becomes one
    agent tool by (sanitized) name (issue #3).
    """
    try:
        from api.integrations.http_tools import build_http_integration_tools

        return build_http_integration_tools(product_id, session_factory=session_factory)
    except Exception as e:
        logger.debug("http integration tools unavailable for product %s: %s", product_id, e)
        return []


def _product_name_by_id(product_id: str) -> str:
    """Look up a product name from the DB; fall back to the id. Non-fatal."""
    try:
        from api.db import SessionLocal
        from api.models import ProductORM

        with SessionLocal() as db:
            p = db.get(ProductORM, product_id)
            if p is not None and getattr(p, "name", None):
                return p.name
    except Exception as e:
        logger.debug("product name lookup failed for %r: %s", product_id, e)
    return product_id


def _build_agent_system_prompt(template: str, product_name: str) -> str:
    """Assemble the agent system prompt from the loaded EN template.

    Substitutes ``{product_name}`` / ``{language_name}`` (the language rule is
    generation-in-the-user's-language) and appends the verification guard
    (anti-hallucination + citation rules).
    """
    from api.utils.llm_helpers import safe_replace as _safe_replace

    system = _safe_replace(
        template,
        {
            "product_name": product_name or "this product",
            "language_name": (
                "the same language as the user's query (keep code "
                "identifiers, file paths, and API names in English)"
            ),
        },
    )
    try:
        from api.prompts import VERIFICATION_GUARD as _guard

        if _guard:
            system = system + "\n\n" + _guard
    except Exception:  # pragma: no cover - import-safe
        pass
    return system


# ---------------------------------------------------------------------------
# astream_events -> ExpertStreamEvent mapping
# ---------------------------------------------------------------------------


def _clean_tool_args(args: Any) -> Dict[str, Any]:
    """Coerce tool-call args into a small JSON-safe dict for the SSE frame."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return {"input": args}
    if isinstance(args, dict):
        out: Dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[str(k)] = v
            else:
                out[str(k)] = str(v)
        return out
    if isinstance(args, list):
        return {"items": [str(a) for a in args]}
    return {"input": str(args)}


def _summarize_tool_content(content: Any, limit: int = 2000) -> str:
    """Summarize a tool result for the ``tool_result`` SSE frame.

    The frame carries a SHORT human-readable summary (the full result already
    goes to the model); large payloads are truncated to keep frames light.
    """
    text = content if isinstance(content, str) else json.dumps(
        content, ensure_ascii=False, default=str
    )
    if len(text) > limit:
        text = text[:limit] + f"... (truncated, {len(text)} chars total)"
    return text


async def _agent_stream_events(
    agent: Any,
    messages: List[Any],
    config: Dict[str, Any],
) -> AsyncIterator[Any]:
    """Yield raw ``astream_events`` events from the agent graph.

    The graph state input is the ``{"messages": [...]}`` dict (passing the
    bare message list raises ``InvalidUpdateError`` in the graph's channel
    reducer). """
    try:
        async for event in agent.astream_events(
            {"messages": messages}, config=config, version="v2"
        ):
            yield event
    except TypeError:
        # Older/newer langgraph: fall back to astream with explicit modes.
        modes = ["values", "updates"]
        async for chunk in agent.astream(
            {"messages": messages}, config=config, stream_mode=modes
        ):
            yield chunk


async def run_agent_chat_stream(
    product_id: str,
    query: str,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    checkpointer: Optional[Any] = None,
    session_factory: Optional[Any] = None,
    seed_history: bool = False,
) -> AsyncIterator[ExpertStreamEvent]:
    """Stream an expert-agent chat turn as typed events.

    Args:
        product_id: The product to ground the answer in.
        query: The user's current question.
        session_id: Optional chat-session id; when given the agent continues
            that checkpointer thread. Without it the turn is stateless (no
            checkpointer) and ``history`` (if provided) is injected inline.
        history: Optional explicit ``[{role, content}]`` history. Ignored when
            a checkpointed session continues (the checkpointer owns the
            state) unless ``seed_history`` is set (first turn of a new
            session: the client's prior context seeds the thread).
        model: Optional model override.
        checkpointer: Optional checkpointer override (tests).
        session_factory: Optional SQLAlchemy session factory for tools (tests).
        seed_history: Inject ``history`` even when ``session_id`` is given
            (used for the first turn of a newly created session).

    Yields:
        ``ExpertStreamEvent`` objects: ``status`` (retrieving / thinking /
        answering), ``reasoning`` deltas, ``content`` deltas, ``tool_call``,
        and ``tool_result`` (payloads JSON-encoded in ``content``).
    """
    import secrets

    from langchain_core.messages import AIMessageChunk, HumanMessage

    if checkpointer is None and session_id:
        from api.agents.runtime import get_checkpointer

        checkpointer = await get_checkpointer()
    # Stateless (session-less) turns get a unique throwaway thread id and no
    # checkpointer, so parallel ad-hoc calls never share or pollute state.
    thread_id = session_id or f"ephemeral_{product_id}_{secrets.token_hex(4)}"
    config: Dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 25,
    }

    # One chat model (and one httpx client) per request; closed in the
    # finally below so repeated /ask calls never leak connections.
    chat = _build_expert_chat_model(model)
    try:
        # MCP tools of the product's enabled bindings + HTTP-integration
        # tools — best-effort; the kwarg is only passed when non-empty so
        # monkeypatched builders without it (tests) keep working.
        mcp_tools = await _gather_mcp_tools(product_id, session_factory)
        mcp_tools = mcp_tools + _gather_http_integration_tools(product_id, session_factory)
        agent = build_expert_agent(
            product_id,
            model=model,
            session_factory=session_factory,
            checkpointer=checkpointer,
            chat=chat,
            **({"extra_tools": mcp_tools} if mcp_tools else {}),
        )

        # Input: fresh user message; when continuing a session the checkpointer
        # already holds prior turns, so explicit history is only injected for
        # ephemeral (session-less) calls — or when seeding a brand-new session.
        inject_history = bool(history) and (not session_id or seed_history)
        if inject_history:
            input_messages: List[Any] = _history_messages(history) + [HumanMessage(content=query)]
        else:
            input_messages = [HumanMessage(content=query)]

        yield ExpertStreamEvent(EVENT_STATUS, EVENT_RETRIEVING)
        emitted_thinking = False
        emitted_answering = False
        seen_tool_calls: set = set()

        async for event in _agent_stream_events(agent, input_messages, config):
            kind = event.get("event") if isinstance(event, dict) else None
            if kind is None:
                # Raw astream chunk (updates/values mode fallback) — not used
                # for fine-grained frames; skip.
                continue
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if not isinstance(chunk, AIMessageChunk):
                    continue
                reasoning, content = _chunk_reasoning_content(chunk)
                if reasoning:
                    if not emitted_thinking:
                        emitted_thinking = True
                        yield ExpertStreamEvent(EVENT_STATUS, EVENT_THINKING)
                    yield ExpertStreamEvent(EVENT_REASONING, reasoning)
                if content:
                    if not emitted_answering:
                        emitted_answering = True
                        yield ExpertStreamEvent(EVENT_STATUS, EVENT_ANSWERING)
                    yield ExpertStreamEvent(EVENT_CONTENT, content)
            elif kind == "on_tool_start":
                name = event.get("name") or ""
                run_id = event.get("run_id") or ""
                raw_args = (event.get("data") or {}).get("input")
                if not name:
                    continue
                seen_tool_calls.add(run_id or name)
                payload = json.dumps(
                    {"name": name, "args": _clean_tool_args(raw_args)},
                    ensure_ascii=False,
                )
                yield ExpertStreamEvent(EVENT_TOOL_CALL, payload)
            elif kind == "on_tool_end":
                name = event.get("name") or ""
                content = (event.get("data") or {}).get("content")
                if not name:
                    continue
                payload = json.dumps(
                    {
                        "name": name,
                        "content": _summarize_tool_content(content),
                    },
                    ensure_ascii=False,
                )
                yield ExpertStreamEvent(EVENT_TOOL_RESULT, payload)
    finally:
        await _close_model_client(chat)


def _chunk_reasoning_content(chunk: Any) -> tuple:
    """Extract (reasoning, content) deltas from an AIMessageChunk.

    Reads the OpenAI-compatible reasoning fields from ``additional_kwargs``
    (``reasoning_content`` / ``reasoning`` / ``thinking`` — DeepSeek / vLLM /
    Qwen shapes); everything in ``content`` is an answer delta (inline
    ``<think>`` handling is not re-done here — the router-level parser in
    ``api.expert.llm`` covers the non-agent path).
    """
    content = getattr(chunk, "content", None)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        content = "".join(parts)
    if not isinstance(content, str):
        content = ""
    reasoning = ""
    additional = getattr(chunk, "additional_kwargs", None) or {}
    if isinstance(additional, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            val = additional.get(key)
            if isinstance(val, str) and val:
                reasoning = val
                break
    if not reasoning:
        meta = getattr(chunk, "response_metadata", None) or {}
        if isinstance(meta, dict):
            val = meta.get("reasoning_content") or meta.get("reasoning")
            if isinstance(val, str) and val:
                reasoning = val
    return reasoning, content


def _history_messages(history: List[Dict[str, Any]]) -> List[Any]:
    """Convert ``[{role, content}]`` dicts into langchain messages (bounded)."""
    from langchain_core.messages import AIMessage, HumanMessage

    out: List[Any] = []
    for m in (history or [])[-MAX_HISTORY_MESSAGES:]:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


async def run_agent_doc(
    product_id: str,
    query: str,
    model: Optional[str] = None,
    checkpointer: Optional[Any] = None,
    session_factory: Optional[Any] = None,
) -> str:
    """One-shot expert document over the agent (returns full Markdown).

    Runs the same product-scoped agent without a checkpointer thread (docs are
    standalone) and with the ``expert_agent_doc.md`` prompt variant. Falls
    back to the legacy non-agent generator on any agent failure so the
    ``/ask/doc`` endpoint keeps working without a live LLM server.
    """
    from langchain_core.messages import HumanMessage

    from api.expert import prompt as _expert_prompt
    from api.expert.knowledge import _product_name_by_id as _legacy_name

    try:
        from langgraph.prebuilt import create_react_agent

        chat = _build_expert_chat_model(model)
        try:
            tools = build_expert_tools(product_id, session_factory=session_factory)
            tools = tools + await _gather_mcp_tools(product_id, session_factory)
            tools = tools + _gather_http_integration_tools(product_id, session_factory)
            system_prompt = _build_agent_system_prompt(
                _expert_prompt.EXPERT_DOC_PROMPT, _legacy_name(product_id)
            )
            agent = create_react_agent(
                model=chat, tools=tools, prompt=system_prompt, checkpointer=checkpointer
            )
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config={"recursion_limit": 25},
            )
            text = _final_ai_text(result)
            if text.strip():
                from api.expert.prompt import _clean_llm_text

                return _clean_llm_text(text)
        finally:
            # Close the per-request httpx client even when the agent fails
            # (the except below then falls back to the legacy generator).
            await _close_model_client(chat)
    except Exception as e:  # pragma: no cover - depends on live LLM
        logger.warning(
            "expert agent doc generation failed (%s); using legacy path.", e
        )
    # Legacy fallback: the Wave-A non-agent generator.
    from api.expert.chat import run_expert_doc

    return await run_expert_doc(product_id, query, model)


def _final_ai_text(result: Any) -> str:
    """Extract the final AIMessage text from an agent invoke result."""
    messages = getattr(result, "messages", None) or (
        result.get("messages") if isinstance(result, dict) else None
    ) or []
    for message in reversed(list(messages)):
        if getattr(message, "type", "") == "ai":
            content = getattr(message, "content", "")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and isinstance(
                        block.get("text"), str
                    ):
                        parts.append(block["text"])
                content = "".join(parts)
            if isinstance(content, str) and content.strip():
                return content
    return ""


__all__ = [
    "MAX_HISTORY_MESSAGES",
    "build_expert_agent",
    "run_agent_chat_stream",
    "run_agent_doc",
]
