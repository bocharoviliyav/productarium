"""Unit tests for the live LangGraph agent stream path (Wave B regression).

``tests/integration/test_expert_chat_sessions.py`` monkeypatches
``run_agent_chat_stream`` at the router, so a broken graph invocation (e.g.
passing a bare message list instead of the ``{"messages": [...]}`` state
dict) is invisible to it — exactly the class of bug the live smoke run
caught (``InvalidUpdateError: Expected dict, got [HumanMessage(...)]``).

These tests exercise the REAL agent graph end-to-end with a fake chat model
(a minimal ``BaseChatModel`` subclass that supports token streaming) and no
checkpointer, asserting:

- the graph invocation succeeds (no ``InvalidUpdateError``) and streams
  ``on_chat_model_stream`` events (the frames the SSE mapper consumes);
- via the public ``run_agent_chat_stream``: ``status: retrieving`` first,
  then ``answering`` + ``content`` deltas for a plain scripted answer;
- ephemeral (session-less) turns inline the prior history into the graph
  input;
- the history helper is bounded and filtered;
- the per-request chat model's ``http_async_client`` is closed when the
  stream ends (fix [4]: one httpx client per /ask, never leaked).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage  # noqa: E402
from langchain_core.outputs import (  # noqa: E402
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)

from api.agents.expert import _agent_stream_events  # noqa: E402


class _FakeChatModel(BaseChatModel):
    """Minimal chat model returning scripted responses, with token streaming."""

    responses: List[str]
    _i: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        # The react agent binds tools; accept and ignore (scripted answers).
        return self

    def _next_response(self) -> str:
        idx = min(self._i, len(self.responses) - 1)
        self._i += 1
        return self.responses[idx]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=self._next_response()))
            ]
        )

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:  # noqa: ANN001
        # Yield the scripted response as two chunks so on_chat_model_stream
        # events fire (a non-streaming model emits no token events).
        text = self._next_response()
        mid = max(1, len(text) // 2)
        for part in (text[:mid], text[mid:]):
            yield ChatGenerationChunk(message=AIMessageChunk(content=part))


def _agent_for(responses: List[str]):
    """Build a real create_react_agent graph over the fake model."""
    from langgraph.prebuilt import create_react_agent

    model = _FakeChatModel(responses=responses)
    return create_react_agent(
        model=model,
        tools=[],
        prompt="You are a test expert agent.",
        checkpointer=None,
    )


async def _collect(agent, messages):
    events = []
    async for event in _agent_stream_events(
        agent, messages, {"configurable": {"thread_id": "t"}, "recursion_limit": 10}
    ):
        events.append(event)
    return events


class TestAgentStreamEvents:
    def test_graph_input_is_state_dict_not_bare_list(self):
        """The regression: a bare list input raised InvalidUpdateError."""
        agent = _agent_for(["hello there"])
        events = asyncio.run(_collect(agent, [HumanMessage(content="hi")]))
        assert events, "no events streamed"
        kinds = [e.get("event") for e in events if isinstance(e, dict)]
        assert "on_chat_model_stream" in kinds
        # A plain answer with no tool calls: no tool events.
        assert not [k for k in kinds if k and k.startswith("on_tool_")]

    def test_run_agent_chat_stream_status_and_content(self, monkeypatch):
        """The public stream yields retrieving -> answering + content."""
        from api.expert.types import EVENT_CONTENT, EVENT_STATUS

        import api.agents.expert as expert_mod

        # run_agent_chat_stream builds a REAL chat model before the stream;
        # pin the endpoint to a dead localhost port so construction is
        # deterministic (local -> placeholder key, no network at ctor time).
        monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://localhost:9/v1")
        monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)

        monkeypatch.setattr(
            expert_mod,
            "build_expert_agent",
            lambda product_id, model=None, session_factory=None, checkpointer=None, chat=None:
                _agent_for(["the answer text"]),
        )
        got: list = []

        async def _run():
            async for ev in expert_mod.run_agent_chat_stream("prod_1", "hi"):
                got.append(ev)

        asyncio.run(_run())

        statuses = [e.content for e in got if e.type == EVENT_STATUS]
        assert statuses[0] == "retrieving"
        assert "answering" in statuses
        content = "".join(e.content for e in got if e.type == EVENT_CONTENT)
        assert content == "the answer text"

    def test_history_injected_for_stateless_turn(self, monkeypatch):
        """Ephemeral (session-less) turns inline the prior history."""
        import api.agents.expert as expert_mod

        seen_inputs: List = []

        class _Spy:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def astream_events(self, messages, config=None, version=None):
                seen_inputs.append(messages)
                return self._inner.astream_events(
                    messages, config=config, version=version
                )

        monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://localhost:9/v1")
        monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            expert_mod,
            "build_expert_agent",
            lambda product_id, model=None, session_factory=None, checkpointer=None, chat=None:
                _Spy(_agent_for(["ok"])),
        )

        async def _run():
            async for _ in expert_mod.run_agent_chat_stream(
                "prod_1", "q", history=[{"role": "user", "content": "h1"}]
            ):
                pass

        asyncio.run(_run())

        assert seen_inputs, "no graph invocation captured"
        graph_input = seen_inputs[0]
        assert isinstance(graph_input, dict) and "messages" in graph_input
        texts = [m.content for m in graph_input["messages"]]
        assert texts == ["h1", "q"]


    def test_model_http_client_closed_after_stream(self, monkeypatch):
        """Fix [4]: the per-request httpx client is closed after the stream.

        ``run_agent_chat_stream`` builds a REAL model (localhost endpoint; the
        constructor performs no network I/O) and must close its
        ``http_async_client`` in the finally block even though the agent here
        is faked — repeated /ask calls must never leak connections.
        """
        import api.agents.expert as expert_mod

        monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://localhost:9/v1")
        monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)

        captured: dict = {}

        def _fake_build(product_id, model=None, session_factory=None,
                        checkpointer=None, chat=None):
            captured["chat"] = chat
            return _agent_for(["closed"])

        monkeypatch.setattr(expert_mod, "build_expert_agent", _fake_build)

        async def _run():
            async for _ in expert_mod.run_agent_chat_stream("prod_1", "hi"):
                pass

        asyncio.run(_run())

        chat = captured.get("chat")
        assert chat is not None, "build_expert_agent did not receive the model"
        client = getattr(chat, "http_async_client", None)
        assert client is not None, "model has no http_async_client attribute"
        assert client.is_closed is True


class TestHistoryMessages:
    def test_history_messages_bounded_and_filtered(self):
        from api.agents.expert import MAX_HISTORY_MESSAGES, _history_messages

        # 25 user rows + assistant + system + empty-user. The bound slices the
        # RAW list to the last 20 entries (q5..q24, 'a', 's', ''), then the
        # filter drops system + empty: 17 q's + 'a' remain.
        history = (
            [
                {"role": "user", "content": f"q{i}"}
                for i in range(MAX_HISTORY_MESSAGES + 5)
            ]
            + [
                {"role": "assistant", "content": "a"},
                {"role": "system", "content": "s"},
                {"role": "user", "content": ""},
            ]
        )
        out = _history_messages(history)
        assert len(out) == MAX_HISTORY_MESSAGES - 3 + 1
        assert all(isinstance(m, (HumanMessage, AIMessage)) for m in out)
        assert all(m.content for m in out)

    def test_history_messages_empty(self):
        from api.agents.expert import _history_messages

        assert _history_messages([]) == []
        assert _history_messages(None) == []
