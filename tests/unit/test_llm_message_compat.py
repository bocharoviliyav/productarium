"""Unit tests for the strict-server chat message compatibility layer.

Regression tests for the live LM Studio failure that killed whole
orchestrated docgen runs (deepagents orchestrator transcripts) with::

    Error code: 400 - {'error': {'message': 'messages.2: Value error,
    "name" is only valid on role="tool" messages.', ...}}

langchain-openai parses a ``name`` echoed by the server into
``AIMessage.name`` (and agent middleware may attach names to messages) and
then re-sends it on every subsequent request; strict local servers reject
``name`` on any non-tool message. ``build_chat_model`` therefore returns
``ServerCompatChatOpenAI``, which strips ``name`` from non-tool messages on
every request path.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def _named_messages():
    return [
        SystemMessage(content="sys", name="sys-name"),
        HumanMessage(content="hi", name="user-1"),
        AIMessage(
            content="ok",
            name="assistant-1",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"subagent_type": "section-overview"},
                    "id": "call_1",
                }
            ],
        ),
        ToolMessage(content="result", tool_call_id="call_1", name="task"),
    ]


class TestStripNonToolMessageNames(unittest.TestCase):
    def test_strips_names_from_non_tool_messages(self):
        from api.llm.client import strip_non_tool_message_names

        msgs = _named_messages()
        out = strip_non_tool_message_names(msgs)

        self.assertIsNone(out[0].name)
        self.assertIsNone(out[1].name)
        self.assertIsNone(out[2].name)
        # The assistant payload (tool calls) is intact — only name goes.
        self.assertEqual(out[2].tool_calls, msgs[2].tool_calls)
        # Tool messages keep their name (valid on the tool role).
        self.assertIs(out[3], msgs[3])

    def test_originals_are_not_mutated(self):
        from api.llm.client import strip_non_tool_message_names

        msgs = _named_messages()
        strip_non_tool_message_names(msgs)

        self.assertEqual(msgs[0].name, "sys-name")
        self.assertEqual(msgs[1].name, "user-1")
        self.assertEqual(msgs[2].name, "assistant-1")

    def test_additional_kwargs_name_cleared_without_mutation(self):
        from api.llm.client import strip_non_tool_message_names

        msg = AIMessage(content="x", additional_kwargs={"name": "echo", "foo": 1})
        out = strip_non_tool_message_names([msg])

        self.assertNotIn("name", out[0].additional_kwargs)
        self.assertEqual(out[0].additional_kwargs["foo"], 1)
        # The original message's kwargs dict is untouched.
        self.assertEqual(msg.additional_kwargs["name"], "echo")

    def test_nameless_messages_pass_through_unchanged(self):
        from api.llm.client import strip_non_tool_message_names

        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        out = strip_non_tool_message_names(msgs)

        self.assertIs(out[0], msgs[0])
        self.assertIs(out[1], msgs[1])

    def test_serialized_payload_has_no_non_tool_name(self):
        from langchain_openai.chat_models.base import _convert_message_to_dict

        from api.llm.client import strip_non_tool_message_names

        for msg in strip_non_tool_message_names(_named_messages()):
            payload = _convert_message_to_dict(msg)
            # No message may serialize a name for the wire (langchain-openai
            # only keeps name on the ToolMessage object, not the payload).
            self.assertFalse(payload.get("name"), payload)
            if payload.get("role") == "tool":
                self.assertEqual(payload.get("tool_call_id"), "call_1")


class TestServerCompatChatOpenAI(unittest.TestCase):
    """The model built by ``build_chat_model`` must scrub names on the
    request path (sync + async) — not just expose a helper."""

    def _build(self):
        from api.llm.client import ServerCompatChatOpenAI

        return ServerCompatChatOpenAI(
            model="test-model",
            api_key="test",
            base_url="http://localhost:9/v1",
        )

    def test_invoke_strips_names(self):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_openai import ChatOpenAI

        captured: dict = {}

        def _fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured["messages"] = messages
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="ok"))]
            )

        model = self._build()
        with patch.object(ChatOpenAI, "_generate", _fake_generate):
            result = model.invoke([HumanMessage(content="hi", name="user-1")])

        self.assertEqual(result.content, "ok")
        self.assertIsNone(captured["messages"][0].name)

    def test_ainvoke_strips_names(self):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_openai import ChatOpenAI

        captured: dict = {}

        async def _fake_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            captured["messages"] = messages
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="ok"))]
            )

        async def _run():
            model = self._build()
            with patch.object(ChatOpenAI, "_agenerate", _fake_agenerate):
                return await model.ainvoke(
                    [SystemMessage(content="s", name="sys"), HumanMessage(content="hi")]
                )

        result = asyncio.run(_run())
        self.assertEqual(result.content, "ok")
        self.assertIsNone(captured["messages"][0].name)
        self.assertIsNone(captured["messages"][1].name)

    def test_build_chat_model_returns_compat_class(self):
        from api.llm.client import ServerCompatChatOpenAI, build_chat_model

        model = build_chat_model(
            model="qwen/qwen3.6-27b",
            base_url="http://localhost:9/v1",
            api_key="not-needed",
        )
        self.assertIsInstance(model, ServerCompatChatOpenAI)
        # And it is still a ChatOpenAI for every downstream isinstance check.
        from langchain_openai import ChatOpenAI

        self.assertIsInstance(model, ChatOpenAI)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
