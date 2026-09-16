"""Unit tests for the process-level LLM rps limiter (api/llm/client.py).

Covers: _LLMRateLimiter slot queuing per (base_url, model),
_is_rate_limit_error matching, and the _agenerate spacing + bounded 429
retry path (the gateway "6/5 requests per second" defect).
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

import api.llm.client as llm_client


def _ok_result():
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


class TestLLMRateLimiterSlots(unittest.TestCase):
    def test_slots_queue_evenly(self):
        lim = llm_client._LLMRateLimiter()
        first = lim.reserve(("u", "m"), 0.1)
        second = lim.reserve(("u", "m"), 0.1)
        third = lim.reserve(("u", "m"), 0.1)
        self.assertEqual(first, 0.0)
        self.assertAlmostEqual(second, 0.1, delta=0.02)
        self.assertAlmostEqual(third, 0.2, delta=0.05)

    def test_keys_are_independent(self):
        lim = llm_client._LLMRateLimiter()
        self.assertEqual(lim.reserve(("u", "m1"), 30.0), 0.0)
        self.assertEqual(lim.reserve(("u", "m2"), 30.0), 0.0)

    def test_expired_slot_is_free(self):
        lim = llm_client._LLMRateLimiter()
        lim._next_ok[("u", "m")] = 0.0  # slot far in the past
        self.assertEqual(lim.reserve(("u", "m"), 0.1), 0.0)


class TestIsRateLimitError(unittest.TestCase):
    def test_gateway_429_detail(self):
        exc = RuntimeError(
            "Error code: 429 - {'detail': 'Rate limit exceeded: 6/5 requests "
            "per second for generation endpoint. Used model: copilot-flash'}"
        )
        self.assertTrue(llm_client._is_rate_limit_error(exc))

    def test_class_name_match(self):
        RateLimitError = type("RateLimitError", (Exception,), {})
        self.assertTrue(llm_client._is_rate_limit_error(RateLimitError("slow down")))

    def test_non_rate_error(self):
        self.assertFalse(llm_client._is_rate_limit_error(ValueError("bad payload")))


class TestAgenerateRateLimitPath(unittest.TestCase):
    def _model(self):
        return llm_client.build_chat_model(
            model="qwen/qwen3.6-27b",
            base_url="http://localhost:9/v1",
            api_key="not-needed",
        )

    def test_build_sets_rate_key(self):
        model = self._model()
        self.assertEqual(model._rate_key, ("http://localhost:9/v1", "qwen/qwen3.6-27b"))

    def test_agenerate_reserves_slot(self):
        model = self._model()
        fresh = llm_client._LLMRateLimiter()

        async def _fake(self, messages, stop=None, run_manager=None, **kwargs):
            return _ok_result()

        async def _run():
            return await model.ainvoke([HumanMessage(content="hi")])

        with patch.object(ChatOpenAI, "_agenerate", _fake), \
                patch.object(llm_client, "_llm_rate_limiter", fresh):
            result = asyncio.run(_run())
        self.assertEqual(result.content, "ok")
        self.assertGreater(fresh._next_ok[model._rate_key], 0.0)

    def test_retries_429_then_succeeds(self):
        state = {"calls": 0}

        async def _fake(self, messages, stop=None, run_manager=None, **kwargs):
            state["calls"] += 1
            if state["calls"] < 3:
                raise RuntimeError("Error code: 429 - rate limit exceeded")
            return _ok_result()

        model = self._model()

        async def _run():
            return await model.ainvoke([HumanMessage(content="hi")])

        with patch.object(ChatOpenAI, "_agenerate", _fake), \
                patch.object(llm_client, "_rate_backoff", lambda attempt: 0.0), \
                patch("api.config.timeout.resolve_llm_rate_limit_rps", lambda: 1e6):
            result = asyncio.run(_run())
        self.assertEqual(result.content, "ok")
        self.assertEqual(state["calls"], 3)

    def test_persistent_429_raises_after_bounded_attempts(self):
        state = {"calls": 0}

        async def _fake(self, messages, stop=None, run_manager=None, **kwargs):
            state["calls"] += 1
            raise RuntimeError("429 too many requests")

        model = self._model()

        async def _run():
            return await model.ainvoke([HumanMessage(content="hi")])

        with patch.object(ChatOpenAI, "_agenerate", _fake), \
                patch.object(llm_client, "_rate_backoff", lambda attempt: 0.0), \
                patch("api.config.timeout.resolve_llm_rate_limit_rps", lambda: 1e6):
            with self.assertRaises(RuntimeError):
                asyncio.run(_run())
        self.assertEqual(state["calls"], 3)

    def test_non_rate_error_propagates_immediately(self):
        state = {"calls": 0}

        async def _fake(self, messages, stop=None, run_manager=None, **kwargs):
            state["calls"] += 1
            raise ValueError("bad payload")

        model = self._model()

        async def _run():
            return await model.ainvoke([HumanMessage(content="hi")])

        with patch.object(ChatOpenAI, "_agenerate", _fake), \
                patch("api.config.timeout.resolve_llm_rate_limit_rps", lambda: 1e6):
            with self.assertRaises(ValueError):
                asyncio.run(_run())
        self.assertEqual(state["calls"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
