"""Unit tests for ``api.tools.rate_limiter`` (EmbedderRateLimiter).

Targets the embedder limiter used by the pgvector memory backend:

- ``EmbedderRateLimiter.get_rate_settings``: defaults (4, 0.1), admin-store
  overrides (embedder.max_concurrency, embedder.delay_seconds,
  embedder.rate_limit_rps), env-var fallbacks (EMBEDDER_MAX_CONCURRENCY,
  EMBEDDER_DELAY_SECONDS, EMBEDDER_RATE_LIMIT_RPS), invalid values ignored.
- ``EmbedderRateLimiter._get_semaphore`` (P1-21): ONE process-level threading
  semaphore, re-created only when max_concurrency changes.
- ``EmbedderRateLimiter.execute``: successful call returns result, concurrency
  semaphore gating, 429 retry with backoff (succeeds on retry), non-429 errors
  propagate, max-retries exhausted raises. Two "loop" contexts (separate
  ``asyncio.run`` invocations / threads) share the same process-level limit.
- The module-level ``_embedder_rate_limiter`` singleton.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.tools.rate_limiter import EmbedderRateLimiter, _embedder_rate_limiter


# --------------------------------------------------------------------------- #
# get_rate_settings
# --------------------------------------------------------------------------- #
class TestGetRateSettings:
    def test_defaults_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("EMBEDDER_MAX_CONCURRENCY", raising=False)
        monkeypatch.delenv("EMBEDDER_DELAY_SECONDS", raising=False)
        monkeypatch.delenv("EMBEDDER_RATE_LIMIT_RPS", raising=False)
        rl = EmbedderRateLimiter()
        with patch("api.config.settings.get_setting", return_value=None):
            mc, ds = rl.get_rate_settings()
        assert mc == 4
        assert ds == 0.1

    def test_env_max_concurrency(self, monkeypatch):
        monkeypatch.setenv("EMBEDDER_MAX_CONCURRENCY", "6")
        rl = EmbedderRateLimiter()
        with patch("api.config.settings.get_setting", return_value=None):
            mc, ds = rl.get_rate_settings()
        assert mc == 6

    def test_env_delay_seconds(self, monkeypatch):
        monkeypatch.setenv("EMBEDDER_DELAY_SECONDS", "0.25")
        rl = EmbedderRateLimiter()
        with patch("api.config.settings.get_setting", return_value=None):
            mc, ds = rl.get_rate_settings()
        assert ds == 0.25

    def test_env_rate_limit_rps_above_default_delay(self, monkeypatch):
        # rps=20 -> 1/20 = 0.05, which is below the 0.1 default delay.
        monkeypatch.delenv("EMBEDDER_DELAY_SECONDS", raising=False)
        monkeypatch.setenv("EMBEDDER_RATE_LIMIT_RPS", "20.0")
        rl = EmbedderRateLimiter()
        with patch("api.config.settings.get_setting", return_value=None):
            mc, ds = rl.get_rate_settings()
        assert ds == 0.1

    def test_env_rate_limit_rps_low_rps(self, monkeypatch):
        # rps=2 -> 1/2 = 0.5, which exceeds the 0.1 default delay.
        monkeypatch.delenv("EMBEDDER_DELAY_SECONDS", raising=False)
        monkeypatch.setenv("EMBEDDER_RATE_LIMIT_RPS", "2.0")
        rl = EmbedderRateLimiter()
        with patch("api.config.settings.get_setting", return_value=None):
            mc, ds = rl.get_rate_settings()
        assert ds == 0.5

    def test_admin_store_max_concurrency(self):
        rl = EmbedderRateLimiter()

        def _fake_get(key, default=None):
            if key == "embedder.max_concurrency":
                return "8"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert mc == 8

    def test_admin_store_delay_seconds(self):
        rl = EmbedderRateLimiter()

        def _fake_get(key, default=None):
            if key == "embedder.delay_seconds":
                return "0.4"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert ds == 0.4

    def test_admin_store_rate_limit_rps(self):
        rl = EmbedderRateLimiter()

        def _fake_get(key, default=None):
            if key == "embedder.rate_limit_rps":
                return "5.0"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        # rps=5 -> delay=0.2 > 0.1 default
        assert ds == 0.2

    def test_invalid_max_concurrency_ignored(self):
        rl = EmbedderRateLimiter()

        def _fake_get(key, default=None):
            if key == "embedder.max_concurrency":
                return "not-a-number"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert mc == 4  # default

    def test_invalid_delay_seconds_ignored(self):
        rl = EmbedderRateLimiter()

        def _fake_get(key, default=None):
            if key == "embedder.delay_seconds":
                return "bad"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert ds == 0.1  # default

    def test_invalid_rate_limit_rps_ignored(self, monkeypatch):
        monkeypatch.delenv("EMBEDDER_DELAY_SECONDS", raising=False)
        rl = EmbedderRateLimiter()

        def _fake_get(key, default=None):
            if key == "embedder.rate_limit_rps":
                return "bad"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert ds == 0.1  # default

    def test_zero_rps_ignored(self, monkeypatch):
        """rps=0 would divide by zero; must be ignored."""
        monkeypatch.delenv("EMBEDDER_DELAY_SECONDS", raising=False)
        rl = EmbedderRateLimiter()

        def _fake_get(key, default=None):
            if key == "embedder.rate_limit_rps":
                return "0"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert ds == 0.1  # default, no crash

    def test_negative_max_concurrency_clamped_to_1(self):
        rl = EmbedderRateLimiter()

        def _fake_get(key, default=None):
            if key == "embedder.max_concurrency":
                return "-3"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert mc == 1  # max(1, int(-3)) == 1

    def test_get_setting_exception_returns_defaults(self):
        rl = EmbedderRateLimiter()

        def _boom(key, default=None):
            raise RuntimeError("db down")

        with patch("api.config.settings.get_setting", side_effect=_boom):
            mc, ds = rl.get_rate_settings()
        assert mc == 4
        assert ds == 0.1


# --------------------------------------------------------------------------- #
# _get_semaphore (P1-21: process-level primitives)
# --------------------------------------------------------------------------- #
class TestGetSemaphore:
    def test_same_semaphore_reused(self):
        rl = EmbedderRateLimiter()
        sem1 = rl._get_semaphore(4)
        sem2 = rl._get_semaphore(4)
        assert sem1 is sem2

    def test_semaphore_recreated_when_max_concurrency_changes(self):
        rl = EmbedderRateLimiter()
        sem1 = rl._get_semaphore(4)
        sem2 = rl._get_semaphore(8)
        assert sem1 is not sem2

    def test_semaphore_is_threading_based_and_loop_agnostic(self):
        rl = EmbedderRateLimiter()
        sem = rl._get_semaphore(4)
        assert isinstance(sem, threading.BoundedSemaphore)
        # Same primitive from two DIFFERENT event loops (sequential runs):
        # a process-level limiter must not reset across loop contexts.
        async def _run():
            return rl._get_semaphore(4)

        assert asyncio.run(_run()) is sem
        assert asyncio.run(_run()) is sem

    def test_two_loop_contexts_share_one_limit(self):
        """P1-21: two concurrent "loop" contexts share one process-level limit."""
        rl = EmbedderRateLimiter()
        active = 0
        max_active = 0
        guard = threading.Lock()

        async def _func():
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            with guard:
                active -= 1
            return "ok"

        def _run_one_loop():
            async def _run():
                with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
                    await asyncio.gather(rl.execute(_func), rl.execute(_func))

            asyncio.run(_run())

        # Two threads, each with its OWN event loop, sharing the ONE limiter.
        t1 = threading.Thread(target=_run_one_loop)
        t2 = threading.Thread(target=_run_one_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert max_active == 1


# --------------------------------------------------------------------------- #
# execute
# --------------------------------------------------------------------------- #
class _NoSleep:
    """No-op async mock for asyncio.sleep to speed up retry backoff tests."""

    async def __call__(self, *args, **kwargs):
        pass


class TestExecute:
    def test_successful_call_returns_result(self):
        rl = EmbedderRateLimiter()

        async def _func():
            return "ok"

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            result = asyncio.run(rl.execute(_func))
        assert result == "ok"

    def test_concurrency_semaphore_gating(self):
        """Only max_concurrency=1 task runs at a time."""
        rl = EmbedderRateLimiter()
        active = 0
        max_active = 0

        async def _func():
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return "ok"

        async def _run():
            with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
                await asyncio.gather(rl.execute(_func), rl.execute(_func), rl.execute(_func))

        asyncio.run(_run())
        assert max_active == 1

    def test_429_retry_succeeds_on_retry(self):
        rl = EmbedderRateLimiter()
        call_count = 0

        async def _func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("429 Too Many Requests")
            return "recovered"

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            with patch("api.tools.rate_limiter.asyncio.sleep", new_callable=_NoSleep):
                result = asyncio.run(rl.execute(_func))
        assert result == "recovered"
        assert call_count == 2

    def test_rate_limit_message_retry(self):
        rl = EmbedderRateLimiter()
        call_count = 0

        async def _func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("rate limit exceeded")
            return "ok"

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            with patch("api.tools.rate_limiter.asyncio.sleep", new_callable=_NoSleep):
                result = asyncio.run(rl.execute(_func))
        assert result == "ok"

    def test_too_many_requests_retry(self):
        rl = EmbedderRateLimiter()
        call_count = 0

        async def _func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Too many requests")
            return "ok"

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            with patch("api.tools.rate_limiter.asyncio.sleep", new_callable=_NoSleep):
                result = asyncio.run(rl.execute(_func))
        assert result == "ok"

    def test_non_429_error_propagates(self):
        rl = EmbedderRateLimiter()

        async def _func():
            raise ValueError("not a rate limit")

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            with pytest.raises(ValueError, match="not a rate limit"):
                asyncio.run(rl.execute(_func))

    def test_max_retries_exhausted_raises(self):
        rl = EmbedderRateLimiter()
        call_count = 0

        async def _func():
            nonlocal call_count
            call_count += 1
            raise Exception("429 rate limited")

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            with patch("api.tools.rate_limiter.asyncio.sleep", new_callable=_NoSleep):
                with pytest.raises(Exception, match="429"):
                    asyncio.run(rl.execute(_func))
        # max_retries = 5
        assert call_count == 5

    def test_semaphore_released_on_error(self):
        """A failing call must release its semaphore slot back."""
        rl = EmbedderRateLimiter()

        async def _func():
            raise ValueError("boom")

        async def _run():
            with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
                with pytest.raises(ValueError):
                    await rl.execute(_func)
            # The slot is free again: a second call succeeds.
            ok = await rl.execute(_func_ok)
            return ok

        async def _func_ok():
            return "ok"

        assert asyncio.run(_run()) == "ok"

    def test_delay_between_calls(self):
        """When delay_sec > 0, execute enforces a minimum gap via the lock."""
        rl = EmbedderRateLimiter()
        timestamps = []

        async def _func():
            timestamps.append(asyncio.get_running_loop().time())
            return "ok"

        async def _run():
            with patch.object(rl, "get_rate_settings", return_value=(1, 0.05)):
                await rl.execute(_func)
                await rl.execute(_func)

        asyncio.run(_run())
        assert len(timestamps) == 2
        assert timestamps[1] - timestamps[0] >= 0.04  # allow small jitter


class TestModuleSingleton:
    def test_module_level_instance_exists(self):
        assert isinstance(_embedder_rate_limiter, EmbedderRateLimiter)
