"""Unit tests for ``api.cognee.rate_limiter``.

Covers:
- ``CogneeRateLimiter.get_rate_settings``: defaults (2, 0.5), admin-store
  overrides (cognee.max_concurrency, cognee.delay_seconds,
  cognee.rate_limit_rps), env-var fallbacks (COGNEE_MAX_CONCURRENCY,
  COGNEE_DELAY_SECONDS, COGNEE_RATE_LIMIT_RPS), invalid values ignored.
- ``CogneeRateLimiter._get_loop_primitives``: per-loop semaphore/lock caching,
  re-creation when max_concurrency changes, None when no running loop.
- ``CogneeRateLimiter.execute``: successful call returns result, concurrency
  semaphore gating, 429 retry with backoff (succeeds on retry), non-429
  errors propagate, max-retries exhausted raises.
- ``_apply_cognee_rate_limit_patches``: idempotent litellm/openai monkeypatch
  application (guarded by _productarium_rate_limited flag), no-op when the
  libraries are absent.
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.cognee.rate_limiter import (
    CogneeRateLimiter,
    _apply_cognee_rate_limit_patches,
    _cognee_rate_limiter,
)


# --------------------------------------------------------------------------- #
# get_rate_settings
# --------------------------------------------------------------------------- #
class TestGetRateSettings:
    def test_defaults_when_nothing_set(self, monkeypatch):
        """Returns (2, 0.5) when no settings or env vars are configured."""
        monkeypatch.delenv("COGNEE_MAX_CONCURRENCY", raising=False)
        monkeypatch.delenv("COGNEE_DELAY_SECONDS", raising=False)
        monkeypatch.delenv("COGNEE_RATE_LIMIT_RPS", raising=False)
        rl = CogneeRateLimiter()
        with patch("api.config.settings.get_setting", return_value=None):
            mc, ds = rl.get_rate_settings()
        assert mc == 2
        assert ds == 0.5

    def test_env_max_concurrency(self, monkeypatch):
        monkeypatch.setenv("COGNEE_MAX_CONCURRENCY", "5")
        rl = CogneeRateLimiter()
        with patch("api.config.settings.get_setting", return_value=None):
            mc, ds = rl.get_rate_settings()
        assert mc == 5

    def test_env_delay_seconds(self, monkeypatch):
        monkeypatch.setenv("COGNEE_DELAY_SECONDS", "1.5")
        rl = CogneeRateLimiter()
        with patch("api.config.settings.get_setting", return_value=None):
            mc, ds = rl.get_rate_settings()
        assert ds == 1.5

    def test_env_rate_limit_rps(self, monkeypatch):
        monkeypatch.delenv("COGNEE_DELAY_SECONDS", raising=False)
        monkeypatch.setenv("COGNEE_RATE_LIMIT_RPS", "4.0")
        rl = CogneeRateLimiter()
        with patch("api.config.settings.get_setting", return_value=None):
            mc, ds = rl.get_rate_settings()
        # rps=4 -> delay = 1/4 = 0.25, but max(delay_sec_default=0.5, 0.25)=0.5
        assert ds == 0.5

    def test_env_rate_limit_rps_low_rps(self, monkeypatch):
        monkeypatch.delenv("COGNEE_DELAY_SECONDS", raising=False)
        monkeypatch.setenv("COGNEE_RATE_LIMIT_RPS", "1.0")
        rl = CogneeRateLimiter()
        with patch("api.config.settings.get_setting", return_value=None):
            mc, ds = rl.get_rate_settings()
        # rps=1 -> delay = 1/1 = 1.0 > 0.5 default
        assert ds == 1.0

    def test_admin_store_max_concurrency(self, monkeypatch):
        rl = CogneeRateLimiter()

        def _fake_get(key, default=None):
            if key == "cognee.max_concurrency":
                return "8"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert mc == 8

    def test_admin_store_delay_seconds(self, monkeypatch):
        rl = CogneeRateLimiter()

        def _fake_get(key, default=None):
            if key == "cognee.delay_seconds":
                return "2.0"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert ds == 2.0

    def test_admin_store_rate_limit_rps(self, monkeypatch):
        rl = CogneeRateLimiter()

        def _fake_get(key, default=None):
            if key == "cognee.rate_limit_rps":
                return "0.5"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        # rps=0.5 -> delay=2.0 > 0.5 default
        assert ds == 2.0

    def test_admin_store_rate_limit_delay_alias(self, monkeypatch):
        """cognee.rate_limit_delay is an alias for cognee.delay_seconds."""
        rl = CogneeRateLimiter()

        def _fake_get(key, default=None):
            if key == "cognee.rate_limit_delay":
                return "3.0"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert ds == 3.0

    def test_invalid_max_concurrency_ignored(self, monkeypatch):
        rl = CogneeRateLimiter()

        def _fake_get(key, default=None):
            if key == "cognee.max_concurrency":
                return "not-a-number"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert mc == 2  # default

    def test_invalid_delay_seconds_ignored(self, monkeypatch):
        rl = CogneeRateLimiter()

        def _fake_get(key, default=None):
            if key == "cognee.delay_seconds":
                return "bad"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert ds == 0.5  # default

    def test_invalid_rate_limit_rps_ignored(self, monkeypatch):
        monkeypatch.delenv("COGNEE_DELAY_SECONDS", raising=False)
        rl = CogneeRateLimiter()

        def _fake_get(key, default=None):
            if key == "cognee.rate_limit_rps":
                return "bad"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert ds == 0.5  # default

    def test_zero_rps_ignored(self, monkeypatch):
        """rps=0 would divide by zero; must be ignored."""
        monkeypatch.delenv("COGNEE_DELAY_SECONDS", raising=False)
        rl = CogneeRateLimiter()

        def _fake_get(key, default=None):
            if key == "cognee.rate_limit_rps":
                return "0"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert ds == 0.5  # default, no crash

    def test_negative_max_concurrency_clamped_to_1(self, monkeypatch):
        rl = CogneeRateLimiter()

        def _fake_get(key, default=None):
            if key == "cognee.max_concurrency":
                return "-5"
            return None

        with patch("api.config.settings.get_setting", side_effect=_fake_get):
            mc, ds = rl.get_rate_settings()
        assert mc == 1  # max(1, int(-5)) == 1 ... actually max(1, -5) == 1

    def test_get_setting_exception_returns_defaults(self, monkeypatch):
        rl = CogneeRateLimiter()

        def _boom(key, default=None):
            raise RuntimeError("db down")

        with patch("api.config.settings.get_setting", side_effect=_boom):
            mc, ds = rl.get_rate_settings()
        assert mc == 2
        assert ds == 0.5


# --------------------------------------------------------------------------- #
# _get_loop_primitives
# --------------------------------------------------------------------------- #
class TestGetLoopPrimitives:
    def test_returns_none_when_no_running_loop(self):
        rl = CogneeRateLimiter()
        # Called outside an event loop -> (None, None)
        sem, lock = rl._get_loop_primitives(2)
        assert sem is None
        assert lock is None

    def test_returns_semaphore_and_lock_in_loop(self):
        rl = CogneeRateLimiter()

        async def _run():
            sem, lock = rl._get_loop_primitives(2)
            return sem, lock

        sem, lock = asyncio.run(_run())
        assert sem is not None
        assert lock is not None
        assert isinstance(sem, asyncio.Semaphore)
        assert isinstance(lock, asyncio.Lock)

    def test_cached_primitives_reused_same_loop(self):
        rl = CogneeRateLimiter()

        async def _run():
            sem1, lock1 = rl._get_loop_primitives(2)
            sem2, lock2 = rl._get_loop_primitives(2)
            return sem1 is sem2, lock1 is lock2

        same_sem, same_lock = asyncio.run(_run())
        assert same_sem
        assert same_lock

    def test_primitives_recreated_when_max_concurrency_changes(self):
        rl = CogneeRateLimiter()

        async def _run():
            sem1, lock1 = rl._get_loop_primitives(2)
            sem2, lock2 = rl._get_loop_primitives(4)
            return sem1, sem2

        sem1, sem2 = asyncio.run(_run())
        assert sem1 is not sem2


# --------------------------------------------------------------------------- #
# execute
# --------------------------------------------------------------------------- #
class TestExecute:
    def test_successful_call_returns_result(self):
        rl = CogneeRateLimiter()

        async def _func():
            return "ok"

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            result = asyncio.run(rl.execute(_func))
        assert result == "ok"

    def test_concurrency_semaphore_gating(self):
        """Only max_concurrency=1 tasks run at a time."""
        rl = CogneeRateLimiter()
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
        rl = CogneeRateLimiter()
        call_count = 0

        async def _func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("429 Too Many Requests")
            return "recovered"

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            # Patch asyncio.sleep to avoid real backoff delay.
            with patch("api.cognee.rate_limiter.asyncio.sleep", new_callable=_NoSleep):
                result = asyncio.run(rl.execute(_func))
        assert result == "recovered"
        assert call_count == 2

    def test_rate_limit_message_retry(self):
        rl = CogneeRateLimiter()
        call_count = 0

        async def _func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("rate limit exceeded")
            return "ok"

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            with patch("api.cognee.rate_limiter.asyncio.sleep", new_callable=_NoSleep):
                result = asyncio.run(rl.execute(_func))
        assert result == "ok"

    def test_too_many_requests_retry(self):
        rl = CogneeRateLimiter()
        call_count = 0

        async def _func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Too many requests")
            return "ok"

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            with patch("api.cognee.rate_limiter.asyncio.sleep", new_callable=_NoSleep):
                result = asyncio.run(rl.execute(_func))
        assert result == "ok"

    def test_non_429_error_propagates(self):
        rl = CogneeRateLimiter()

        async def _func():
            raise ValueError("not a rate limit")

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            with pytest.raises(ValueError, match="not a rate limit"):
                asyncio.run(rl.execute(_func))

    def test_max_retries_exhausted_raises(self):
        rl = CogneeRateLimiter()
        call_count = 0

        async def _func():
            nonlocal call_count
            call_count += 1
            raise Exception("429 rate limited")

        with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
            with patch("api.cognee.rate_limiter.asyncio.sleep", new_callable=_NoSleep):
                with pytest.raises(Exception, match="429"):
                    asyncio.run(rl.execute(_func))
        # max_retries = 5
        assert call_count == 5

    def test_no_semaphore_when_no_loop_in_get_primitives(self):
        """When _get_loop_primitives returns (None, None), execute still runs."""
        rl = CogneeRateLimiter()

        async def _func():
            return "ok"

        async def _run():
            with patch.object(rl, "get_rate_settings", return_value=(1, 0.0)):
                with patch.object(rl, "_get_loop_primitives", return_value=(None, None)):
                    return await rl.execute(_func)

        result = asyncio.run(_run())
        assert result == "ok"

    def test_delay_between_calls(self):
        """When delay_sec > 0, execute enforces a minimum gap via the lock."""
        rl = CogneeRateLimiter()
        timestamps = []

        async def _func():
            timestamps.append(asyncio.get_running_loop().time())
            return "ok"

        async def _run():
            with patch.object(rl, "get_rate_settings", return_value=(1, 0.05)):
                await rl.execute(_func)
                await rl.execute(_func)

        asyncio.run(_run())
        # Second call should be delayed by ~0.05s after the first.
        assert len(timestamps) == 2
        assert timestamps[1] - timestamps[0] >= 0.04  # allow small jitter


class _NoSleep:
    """A no-op async mock for asyncio.sleep to speed up retry backoff tests."""

    async def __call__(self, *args, **kwargs):
        pass


# --------------------------------------------------------------------------- #
# _apply_cognee_rate_limit_patches
# --------------------------------------------------------------------------- #
class TestApplyRateLimitPatches:
    def test_module_level_instance_exists(self):
        assert isinstance(_cognee_rate_limiter, CogneeRateLimiter)

    def test_patches_noop_when_litellm_absent(self):
        """litellm is not installed locally; the patch must not raise."""
        # Should be a no-op (caught by the except branch).
        _apply_cognee_rate_limit_patches()

    def test_patches_idempotent_when_already_patched(self):
        """Calling twice does not re-patch (guarded by _productarium_rate_limited)."""
        _apply_cognee_rate_limit_patches()
        _apply_cognee_rate_limit_patches()  # should not raise

    def test_patches_litellm_when_present(self, monkeypatch):
        """When a fake litellm module is injected, the patch wraps acompletion."""
        import types

        fake_litellm = types.ModuleType("litellm")
        call_log = []

        async def _orig_acompletion(*args, **kwargs):
            call_log.append("acompletion")
            return {"result": "ok"}

        async def _orig_aembedding(*args, **kwargs):
            call_log.append("aembedding")
            return {"result": "ok"}

        fake_litellm.acompletion = _orig_acompletion
        fake_litellm.aembedding = _orig_aembedding
        fake_litellm._productarium_rate_limited = False

        # Temporarily inject into sys.modules
        monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

        _apply_cognee_rate_limit_patches()

        # Verify the flag was set
        assert fake_litellm._productarium_rate_limited is True

        # The patched functions should route through the rate limiter.
        # Verify they are wrapped (not the originals).
        assert fake_litellm.acompletion is not _orig_acompletion
        assert fake_litellm.aembedding is not _orig_aembedding

        # Call them and verify they still work.
        result = asyncio.run(fake_litellm.acompletion("test"))
        assert result == {"result": "ok"}

        result2 = asyncio.run(fake_litellm.aembedding("test"))
        assert result2 == {"result": "ok"}

    def test_patches_openai_async_completions_when_present(self, monkeypatch):
        """When openai AsyncCompletions exists, the patch wraps create."""
        import types

        fake_openai = types.ModuleType("openai")
        fake_resources = types.ModuleType("openai.resources")
        fake_chat = types.ModuleType("openai.resources.chat")
        fake_embeddings = types.ModuleType("openai.resources.embeddings")

        class _AsyncCompletions:
            _productarium_rate_limited = False

            async def create(self, *args, **kwargs):
                return {"completion": "ok"}

        class _AsyncEmbeddings:
            _productarium_rate_limited = False

            async def create(self, *args, **kwargs):
                return {"embedding": "ok"}

        _AsyncCompletions_orig = _AsyncCompletions.create
        _AsyncEmbeddings_orig = _AsyncEmbeddings.create

        fake_chat.AsyncCompletions = _AsyncCompletions
        fake_embeddings.AsyncEmbeddings = _AsyncEmbeddings
        fake_resources.chat = fake_chat
        fake_resources.embeddings = fake_embeddings
        fake_openai.resources = fake_resources

        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.setitem(sys.modules, "openai.resources", fake_resources)
        monkeypatch.setitem(sys.modules, "openai.resources.chat", fake_chat)
        monkeypatch.setitem(sys.modules, "openai.resources.embeddings", fake_embeddings)

        _apply_cognee_rate_limit_patches()

        # Verify flags set
        assert _AsyncCompletions._productarium_rate_limited is True
        assert _AsyncEmbeddings._productarium_rate_limited is True

        # Verify create is wrapped
        assert _AsyncCompletions.create is not _AsyncCompletions_orig
        assert _AsyncEmbeddings.create is not _AsyncEmbeddings_orig

        # Call and verify
        inst = _AsyncCompletions()
        result = asyncio.run(inst.create("test"))
        assert result == {"completion": "ok"}
