from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class CogneeRateLimiter:
    """Async Semaphore, Rate Limiter, and 429 Retry Handler for Cognee calls."""

    def __init__(self):
        # Map: loop_id -> (Semaphore, Lock, max_concurrency)
        self._loop_primitives: Dict[int, Tuple[asyncio.Semaphore, asyncio.Lock, int]] = {}
        self.last_call_time: float = 0.0

    def get_rate_settings(self) -> Tuple[int, float]:
        """Read rate limit settings from DB settings store or environment.

        - cognee.max_concurrency: int (default 2)
        - cognee.delay_seconds: float (default 0.5s -> max 2 requests/sec)
        - cognee.rate_limit_rps: float (e.g. 2.0 -> 0.5s delay)
        """
        max_conc = 2
        delay_sec = 0.5
        try:
            from api.config.settings import get_setting
            mc = get_setting("cognee.max_concurrency") or os.environ.get("COGNEE_MAX_CONCURRENCY")
            if mc:
                try:
                    max_conc = max(1, int(str(mc).strip()))
                except ValueError:
                    pass
            ds = get_setting("cognee.delay_seconds") or get_setting("cognee.rate_limit_delay") or os.environ.get("COGNEE_DELAY_SECONDS")
            if ds:
                try:
                    delay_sec = max(0.0, float(str(ds).strip()))
                except ValueError:
                    pass
            rps = get_setting("cognee.rate_limit_rps") or os.environ.get("COGNEE_RATE_LIMIT_RPS")
            if rps:
                try:
                    val = float(str(rps).strip())
                    if val > 0:
                        delay_sec = max(delay_sec, 1.0 / val)
                except ValueError:
                    pass
        except Exception:
            pass
        return max_conc, delay_sec

    def _get_loop_primitives(self, max_concurrency: int) -> Tuple[Optional[asyncio.Semaphore], Optional[asyncio.Lock]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None, None

        loop_id = id(loop)
        if loop_id in self._loop_primitives:
            sem, lock, cached_max = self._loop_primitives[loop_id]
            if cached_max == max_concurrency:
                return sem, lock

        sem = asyncio.Semaphore(max_concurrency)
        lock = asyncio.Lock()
        self._loop_primitives[loop_id] = (sem, lock, max_concurrency)
        return sem, lock

    async def execute(self, func, *args, **kwargs):
        max_conc, delay_sec = self.get_rate_settings()
        sem, lock = self._get_loop_primitives(max_conc)

        async def _run():
            if lock and delay_sec > 0:
                async with lock:
                    now = asyncio.get_running_loop().time()
                    elapsed = now - self.last_call_time
                    if elapsed < delay_sec:
                        await asyncio.sleep(delay_sec - elapsed)
                    self.last_call_time = asyncio.get_running_loop().time()

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    err_msg = str(e).lower()
                    if ("429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg) and attempt < max_retries - 1:
                        backoff = (attempt + 1) * 2.5
                        logger.warning(
                            "Cognee LLM/Embedding call hit rate limit (attempt %d/%d). Sleeping %.1fs: %s",
                            attempt + 1, max_retries, backoff, e,
                        )
                        await asyncio.sleep(backoff)
                    else:
                        raise

        if sem:
            async with sem:
                return await _run()
        else:
            return await _run()


_cognee_rate_limiter = CogneeRateLimiter()


def _apply_cognee_rate_limit_patches():
    """Apply rate limiter monkeypatches to litellm and openai clients for cognee."""
    try:
        import litellm
        if not getattr(litellm, "_productarium_rate_limited", False):
            orig_acompletion = getattr(litellm, "acompletion", None)
            orig_aembedding = getattr(litellm, "aembedding", None)

            if callable(orig_acompletion):
                async def _patched_acompletion(*args, **kwargs):
                    return await _cognee_rate_limiter.execute(orig_acompletion, *args, **kwargs)
                setattr(litellm, "acompletion", _patched_acompletion)

            if callable(orig_aembedding):
                async def _patched_aembedding(*args, **kwargs):
                    return await _cognee_rate_limiter.execute(orig_aembedding, *args, **kwargs)
                setattr(litellm, "aembedding", _patched_aembedding)

            setattr(litellm, "_productarium_rate_limited", True)
            logger.info("Cognee litellm rate-limiter & 429 retry patch applied.")
    except Exception as e:
        logger.debug("Could not patch litellm for cognee rate limiting: %s", e)

    try:
        import openai
        if hasattr(openai, "resources") and hasattr(openai.resources.chat, "AsyncCompletions"):
            ac_cls = openai.resources.chat.AsyncCompletions
            if not getattr(ac_cls, "_productarium_rate_limited", False):
                orig_ac_create = ac_cls.create
                async def _patched_ac_create(self_obj, *args, **kwargs):
                    return await _cognee_rate_limiter.execute(orig_ac_create, self_obj, *args, **kwargs)
                ac_cls.create = _patched_ac_create
                ac_cls._productarium_rate_limited = True

        if hasattr(openai, "resources") and hasattr(openai.resources.embeddings, "AsyncEmbeddings"):
            ae_cls = openai.resources.embeddings.AsyncEmbeddings
            if not getattr(ae_cls, "_productarium_rate_limited", False):
                orig_ae_create = ae_cls.create
                async def _patched_ae_create(self_obj, *args, **kwargs):
                    return await _cognee_rate_limiter.execute(orig_ae_create, self_obj, *args, **kwargs)
                ae_cls.create = _patched_ae_create
                ae_cls._productarium_rate_limited = True
    except Exception as e:
        logger.debug("Could not patch openai AsyncCompletions for cognee rate limiting: %s", e)
