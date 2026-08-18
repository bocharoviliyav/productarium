"""Rate limiter for the shared embedder (OpenAI-compatible /v1/embeddings).

Mirrors ``api.cognee.rate_limiter.CogneeRateLimiter`` (async semaphore + request
spacing + 429 retry) but is scoped to the embedder used by the pgvector memory
backend. Settings live under the ``embedder.*`` prefix, resolved with the same
precedence as cognee's (admin settings store > env var > default):

- ``embedder.max_concurrency`` (int, default 4)
- ``embedder.delay_seconds`` (float, default 0.1)
- ``embedder.rate_limit_rps`` (float, default 10.0 -> 0.1s spacing)

The adalflow ``Embedder`` is synchronous, so callers wrap their
``asyncio.to_thread(...)`` call in :meth:`execute` to throttle concurrent and
bursty embedding requests without blocking the event loop. Settings are read
through on every call (no caching), so an admin save takes effect immediately
without a restart.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class EmbedderRateLimiter:
    """Async semaphore + request spacing + 429 retry for embedder calls."""

    def __init__(self):
        # Map: loop_id -> (Semaphore, Lock, max_concurrency)
        self._loop_primitives: Dict[int, Tuple[asyncio.Semaphore, asyncio.Lock, int]] = {}
        self.last_call_time: float = 0.0

    def get_rate_settings(self) -> Tuple[int, float]:
        """Read embedder rate-limit settings (admin store > env > default).

        Returns ``(max_concurrency, delay_seconds)`` where ``delay_seconds`` is
        the larger of the explicit delay and the delay implied by
        ``rate_limit_rps`` (1.0 / rps), so a burst can never exceed the RPS cap.
        """
        max_conc = 4
        delay_sec = 0.1
        try:
            from api.config.settings import get_setting

            mc = get_setting("embedder.max_concurrency") or os.environ.get("EMBEDDER_MAX_CONCURRENCY")
            if mc:
                try:
                    max_conc = max(1, int(str(mc).strip()))
                except ValueError:
                    pass
            ds = get_setting("embedder.delay_seconds") or os.environ.get("EMBEDDER_DELAY_SECONDS")
            if ds:
                try:
                    delay_sec = max(0.0, float(str(ds).strip()))
                except ValueError:
                    pass
            rps = get_setting("embedder.rate_limit_rps") or os.environ.get("EMBEDDER_RATE_LIMIT_RPS")
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

    def _get_loop_primitives(
        self, max_concurrency: int
    ) -> Tuple[Optional[asyncio.Semaphore], Optional[asyncio.Lock]]:
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
        """Run ``func`` (an awaitable-returning call) under the rate limits.

        ``func`` is typically ``asyncio.to_thread`` wrapping a synchronous
        embedder call, so the concurrency boundary is the spawned worker
        threads rather than the embedder itself.
        """
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
                    if (
                        ("429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg)
                        and attempt < max_retries - 1
                    ):
                        backoff = (attempt + 1) * 2.5
                        logger.warning(
                            "Embedder call hit rate limit (attempt %d/%d). Sleeping %.1fs: %s",
                            attempt + 1, max_retries, backoff, e,
                        )
                        await asyncio.sleep(backoff)
                    else:
                        raise

        if sem:
            async with sem:
                return await _run()
        return await _run()


_embedder_rate_limiter = EmbedderRateLimiter()
