"""Rate limiter for the shared embedder (OpenAI-compatible /v1/embeddings).

Async semaphore + request spacing + 429 retry, scoped to the embedder used by
the pgvector memory backend. Settings live under the ``embedder.*`` prefix,
resolved with admin settings store > env var > default precedence:

- ``embedder.max_concurrency`` (int, default 4)
- ``embedder.delay_seconds`` (float, default 0.1)
- ``embedder.rate_limit_rps`` (float, default 10.0 -> 0.1s spacing)

P1-21: the limiter is PROCESS-LEVEL. Concurrency is a single
``threading.BoundedSemaphore`` and spacing is a monotonic-clock slot reservation
guarded by a ``threading.Lock`` — so every event loop in the process (the main
FastAPI loop AND the per-job docgen worker loops) shares ONE limit. The old
per-``id(loop)`` asyncio-primitive dict was dropped: it created independent
limits per loop and leaked entries for dead loops. Waiting on the semaphore
happens in a worker thread (``asyncio.to_thread(sem.acquire)``) so a saturated
limiter never blocks the event loop.

The langchain ``OpenAIEmbeddings`` client exposes synchronous embed calls, so
callers wrap their ``asyncio.to_thread(...)`` call in :meth:`execute` to
throttle concurrent and bursty embedding requests without blocking the event
loop. Settings are read through on every call (no caching), so an admin save
takes effect immediately without a restart.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class EmbedderRateLimiter:
    """Process-level concurrency semaphore + request spacing + 429 retry."""

    def __init__(self):
        # Process-level concurrency boundary shared by ALL event loops.
        self._sem: Optional[threading.BoundedSemaphore] = None
        self._sem_size: int = 0
        # Guards semaphore (re)construction and the spacing slot state.
        self._state_lock = threading.Lock()
        # Monotonic timestamp of the last reserved call slot (0 = never).
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

    def _get_semaphore(self, max_concurrency: int) -> threading.BoundedSemaphore:
        """Return the process-level semaphore, (re)creating it on config change."""
        with self._state_lock:
            if self._sem is None or self._sem_size != max_concurrency:
                self._sem = threading.BoundedSemaphore(max_concurrency)
                self._sem_size = max_concurrency
            return self._sem

    def _reserve_slot(self, delay_sec: float) -> float:
        """Reserve the next call slot under the spacing delay.

        Returns the number of seconds the caller must still wait before firing.
        Slots are queued (last_call_time advances by delay_sec per reservation)
        so N concurrent callers get evenly spaced slots instead of all waiting
        for the same instant and firing together.
        """
        with self._state_lock:
            now = time.monotonic()
            next_ok = self.last_call_time + delay_sec
            if self.last_call_time <= 0.0 or now >= next_ok:
                self.last_call_time = now
                return 0.0
            self.last_call_time = next_ok
            return next_ok - now

    async def execute(self, func, *args, **kwargs):
        """Run ``func`` (an awaitable-returning call) under the rate limits.

        ``func`` is typically ``asyncio.to_thread`` wrapping a synchronous
        embedder call. The concurrency boundary is a process-level threading
        semaphore (shared across every event loop); waiting for a free slot
        happens in a worker thread so the event loop is never blocked.
        """
        max_conc, delay_sec = self.get_rate_settings()
        sem = self._get_semaphore(max_conc)

        # Park the blocking acquire in a worker thread (never blocks the loop).
        await asyncio.to_thread(sem.acquire)
        try:
            if delay_sec > 0:
                wait = self._reserve_slot(delay_sec)
                if wait > 0:
                    await asyncio.sleep(wait)

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
        finally:
            sem.release()


_embedder_rate_limiter = EmbedderRateLimiter()
