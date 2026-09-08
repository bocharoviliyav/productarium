"""In-memory token-bucket rate limiting (P0-7; shared infra for P1-17).

Two scoped helpers over one bucket registry:

- :func:`enforce_ip_rate_limit` — per-IP buckets for unauthenticated endpoints
  (``/api/auth/login``, ``/api/auth/reset-password``). Default ~10 requests /
  minute / IP, configurable via the admin settings store
  (``rate.auth.per_ip_minute``) with an env fallback
  (``RATE_AUTH_PER_IP_MINUTE``).
- :func:`enforce_user_rate_limit` — per-user buckets for expensive operations
  (docgen generate, expert ask, public ask — wired in P1-17). Limits read from
  the settings store with env fallbacks.

Buckets live in a plain module-level dict guarded by a ``threading.Lock``:
FastAPI runs sync endpoints in a threadpool, so thread-safety is required;
async endpoints call these helpers before the first ``await`` so a plain lock
never stalls the loop for long (the critical section is O(1)).

On limit exhaustion the helpers raise ``HTTPException(429)`` with a
``Retry-After`` header (ceil of the refill time to the next full token),
following the plan's contract.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class _Bucket:
    """Classic token bucket: ``capacity`` tokens, refilled at ``rate``/sec."""

    __slots__ = ("capacity", "rate", "tokens", "updated")

    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        self.capacity = float(capacity)
        self.rate = float(refill_per_sec)
        self.tokens = float(capacity)
        self.updated = time.monotonic()

    def try_take(self) -> Tuple[bool, float]:
        """Take one token. Returns ``(allowed, retry_after_seconds)``."""
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0.0
        # Seconds until one full token is available.
        return False, max(0.0, (1.0 - self.tokens) / self.rate) if self.rate > 0 else math.inf


_lock = threading.Lock()
_buckets: Dict[str, _Bucket] = {}

# Soft cap so a burst of unique IPs/users cannot grow the registry unboundedly.
# Eviction is lazy (oldest-touch-first would need an ordered structure; instead
# we rebuild the dict when it exceeds the cap, dropping already-dry buckets).
_MAX_BUCKETS = 10_000


def _get_bucket(key: str, capacity: float, refill_per_sec: float) -> _Bucket:
    with _lock:
        bucket = _buckets.get(key)
        if bucket is None or bucket.capacity != capacity or bucket.rate != refill_per_sec:
            # First sight of the key (or the config changed): fresh bucket.
            bucket = _Bucket(capacity, refill_per_sec)
            if len(_buckets) >= _MAX_BUCKETS:
                _evict_dry_buckets()
            _buckets[key] = bucket
        return bucket


def _evict_dry_buckets() -> None:
    """Drop fully-refilled buckets (idle keys) when the registry is too large."""
    dry = [k for k, b in _buckets.items() if b.tokens >= b.capacity]
    for k in dry:
        _buckets.pop(k, None)


def reset_rate_limits() -> None:
    """Clear all buckets (used by tests and admin 'clear limits' actions)."""
    with _lock:
        _buckets.clear()


def _setting_int(key: str, env_name: str, default: int) -> int:
    """Admin settings store > env var > default. Non-fatal on any error."""
    raw: Optional[str] = None
    try:
        from api.config.settings import get_setting

        raw = get_setting(key)
    except Exception:  # pragma: no cover - import-safe / DB down
        raw = None
    if raw is None or not str(raw).strip():
        raw = os.environ.get(env_name)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _enforce(key: str, limit: int, window_seconds: float, scope: str) -> None:
    """Take one token from the bucket ``key`` or raise 429 with Retry-After."""
    refill_per_sec = limit / window_seconds if window_seconds > 0 else math.inf
    bucket = _get_bucket(key, float(limit), refill_per_sec)
    with _lock:
        allowed, retry_after = bucket.try_take()
    if not allowed:
        retry_after = max(1, int(math.ceil(retry_after))) if math.isfinite(retry_after) else 60
        logger.info("Rate limit hit for %s key=%r: %s/%ss", scope, key, limit, window_seconds)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please retry later.",
            headers={"Retry-After": str(retry_after)},
        )


def client_ip(request) -> str:
    """Best-effort client IP (first X-Forwarded-For hop, else transport peer)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = getattr(getattr(request, "client", None), "host", None)
    return client or "unknown"


def enforce_ip_rate_limit(request, *, setting_key: str, env_name: str, default_per_minute: int) -> None:
    """Per-IP limit for an unauthenticated endpoint (login / reset-password)."""
    limit = _setting_int(setting_key, env_name, default_per_minute)
    _enforce(f"ip:{client_ip(request)}", limit, 60.0, "ip")


def enforce_user_rate_limit(
    user_key: str,
    *,
    setting_key: str,
    env_name: str,
    default_per_minute: int,
    window_seconds: float = 60.0,
) -> None:
    """Per-user limit for an expensive operation (docgen/expert/public ask).

    ``window_seconds`` sizes the refill window: 60 (default) = per-minute
    buckets; 3600 = hourly buckets (docgen generate). The ``*_per_minute``
    naming of the settings/env keys is historical — for hourly buckets the
    value means "requests per hour".
    """
    limit = _setting_int(setting_key, env_name, default_per_minute)
    _enforce(f"user:{user_key}", limit, float(window_seconds), "user")
