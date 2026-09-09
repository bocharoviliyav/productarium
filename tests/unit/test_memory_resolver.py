"""Unit tests for ``api.memory.resolver``.

Covers:
- ``get_memory_backend_name``: default (pgvector), stale ``cognee`` value from
  a pre-migration install → default, invalid→default, case-insensitive
  normalization, get_setting failure→default.
- ``get_memory_backend``: instance caching (same name → same instance), fresh
  instance after cache reset.
- ``_build_backend``: pgvector is the sole backend; any other name falls back
  to pgvector instead of raising.
- ``reset_memory_backend_cache``: drops the cached instance.

The resolver reads ``memory.backend`` from the settings store (admin > env >
default). Tests use the ``isolated_db`` fixture so the settings store is a real
SQLite table; no pgvector/Postgres dependency is required (PgVectorMemoryBackend
construction does not connect — connections happen per method).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


# --------------------------------------------------------------------------- #
# get_memory_backend_name
# --------------------------------------------------------------------------- #
class TestGetMemoryBackendName:
    def test_default_is_pgvector(self, isolated_db):
        from api.memory.resolver import get_memory_backend_name

        # No memory.backend setting stored → default.
        assert get_memory_backend_name() == "pgvector"

    def test_stale_cognee_setting_falls_back_to_default(self, isolated_db):
        # A leftover ``cognee`` value (pre-LangChain-migration install) must
        # not brick the memory path: the resolver falls back to pgvector.
        from api.config.settings import set_setting
        from api.memory.resolver import get_memory_backend_name

        set_setting("memory.backend", "cognee")
        assert get_memory_backend_name() == "pgvector"

    def test_invalid_setting_falls_back_to_default(self, isolated_db):
        from api.config.settings import set_setting
        from api.memory.resolver import get_memory_backend_name

        set_setting("memory.backend", "bogus")
        assert get_memory_backend_name() == "pgvector"

    def test_empty_setting_falls_back_to_default(self, isolated_db):
        from api.config.settings import set_setting
        from api.memory.resolver import get_memory_backend_name

        set_setting("memory.backend", "")
        assert get_memory_backend_name() == "pgvector"

    def test_case_insensitive_and_stripped(self, isolated_db):
        from api.config.settings import set_setting
        from api.memory.resolver import get_memory_backend_name

        set_setting("memory.backend", "  PGVector  ")
        assert get_memory_backend_name() == "pgvector"

    def test_get_setting_failure_falls_back_to_default(self, monkeypatch):
        # Force get_setting to raise — the resolver must not propagate.
        import api.config.settings as ss
        monkeypatch.setattr(ss, "get_setting", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))

        # Re-import the resolver name lookup by clearing the cached module so
        # the `from api.config.settings import get_setting` inside the function
        # resolves to the patched attribute. The function imports lazily per
        # call, so patching the module attribute is enough.
        from api.memory.resolver import get_memory_backend_name

        assert get_memory_backend_name() == "pgvector"


# --------------------------------------------------------------------------- #
# get_memory_backend (caching)
# --------------------------------------------------------------------------- #
class TestGetMemoryBackendCaching:
    def test_same_name_returns_same_instance(self, isolated_db):
        from api.memory.resolver import get_memory_backend, reset_memory_backend_cache

        reset_memory_backend_cache()
        b1 = get_memory_backend()
        b2 = get_memory_backend()
        assert b1 is b2
        assert b1.name == "pgvector"

    def test_stale_cognee_setting_still_builds_pgvector_after_reset(self, isolated_db):
        from api.config.settings import set_setting
        from api.memory.resolver import (
            get_memory_backend,
            reset_memory_backend_cache,
        )

        reset_memory_backend_cache()
        pg = get_memory_backend()
        assert pg.name == "pgvector"

        # A stale ``cognee`` value still resolves to pgvector; the cache is
        # keyed by the resolved name, so the same instance comes back.
        set_setting("memory.backend", "cognee")
        reset_memory_backend_cache()
        again = get_memory_backend()
        assert again.name == "pgvector"

    def test_reset_without_switch_keeps_type(self, isolated_db):
        from api.memory.resolver import (
            get_memory_backend,
            reset_memory_backend_cache,
        )

        reset_memory_backend_cache()
        b1 = get_memory_backend()
        reset_memory_backend_cache()
        b2 = get_memory_backend()
        # Same name → same type, but a fresh instance after reset.
        assert type(b1) is type(b2)
        assert b1 is not b2


# --------------------------------------------------------------------------- #
# _build_backend (pgvector is the sole backend)
# --------------------------------------------------------------------------- #
class TestBuildBackendFallback:
    def test_unknown_name_falls_back_to_pgvector(self):
        from api.memory.resolver import _build_backend

        # Unknown/stale names must not raise — they build pgvector.
        for name in ("cognee", "bogus", ""):
            assert _build_backend(name).name == "pgvector"

    def test_build_pgvector_directly(self):
        from api.memory.resolver import _build_backend

        result = _build_backend("pgvector")
        assert result.name == "pgvector"


# --------------------------------------------------------------------------- #
# reset_memory_backend_cache
# --------------------------------------------------------------------------- #
class TestResetCache:
    def test_reset_clears_cache(self, isolated_db):
        from api.memory import resolver

        resolver.reset_memory_backend_cache()
        # Populate the cache.
        b1 = resolver.get_memory_backend()
        assert resolver._cache["backend"] is b1
        resolver.reset_memory_backend_cache()
        assert resolver._cache["backend"] is None
        assert resolver._cache["name"] is None
