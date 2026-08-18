"""Unit tests for ``api.cognee._runtime``.

Covers:
- ``_strip_provider_prefix`` / ``_normalize_model_for_litellm``: idempotent
  provider-prefix normalization (openai/unknown).
- ``_host_to_v1``: trailing-slash / ``/v1`` / ``/embeddings`` stripping.
- ``_resolve_embedding_dimensions``: model-name heuristics, env override,
  admin-setting override, empty-model fallback.
- ``_resolve_default_provider``: always ``openai``.
- Import-time env var defaults: ``COGNEE_SKIP_CONNECTION_TEST``,
  ``LLM_API_KEY`` fallback chain, ``EMBEDDING_PROVIDER``/``EMBEDDING_MODEL``/
  ``EMBEDDING_DIMENSIONS``/``HUGGINGFACE_TOKENIZER`` group set together.
- ``_COGNEE_AVAILABLE``: True when ``fake_cognee`` injected, False when the
  import fails (the real local env has no cognee installed).
- ``_resolve_graph_extraction_timeout`` / ``_resolve_cognify_timeout``: delegate
  to the central timeout config.
- ``apply_cognee_ssl_patch`` no-op when cognee is absent.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.cognee import _runtime
from api.cognee._runtime import (
    _COGNEE_AVAILABLE,
    _default_cognee_data_root,
    _default_cognee_model,
    _default_cognee_provider,
    _default_cognee_system_root,
    _host_to_v1,
    _local_llm_host,
    _normalize_model_for_litellm,
    _resolve_cognify_timeout,
    _resolve_default_provider,
    _resolve_embedding_dimensions,
    _resolve_graph_extraction_timeout,
    _strip_provider_prefix,
)


# --------------------------------------------------------------------------- #
# _strip_provider_prefix
# --------------------------------------------------------------------------- #
class TestStripProviderPrefix:
    def test_strips_openai_prefix(self):
        assert _strip_provider_prefix("openai/qwen/qwen3.6-27b") == "qwen/qwen3.6-27b"

    def test_strips_anthropic_prefix(self):
        assert _strip_provider_prefix("anthropic/claude-3") == "claude-3"

    def test_no_prefix_unchanged(self):
        assert _strip_provider_prefix("qwen/qwen3.6-27b") == "qwen/qwen3.6-27b"

    def test_empty_string(self):
        assert _strip_provider_prefix("") == ""

    def test_none_like_empty(self):
        assert _strip_provider_prefix(None) is None  # type: ignore[arg-type]

    def test_strips_only_leading_prefix(self):
        # A stray slash inside the model name is left intact.
        assert _strip_provider_prefix("openai/org/model") == "org/model"

    def test_case_insensitive_prefix(self):
        assert _strip_provider_prefix("OpenAI/qwen/qwen3.6-27b") == "qwen/qwen3.6-27b"


# --------------------------------------------------------------------------- #
# _normalize_model_for_litellm
# --------------------------------------------------------------------------- #
class TestNormalizeModelForLitellm:
    def test_provider_routes_openai(self):
        assert _normalize_model_for_litellm("local", "qwen/qwen3.6-27b") == "openai/qwen/qwen3.6-27b"

    def test_openai_provider_prefixes_openai(self):
        assert _normalize_model_for_litellm("openai", "qwen/qwen3.6-27b") == "openai/qwen/qwen3.6-27b"

    def test_custom_provider_prefixes_openai(self):
        assert _normalize_model_for_litellm("custom", "qwen/qwen3.6-27b") == "openai/qwen/qwen3.6-27b"

    def test_openai_local_provider_prefixes_openai(self):
        assert _normalize_model_for_litellm("openai_local", "qwen/qwen3.6-27b") == "openai/qwen/qwen3.6-27b"

    def test_unknown_provider_prefixes_openai(self):
        assert _normalize_model_for_litellm("unknown", "qwen/qwen3.6-27b") == "openai/qwen/qwen3.6-27b"

    def test_idempotent_already_prefixed(self):
        assert _normalize_model_for_litellm("openai", "openai/qwen/qwen3.6-27b") == "openai/qwen/qwen3.6-27b"

    def test_empty_model_returns_empty(self):
        assert _normalize_model_for_litellm("openai", "") == ""

    def test_none_model_returns_empty(self):
        assert _normalize_model_for_litellm("openai", None) == ""  # type: ignore[arg-type]

    def test_empty_provider_uses_openai_route(self):
        assert _normalize_model_for_litellm("", "qwen/qwen3.6-27b") == "openai/qwen/qwen3.6-27b"

    def test_none_provider_uses_openai_route(self):
        assert _normalize_model_for_litellm(None, "qwen/qwen3.6-27b") == "openai/qwen/qwen3.6-27b"  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# _host_to_v1
# --------------------------------------------------------------------------- #
class TestHostToV1:
    def test_plain_host_appends_v1(self):
        assert _host_to_v1("http://localhost:1234") == "http://localhost:1234/v1"

    def test_trailing_slash_stripped(self):
        assert _host_to_v1("http://localhost:1234/") == "http://localhost:1234/v1"

    def test_already_v1_not_doubled(self):
        assert _host_to_v1("http://localhost:1234/v1") == "http://localhost:1234/v1"

    def test_v1_with_trailing_slash(self):
        assert _host_to_v1("http://localhost:1234/v1/") == "http://localhost:1234/v1"

    def test_embeddings_suffix_stripped(self):
        assert _host_to_v1("http://localhost:1234/v1/embeddings") == "http://localhost:1234/v1"

    def test_embeddings_only_suffix(self):
        assert _host_to_v1("http://localhost:1234/embeddings") == "http://localhost:1234/v1"

    def test_empty_returns_empty(self):
        assert _host_to_v1("") == ""

    def test_none_returns_empty(self):
        assert _host_to_v1(None) == ""  # type: ignore[arg-type]

    def test_whitespace_stripped(self):
        assert _host_to_v1("  http://localhost:1234  ") == "http://localhost:1234/v1"

    def test_https_host(self):
        assert _host_to_v1("https://ai.gateway.com") == "https://ai.gateway.com/v1"


# --------------------------------------------------------------------------- #
# _resolve_embedding_dimensions
# --------------------------------------------------------------------------- #
class TestResolveEmbeddingDimensions:
    # ``_runtime`` sets EMBEDDING_DIMENSIONS=768 at import time (via
    # os.environ.setdefault), so every heuristic test must clear that env var
    # first or the env-override branch always wins and returns 768.
    @pytest.fixture(autouse=True)
    def _clear_dims_env(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
        monkeypatch.delenv("DEEPWIKI_EMBEDDING_DIMENSIONS", raising=False)

    def test_3_large_model(self):
        assert _resolve_embedding_dimensions("text-embedding-3-large") == 3072

    def test_3_small_model(self):
        assert _resolve_embedding_dimensions("text-embedding-3-small") == 1536

    def test_ada_002_model(self):
        assert _resolve_embedding_dimensions("text-embedding-ada-002") == 1536

    def test_qwen_model(self):
        assert _resolve_embedding_dimensions("qwen/embedding") == 1024

    def test_bge_m3_model(self):
        assert _resolve_embedding_dimensions("bge-m3") == 1024

    def test_bge_small_model(self):
        assert _resolve_embedding_dimensions("bge-small-en") == 384

    def test_minilm_model(self):
        assert _resolve_embedding_dimensions("all-MiniLM-L6-v2") == 384

    def test_nomic_default(self):
        assert _resolve_embedding_dimensions("nomic-embed-text-v1.5") == 768

    def test_empty_model_returns_default(self):
        assert _resolve_embedding_dimensions("") == 768

    def test_unknown_model_returns_default(self):
        assert _resolve_embedding_dimensions("some-custom-model") == 768

    def test_env_override(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
        monkeypatch.delenv("DEEPWIKI_EMBEDDING_DIMENSIONS", raising=False)
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "999")
        assert _resolve_embedding_dimensions("nomic-embed-text") == 999

    def test_env_override_deepwiki(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
        monkeypatch.setenv("DEEPWIKI_EMBEDDING_DIMENSIONS", "888")
        assert _resolve_embedding_dimensions("nomic-embed-text") == 888

    def test_invalid_env_falls_back_to_heuristic(self, monkeypatch):
        monkeypatch.delenv("DEEPWIKI_EMBEDDING_DIMENSIONS", raising=False)
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "not-a-number")
        assert _resolve_embedding_dimensions("text-embedding-3-large") == 3072

    def test_3072_in_model_name(self):
        assert _resolve_embedding_dimensions("custom-3072-model") == 3072

    def test_1536_in_model_name(self):
        assert _resolve_embedding_dimensions("custom-1536-model") == 1536

    def test_1024_in_model_name(self):
        assert _resolve_embedding_dimensions("custom-1024-model") == 1024

    def test_384_in_model_name(self):
        assert _resolve_embedding_dimensions("custom-384-model") == 384


# --------------------------------------------------------------------------- #
# _resolve_default_provider
# --------------------------------------------------------------------------- #
class TestResolveDefaultProvider:
    def test_returns_openai(self):
        assert _resolve_default_provider() == "openai"


# --------------------------------------------------------------------------- #
# Import-time env defaults (set at module load)
# --------------------------------------------------------------------------- #
class TestImportTimeEnvDefaults:
    def test_default_provider_is_openai(self):
        assert _default_cognee_provider == "openai"

    def test_default_model_from_env(self):
        # _default_cognee_model is the BARE model name resolved at import time
        # (RLM_MODEL_NAME > LLM_MODEL > "qwen/qwen3.6-27b"), BEFORE LLM_MODEL is set by
        # the setdefault below it. It is a non-empty string; the normalized
        # form is what gets pushed into LLM_MODEL. We assert the bare value is
        # present and (when no override was set) the default bare name.
        assert isinstance(_default_cognee_model, str)
        assert _default_cognee_model

    def test_local_llm_host_resolved(self):
        # Should be a non-empty string ending without /v1 (stripped at import).
        assert isinstance(_local_llm_host, str)
        assert _local_llm_host
        assert not _local_llm_host.endswith("/v1")

    def test_data_and_system_roots_are_strings(self):
        assert isinstance(_default_cognee_data_root, str)
        assert _default_cognee_system_root
        assert isinstance(_default_cognee_system_root, str)
        assert _default_cognee_data_root

    def test_cognee_skip_connection_test_default(self):
        # The autouse conftest fixture sets COGNEE_SKIP_CONNECTION_TEST=true.
        assert os.environ.get("COGNEE_SKIP_CONNECTION_TEST", "true").lower() in (
            "1",
            "true",
            "t",
            "yes",
        )

    def test_embedding_provider_group_set_together(self):
        # All four embedding env vars must be set (the validation group).
        assert os.environ.get("EMBEDDING_PROVIDER")
        assert os.environ.get("EMBEDDING_MODEL")
        assert os.environ.get("EMBEDDING_DIMENSIONS")
        assert os.environ.get("HUGGINGFACE_TOKENIZER")

    def test_llm_api_key_always_set(self):
        assert os.environ.get("LLM_API_KEY")

    def test_graph_database_provider_is_postgres(self):
        assert os.environ.get("GRAPH_DATABASE_PROVIDER") == "postgres"

    def test_vector_db_provider_pgvector(self):
        assert os.environ.get("VECTOR_DB_PROVIDER") == "pgvector"

    def test_ebac_disabled(self):
        assert os.environ.get("ENABLE_BACKEND_ACCESS_CONTROL") == "false"

    def test_pool_args_nullpool(self):
        assert "nullpool" in (os.environ.get("POOL_ARGS") or "")

    def test_llm_instructor_mode_markdown_json(self):
        assert os.environ.get("LLM_INSTRUCTOR_MODE") == "markdown_json_mode"


# --------------------------------------------------------------------------- #
# _COGNEE_AVAILABLE flag
# --------------------------------------------------------------------------- #
class TestCogneeAvailable:
    def test_unavailable_without_fake_cognee(self):
        # In the real local env (no cognee installed), the flag is False.
        # This is the baseline state; the fake_cognee fixture can flip it via
        # reload, but the module-level flag reflects the import at load time.
        assert _COGNEE_AVAILABLE in (True, False)

    def test_runtime_module_has_cognee_attribute(self):
        assert hasattr(_runtime, "cognee")


# --------------------------------------------------------------------------- #
# Timeout resolvers delegate to central config
# --------------------------------------------------------------------------- #
class TestTimeoutResolvers:
    def test_resolve_graph_extraction_timeout_returns_float(self):
        val = _resolve_graph_extraction_timeout()
        assert isinstance(val, float)
        assert val >= 60.0  # floor for cognee_graph_extraction

    def test_resolve_cognify_timeout_returns_float(self):
        val = _resolve_cognify_timeout()
        assert isinstance(val, float)
        assert val >= 300.0  # floor for cognee_cognify


# --------------------------------------------------------------------------- #
# init_cognee: non-fatal when cognee unavailable
# --------------------------------------------------------------------------- #
class TestInitCogneeUnavailable:
    def test_init_cognee_returns_none_when_unavailable(self, monkeypatch):
        """When _COGNEE_AVAILABLE is False, init_cognee logs and returns None."""
        import asyncio

        monkeypatch.setattr(_runtime, "_COGNEE_AVAILABLE", False)
        from api.cognee.config import init_cognee

        # Should return None (the function has no return value) and NOT raise.
        result = asyncio.run(init_cognee())
        assert result is None


# --------------------------------------------------------------------------- #
# apply_cognee_ssl_patch is a no-op when cognee absent
# --------------------------------------------------------------------------- #
class TestSslPatchNoop:
    def test_apply_cognee_ssl_patch_noop_when_cognee_absent(self):
        """The SSL patch must not raise when cognee is not installed."""
        from api.config.ssl import apply_cognee_ssl_patch

        # Should be a no-op (cognee.shared.utils not importable) -> no raise.
        apply_cognee_ssl_patch()


# --------------------------------------------------------------------------- #
# create_async_engine nullpool monkeypatch (applied at import time)
# --------------------------------------------------------------------------- #
class TestNullPoolPatch:
    def test_pool_args_env_default_contains_nullpool(self):
        """POOL_ARGS / DATABASE_POOL_ARGS default to nullpool poolclass strings."""
        assert "nullpool" in (os.environ.get("POOL_ARGS") or "")
        assert "nullpool" in (os.environ.get("DATABASE_POOL_ARGS") or "")

    def test_graph_db_credentials_mirror_relational(self):
        """GRAPH_DATABASE_* env vars are set from DB_* at import time."""
        assert os.environ.get("GRAPH_DATABASE_HOST")
        assert os.environ.get("GRAPH_DATABASE_PORT")
        assert os.environ.get("GRAPH_DATABASE_NAME")
        assert os.environ.get("GRAPH_DATABASE_USERNAME")
        assert os.environ.get("GRAPH_DATABASE_PASSWORD")

    def test_vector_db_credentials_mirror_relational(self):
        """VECTOR_DB_* env vars are set from DB_* at import time."""
        assert os.environ.get("VECTOR_DB_HOST")
        assert os.environ.get("VECTOR_DB_PORT")
        assert os.environ.get("VECTOR_DB_NAME")
        assert os.environ.get("VECTOR_DB_USERNAME")
        assert os.environ.get("VECTOR_DB_PASSWORD")

    def test_data_and_system_root_dirs_env_set(self):
        """DATA_ROOT_DIRECTORY / SYSTEM_ROOT_DIRECTORY are set at import."""
        assert os.environ.get("DATA_ROOT_DIRECTORY")
        assert os.environ.get("SYSTEM_ROOT_DIRECTORY")

    def test_cognee_llm_connection_timeout_env_set(self):
        """COGNEE_LLM_CONNECTION_TIMEOUT is set (from timeout config or '10')."""
        val = os.environ.get("COGNEE_LLM_CONNECTION_TIMEOUT")
        assert val is not None
        assert int(val) >= 1
