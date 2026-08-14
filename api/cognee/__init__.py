"""Cognee integration layer.

This package splits the former ``api/cognee_manager.py`` (1529 LOC) by concern:

- ``_runtime`` — import-time environment/SSL setup + ``import cognee`` + shared
  globals (``cognee``, ``_COGNEE_AVAILABLE``, resolved hosts/models/timeouts).
  Imported first so its side effects (env vars, ``import cognee``) run exactly
  once before anything else in the package touches cognee.
- ``rate_limiter`` — ``CogneeRateLimiter`` + litellm/openai monkeypatches.
- ``config`` — ``init_cognee``, ``apply_cognee_runtime_config``,
  ``apply_cognee_retry_patch``.
- ``indexing`` — ``add_document``, ``cognify_dataset``,
  ``add_and_index_document``, ``add_documents_and_cognify_once``,
  ``reindex_product_knowledge_graph``, stale-data reconciliation.
- ``query`` — ``query_cognee``.

Public API (the symbols callers import as ``from api.cognee import X``
or ``api.cognee.X``) is re-exported below.
"""

# IMPORTANT: _runtime must be imported first — it sets env vars and imports
# cognee before any other submodule reads them.
from api.cognee import _runtime  # noqa: F401  (side effects)

from api.cognee._runtime import (
    _COGNEE_AVAILABLE,
    _default_cognee_data_root,
    _default_cognee_model,
    _default_cognee_provider,
    _default_cognee_system_root,
    _local_llm_host,
    _normalize_model_for_litellm,
    _resolve_cognify_timeout,
    _resolve_graph_extraction_timeout,
    cognee,
)
from api.cognee.config import (
    apply_cognee_retry_patch,
    apply_cognee_runtime_config,
    init_cognee,
)
from api.cognee.indexing import (
    _direct_delete_dataset_data,
    _empty_cognee_dataset,
    _read_repo_text_for_cognee,
    _reconcile_stale_cognee_data,
    add_and_index_document,
    add_document,
    add_documents_and_cognify_once,
    cognify_dataset,
    reindex_product_knowledge_graph,
)
from api.cognee.query import query_cognee
from api.cognee.rate_limiter import (
    CogneeRateLimiter,
    _apply_cognee_rate_limit_patches,
    _cognee_rate_limiter,
)

__all__ = [
    "cognee",
    "_COGNEE_AVAILABLE",
    "_cognee_rate_limiter",
    "CogneeRateLimiter",
    "_apply_cognee_rate_limit_patches",
    "init_cognee",
    "apply_cognee_runtime_config",
    "apply_cognee_retry_patch",
    "add_document",
    "cognify_dataset",
    "add_documents_and_cognify_once",
    "add_and_index_document",
    "query_cognee",
    "reindex_product_knowledge_graph",
    "_resolve_graph_extraction_timeout",
    "_resolve_cognify_timeout",
    "_normalize_model_for_litellm",
    "_default_cognee_provider",
    "_default_cognee_model",
    "_local_llm_host",
    "_default_cognee_data_root",
    "_default_cognee_system_root",
    "_empty_cognee_dataset",
    "_direct_delete_dataset_data",
    "_read_repo_text_for_cognee",
    "_reconcile_stale_cognee_data",
]
