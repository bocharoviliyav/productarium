"""Backwards-compatibility shim.

The cognee integration has been split into the :mod:`api.cognee` package
(``_runtime``, ``rate_limiter``, ``config``, ``indexing``, ``query``).
This module re-exports the public API so existing
``from api.cognee_manager import X`` imports keep working.
"""
from api.cognee import (  # noqa: F401
    _COGNEE_AVAILABLE,
    CogneeRateLimiter,
    _apply_cognee_rate_limit_patches,
    _cognee_rate_limiter,
    add_and_index_document,
    add_document,
    add_documents_and_cognify_once,
    apply_cognee_retry_patch,
    apply_cognee_runtime_config,
    cognify_dataset,
    cognee,
    init_cognee,
    query_cognee,
    reindex_product_knowledge_graph,
    _resolve_cognify_timeout,
    _resolve_graph_extraction_timeout,
    _normalize_model_for_litellm,
    _default_cognee_provider,
    _default_cognee_model,
    _local_llm_host,
    _default_cognee_data_root,
    _default_cognee_system_root,
    _empty_cognee_dataset,
    _direct_delete_dataset_data,
    _read_repo_text_for_cognee,
    _reconcile_stale_cognee_data,
)
