"""Agent memory facade — backend-agnostic API over pluggable memory backends.

Public functions delegate to the active backend resolved from the
``memory.backend`` admin setting (``pgvector`` default, ``cognee`` alt):
- ``index_document`` — chunk + embed + store content for a product.
- ``query_memory`` — semantic recall of top-k chunks as joined text.
- ``clear_memory`` — drop all chunks for a product.
- ``reindex_product_memory`` — rebuild from source artifacts for one/all products.
- ``get_memory_backend`` / ``get_memory_backend_name`` — backend introspection.
- ``init_memory`` — startup hook for the active backend (cognee migrations etc.).
- ``reset_memory_backend_cache`` — invalidate the cached backend after a switch.

All functions are async (except the introspection ones) and non-fatal: they
log and return a safe empty/zero result on any backend error so the expert /
docgen paths degrade to their artifact-doc fallbacks instead of crashing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from api.memory.base import MemoryBackend
from api.memory.resolver import (
    get_memory_backend,
    get_memory_backend_name,
    reset_memory_backend_cache,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryBackend",
    "get_memory_backend",
    "get_memory_backend_name",
    "reset_memory_backend_cache",
    "index_document",
    "query_memory",
    "clear_memory",
    "reindex_product_memory",
    "init_memory",
]


async def index_document(
    content: str,
    product_id: str,
    source_type: str = "codebase",
    source_id: Optional[str] = None,
) -> int:
    """Index content for a product via the active backend. Non-fatal."""
    try:
        backend = get_memory_backend()
        return await backend.index(content, product_id, source_type=source_type, source_id=source_id)
    except Exception as e:
        logger.warning("memory.index_document failed for product %s: %s", product_id, e)
        return 0


async def query_memory(query: str, product_id: str, top_k: int = 20) -> str:
    """Recall context for a product via the active backend. Non-fatal."""
    try:
        backend = get_memory_backend()
        return await backend.query(query, product_id, top_k=top_k)
    except Exception as e:
        logger.warning("memory.query_memory failed for product %s: %s", product_id, e)
        return ""


async def clear_memory(product_id: str) -> bool:
    """Clear all indexed chunks for a product. Non-fatal."""
    try:
        backend = get_memory_backend()
        return await backend.clear_product(product_id)
    except Exception as e:
        logger.warning("memory.clear_memory failed for product %s: %s", product_id, e)
        return False


async def reindex_product_memory(product_id: Optional[str] = None) -> Dict[str, Any]:
    """Rebuild the index from source artifacts. Non-fatal."""
    try:
        backend = get_memory_backend()
        return await backend.reindex_product(product_id)
    except Exception as e:
        logger.error("memory.reindex_product_memory failed: %s", e, exc_info=True)
        return {"success": False, "message": f"Reindex error: {e}", "reindexed_count": 0}


async def init_memory() -> None:
    """Startup hook for the active backend (e.g. cognee migrations).

    Called from ``api.api.startup_event`` after ``init_db``. For the pgvector
    backend this is a no-op (the extension + HNSW index are created in
    ``init_db``); for cognee it runs the (timeout-capped, non-fatal)
    ``init_cognee``. Non-fatal.
    """
    try:
        backend = get_memory_backend()
        await backend.init()
    except Exception as e:  # pragma: no cover - non-fatal
        logger.warning("memory.init_memory failed (non-fatal): %s", e)
