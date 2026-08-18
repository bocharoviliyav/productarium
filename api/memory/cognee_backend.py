"""Cognee memory backend adapter — wraps the legacy cognee knowledge-graph path.

Delegates to ``api.cognee`` (add_and_index_document / query_cognee /
reindex_product_knowledge_graph) so all existing timeout/retry/cognify logic
is preserved unchanged. The dataset name follows the cognee convention
``prod_{product_id}``.

This backend is kept available behind the admin ``memory.backend`` switch for
users who want the knowledge-graph recall semantics; the default is the faster
pgvector-direct backend. All methods are non-fatal.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from api.memory.base import MemoryBackend

logger = logging.getLogger(__name__)


class CogneeMemoryBackend(MemoryBackend):
    """Thin adapter over ``api.cognee`` (cognify + recall)."""

    name = "cognee"

    async def init(self) -> None:
        """Run cognee startup migrations (non-fatal, fire-and-forget caller)."""
        try:
            from api.cognee import init_cognee

            await init_cognee()
        except Exception as e:  # pragma: no cover - non-fatal
            logger.warning("cognee memory: init_cognee failed (non-fatal): %s", e)

    async def index(
        self,
        content: str,
        product_id: str,
        source_type: str = "codebase",
        source_id: Optional[str] = None,
    ) -> int:
        """Add + cognify content into the cognee dataset ``prod_{product_id}``.

        cognee's ``add_and_index_document`` returns None; we return 1 on
        success / 0 on failure as a coarse signal (cognee does not expose a
        chunk count). Non-fatal.
        """
        if not content or not content.strip() or not product_id:
            return 0
        dataset = f"prod_{product_id}"
        try:
            from api.cognee import add_and_index_document

            await add_and_index_document(content, dataset_name=dataset)
            return 1
        except Exception as e:
            logger.warning("cognee memory: index failed for %r: %s", dataset, e)
            return 0

    async def query(self, query: str, product_id: str, top_k: int = 20) -> str:
        """Recall context from the cognee dataset for the product."""
        if not query or not product_id:
            return ""
        dataset = f"prod_{product_id}"
        try:
            from api.cognee import query_cognee

            return await query_cognee(query, dataset_name=dataset, top_k=top_k)
        except Exception as e:
            logger.warning("cognee memory: query failed for %r: %s", dataset, e)
            return ""

    async def clear_product(self, product_id: str) -> bool:
        """Empty the cognee dataset for a product (best-effort)."""
        if not product_id:
            return False
        dataset = f"prod_{product_id}"
        try:
            # _empty_cognee_dataset is the per-dataset reset used by reindex.
            from api.cognee import _empty_cognee_dataset

            return await _empty_cognee_dataset(dataset)
        except Exception as e:
            logger.warning("cognee memory: clear_product(%s) failed: %s", dataset, e)
            return False

    async def reindex_product(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        """Delegate to cognee's full reindex over product artifacts + nodes."""
        try:
            from api.cognee import reindex_product_knowledge_graph

            return await reindex_product_knowledge_graph(product_id)
        except Exception as e:
            logger.error("cognee memory: reindex failed: %s", e, exc_info=True)
            return {"success": False, "message": f"Reindex error: {e}", "reindexed_count": 0}

    def status(self) -> Dict[str, Any]:
        """Report cognee availability for the admin UI (non-fatal)."""
        out: Dict[str, Any] = {"backend": self.name}
        try:
            from api.cognee import _COGNEE_AVAILABLE

            out["available"] = bool(_COGNEE_AVAILABLE)
        except Exception:
            out["available"] = False
        return out
