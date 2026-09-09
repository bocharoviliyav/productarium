"""Abstract memory backend interface.

Single implementation:
- ``PgVectorMemoryBackend`` (api.memory.pgvector_backend) — direct Postgres +
  pgvector chunk store with cosine search (no LLM graph extraction; fast).

The active backend is resolved by ``api.memory.resolver.get_memory_backend``
from the ``memory.backend`` setting (``pgvector`` only). Callers use the
facade functions in ``api.memory`` (``index_document`` / ``query_memory`` /
...) which delegate to the active backend.

All methods are async and NEVER raise on normal operation errors (DB down,
embedder unavailable, pgvector absent): they log and return a safe empty/zero
result so the expert/docgen paths degrade gracefully to their artifact-doc
fallbacks instead of crashing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MemoryBackend(ABC):
    """Pluggable agent-memory backend.

    Implementations own the full lifecycle of product-scoped knowledge:
    chunking + embedding + storage (``index``), semantic recall (``query``),
    per-product reset (``clear_product``), and a full rebuild from source
    artifacts (``reindex_product``). The ``name`` attribute identifies the
    backend in admin status responses.
    """

    name: str = "base"

    @abstractmethod
    async def index(
        self,
        content: str,
        product_id: str,
        source_type: str = "codebase",
        source_id: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> int:
        """Chunk, embed, and store ``content`` for ``product_id``.

        Returns the number of chunks indexed (0 on failure / empty content).
        Implementations should be idempotent per (product_id, source_id): a
        re-index of the same source replaces its chunks rather than appending.
        ``source_path`` is optional citation provenance (the repo-relative
        path of the source document) stored on the chunks when supported.
        """
        raise NotImplementedError

    @abstractmethod
    async def query(
        self, query: str, product_id: str, top_k: int = 20
    ) -> str:
        """Retrieve the top-k most relevant chunks for ``query`` as joined text.

        Returns "" when nothing is available or the backend is unavailable.
        """
        raise NotImplementedError

    @abstractmethod
    async def clear_product(self, product_id: str) -> bool:
        """Delete all indexed chunks for a product. Returns True on success."""
        raise NotImplementedError

    @abstractmethod
    async def reindex_product(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        """Rebuild the index from source artifacts for one or all products.

        Returns a dict with ``success``, ``message``, ``reindexed_count``
        (the reindex contract consumed by the admin UI).
        """
        raise NotImplementedError

    async def init(self) -> None:
        """Backend startup hook. Default: no-op."""
        return None

    def status(self) -> Dict[str, Any]:
        """Backend-specific status for the admin UI. Default: just the name."""
        return {"backend": self.name}
