"""Repository layer — DB access by data domain.

Each module here owns the ORM<->Pydantic mapping and persistence helpers
for one domain. Routers depend on these instead of touching SQLAlchemy
sessions inline, keeping DB logic in one place.

Members:
- ``product_repo`` — Product/Artifact ORM<->Pydantic mapping + persistence.
- ``documents``    — Document reading + FAISS indexing pipeline + DatabaseManager
  (clone via ``api.clients.git.download_repo`` -> read -> transform -> persist).
"""

from api.repositories.documents import (
    DatabaseManager,
    count_tokens,
    read_all_documents,
    prepare_data_pipeline,
    transform_documents_and_save_to_db,
    MAX_EMBEDDING_TOKENS,
)

__all__ = [
    "DatabaseManager",
    "count_tokens",
    "read_all_documents",
    "prepare_data_pipeline",
    "transform_documents_and_save_to_db",
    "MAX_EMBEDDING_TOKENS",
]
