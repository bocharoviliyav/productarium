"""Doc version router — immutable documentation version history (Vault KV-v2).

Endpoints (prefix ``/api/products``, tags ``doc-versions``):
- ``GET  /{product_id}/{segment}/{entity_id}/versions``          (list)
- ``GET  /{product_id}/{segment}/{entity_id}/versions/{version}`` (read-only view)
- ``POST /{product_id}/{segment}/{entity_id}/versions/{version}/restore``

``segment`` ∈ codebases|specs|databases. Restore writes the old snapshot back
onto the artifact AND appends a new ``rollback`` version (history is never
rewritten), then re-indexes the restored text so the vector memory always
holds only the current version.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth.deps import get_current_user, require_product_access
from api.db import get_db
from api.docgen.jobs import EntityBusyError, lock_for_entity
from api.models import ProductORM
from api.repositories import doc_version_repo, product_repo
from api.routers.products import _reindex
from api.schemas import Product

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/products",
    tags=["doc-versions"],
    dependencies=[Depends(get_current_user)],
)

_SEGMENT_TO_TYPE = {"codebases": "codebase", "specs": "spec", "databases": "database"}


class DocVersionItem(BaseModel):
    version: int
    source: str
    model: Optional[str] = None
    job_id: Optional[str] = None
    created_at: Optional[datetime] = None
    is_current: bool = False


class DocVersionList(BaseModel):
    entity_type: str
    entity_id: str
    current_version: Optional[int] = None
    versions: List[DocVersionItem]


class DocVersionDetail(DocVersionItem):
    generated_docs: Optional[str] = None
    pages: Optional[Dict[str, Any]] = None
    content: Optional[str] = None


def _load_entity(db: Session, product_id: str, segment: str, entity_id: str):
    entity_type = _SEGMENT_TO_TYPE.get(segment)
    if entity_type is None:
        raise HTTPException(status_code=404, detail="Unknown artifact type")
    p_orm = product_repo.load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")
    collection = {
        "codebase": p_orm.codebases,
        "spec": p_orm.specs,
        "database": p_orm.databases,
    }[entity_type]
    entity = next((e for e in collection if e.id == entity_id), None)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"{entity_type.capitalize()} not found")
    return p_orm, entity_type, entity


@router.get(
    "/{product_id}/{segment}/{entity_id}/versions", response_model=DocVersionList
)
async def list_doc_versions(
    product_id: str, segment: str, entity_id: str,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("ro")),
):
    _, entity_type, entity = _load_entity(db, product_id, segment, entity_id)
    current = entity.current_version
    rows = doc_version_repo.list_versions(db, entity_type, entity_id)
    return DocVersionList(
        entity_type=entity_type,
        entity_id=entity_id,
        current_version=current,
        versions=[
            DocVersionItem(
                version=r.version,
                source=r.source,
                model=r.model,
                job_id=r.job_id,
                created_at=r.created_at,
                is_current=(r.version == current),
            )
            for r in rows
        ],
    )


@router.get(
    "/{product_id}/{segment}/{entity_id}/versions/{version:int}",
    response_model=DocVersionDetail,
)
async def get_doc_version(
    product_id: str, segment: str, entity_id: str, version: int,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("ro")),
):
    _, entity_type, entity = _load_entity(db, product_id, segment, entity_id)
    row = doc_version_repo.get_version(db, entity_type, entity_id, version)
    if row is None:
        raise HTTPException(status_code=404, detail="Version not found")
    return DocVersionDetail(
        version=row.version,
        source=row.source,
        model=row.model,
        job_id=row.job_id,
        created_at=row.created_at,
        is_current=(row.version == entity.current_version),
        generated_docs=row.generated_docs,
        pages=row.pages,
        content=row.content,
    )


@router.post(
    "/{product_id}/{segment}/{entity_id}/versions/{version:int}/restore",
    response_model=Product,
)
async def restore_doc_version(
    product_id: str, segment: str, entity_id: str, version: int,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
):
    """Roll the artifact back to ``version`` (appends a new rollback version)."""
    p_orm, entity_type, entity = _load_entity(db, product_id, segment, entity_id)
    indexed_text: Optional[str] = None
    try:
        with lock_for_entity(entity_type, entity_id):
            indexed_text = doc_version_repo.restore_version(
                db, entity_type, entity, version
            )
            if indexed_text is None:
                raise HTTPException(status_code=404, detail="Version not found")
            db.commit()
            db.refresh(p_orm)
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    _reindex(product_id, indexed_text, entity_id, source_type=entity_type)
    return product_repo.orm_to_product(p_orm)


__all__ = ["router"]
