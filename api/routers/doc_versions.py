"""Doc version router — immutable documentation version history (Vault KV-v2).

Endpoints (prefix ``/api/products``, tags ``doc-versions``):
- ``GET    /{product_id}/{segment}/{entity_id}/versions``          (list)
- ``GET    /{product_id}/{segment}/{entity_id}/versions/{version}`` (read-only view)
- ``POST   /{product_id}/{segment}/{entity_id}/versions/{version}/restore``
- ``DELETE /{product_id}/{segment}/{entity_id}/versions/{version}`` (manage)

``segment`` ∈ codebases|specs|databases. Restore writes the old snapshot back
onto the artifact AND appends a new ``rollback`` version (history is never
rewritten), then re-indexes the restored text so the vector memory always
holds only the current version. SPEC-012: page-scoped versions are deltas
(``scope="page"``) resolved through their base on read. SPEC-013: delete
removes ONE snapshot row; the current, the only and the referenced-base
versions are refused with 409, unknown with 404 — the artifact itself,
its memory chunks and ``current_version`` are never touched.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth.deps import (
    get_current_user,
    require_product_access,
    require_product_permission,
)
from api.db import get_db
from api.docgen.jobs import EntityBusyError, lock_for_entity
from api.models import ProductORM
from api.repositories import doc_version_repo, product_repo
from api.routers.products import _maybe_light, _reindex
from api.schemas import Product

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/products",
    tags=["doc-versions"],
    dependencies=[Depends(get_current_user)],
)

_SEGMENT_TO_TYPE = {
    "codebases": "codebase",
    "specs": "spec",
    "databases": "database",
    "hlds": "hld",
    # Singular alias: the UI's entityPath("hld") builds version URLs with it.
    "hld": "hld",
}


class DocVersionItem(BaseModel):
    version: int
    source: str
    model: Optional[str] = None
    job_id: Optional[str] = None
    created_at: Optional[datetime] = None
    is_current: bool = False
    scope: str = "full"  # "page" = delta (SPEC-012), "full" = complete snapshot


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
    entity = product_repo.find_entity(p_orm, entity_type, entity_id)
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
                scope="page" if r.base_version is not None else "full",
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
    light: bool = Query(False, description="Strip page bodies; load them via the page-content endpoint"),
):
    _, entity_type, entity = _load_entity(db, product_id, segment, entity_id)
    row = doc_version_repo.get_version(db, entity_type, entity_id, version)
    if row is None:
        raise HTTPException(status_code=404, detail="Version not found")
    pages, generated_docs = row.pages, row.generated_docs
    if light and isinstance(pages, dict) and pages:
        pages = product_repo.light_pages(pages)
        generated_docs = None  # redundant with the page bodies
    return DocVersionDetail(
        version=row.version,
        source=row.source,
        model=row.model,
        job_id=row.job_id,
        created_at=row.created_at,
        is_current=(row.version == entity.current_version),
        scope="page" if row.base_version is not None else "full",
        generated_docs=generated_docs,
        pages=pages,
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
    light: bool = Query(False),
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
    # SPEC-010: restored page-bearing docs re-index per page (matching
    # generation); spec content stays artifact-level.
    reindex_kwargs = {"source_type": entity_type}
    if entity_type in ("codebase", "database") and isinstance(entity.pages, dict):
        reindex_kwargs["pages"] = entity.pages
    _reindex(product_id, indexed_text, entity_id, **reindex_kwargs)
    return _maybe_light(product_repo.orm_to_product(p_orm), light)


# SPEC-013 reason → (status, client message). Only the snapshot row dies;
# the artifact, its memory chunks and current_version stay untouched.
_DELETE_REASONS = {
    "current": (409, "Cannot delete the current version"),
    "single": (409, "Cannot delete the only version — delete the artifact instead"),
    "base": (409, "Cannot delete a version another version is based on"),
}


@router.delete(
    "/{product_id}/{segment}/{entity_id}/versions/{version:int}",
    status_code=204,
)
async def delete_doc_version(
    product_id: str, segment: str, entity_id: str, version: int,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_permission("manage")),
):
    """Delete ONE snapshot row (history pruning, not artifact deletion)."""
    _, entity_type, entity = _load_entity(db, product_id, segment, entity_id)
    try:
        with lock_for_entity(entity_type, entity_id):
            reason = doc_version_repo.delete_version(db, entity_type, entity, version)
            if reason == "not_found":
                raise HTTPException(status_code=404, detail="Version not found")
            if reason:
                status, detail = _DELETE_REASONS[reason]
                raise HTTPException(status_code=status, detail=detail)
            db.commit()
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))


__all__ = ["router"]
