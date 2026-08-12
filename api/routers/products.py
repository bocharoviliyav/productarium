"""Product/Artifact CRUD router.

Endpoints (prefix ``/api/products``, tags ``products``):
- ``GET    /api/products``                          — list products
- ``POST   /api/products``                          — create product
- ``GET    /api/products/{product_id}``             — get product
- ``PUT    /api/products/{product_id}``             — update product
- ``DELETE /api/products/{product_id}``             — delete product
- ``POST   /api/products/{product_id}/artifacts``   — add artifact
- ``DELETE /api/products/{product_id}/artifacts/{artifact_id}`` — delete artifact
- ``PUT    /api/products/{product_id}/artifacts/{artifact_id}``  — update docs (WYSIWYG)

Thin layer: request parsing + cognee re-index handoff; all DB access lives in
``api.repositories.product_repo``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db import get_db
from api.repositories import product_repo
from api.schemas import Artifact, Product

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/products", tags=["products"])


class ArtifactDocUpdate(BaseModel):
    """Partial update of an artifact's documentation (WYSIWYG saves).

    Exactly one of the doc shapes should be provided:
      - ``pages``               → replace the whole pages dict wholesale
      - ``page_id`` + ``content`` → upsert a single page's content field
      - ``generated_docs``      → replace the top-level generated_docs blob
      - ``raw_content``         → replace the artifact's raw ``content``
                                  (spec/links authored directly, no generation)
    """
    generated_docs: Optional[str] = None
    page_id: Optional[str] = None
    content: Optional[str] = None
    pages: Optional[Dict[str, Any]] = None
    raw_content: Optional[str] = None


@router.get("", response_model=List[Product])
async def list_products(db: Session = Depends(get_db)):
    return product_repo.list_products(db)


@router.post("", response_model=Product)
async def create_product(product: Product, db: Session = Depends(get_db)):
    p_orm = product_repo.upsert_product(db, product)
    return product_repo.orm_to_product(p_orm)


@router.get("/{product_id}", response_model=Product)
async def get_product(product_id: str, db: Session = Depends(get_db)):
    p_orm = product_repo.load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product_repo.orm_to_product(p_orm)


@router.put("/{product_id}", response_model=Product)
async def update_product(product_id: str, product: Product, db: Session = Depends(get_db)):
    # Preserve previous overwrite semantics: the body Product is saved as-is.
    p_orm = product_repo.upsert_product(db, product)
    return product_repo.orm_to_product(p_orm)


@router.delete("/{product_id}")
async def delete_product(product_id: str, db: Session = Depends(get_db)):
    product_repo.delete_product(db, product_id)
    # Match previous behavior: always return success, even if missing.
    return {"message": "Product deleted successfully"}


@router.post("/{product_id}/artifacts", response_model=Product)
async def add_artifact(product_id: str, artifact: Artifact, db: Session = Depends(get_db)):
    try:
        return product_repo.add_artifact(db, product_id, artifact)
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.delete("/{product_id}/artifacts/{artifact_id}", response_model=Product)
async def delete_artifact(product_id: str, artifact_id: str, db: Session = Depends(get_db)):
    try:
        return product_repo.delete_artifact(db, product_id, artifact_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.put("/{product_id}/artifacts/{artifact_id}", response_model=Product)
async def update_artifact_docs(
    product_id: str,
    artifact_id: str,
    body: ArtifactDocUpdate,
    db: Session = Depends(get_db),
):
    """Edit an artifact's generated documentation (WYSIWYG editor saves).

    Supports three save shapes (see ``ArtifactDocUpdate``): whole ``pages``
    replace, single ``page_id``+``content`` upsert, or ``generated_docs`` blob
    replace. The edited text is re-indexed into the product's cognee dataset in
    the background (fire-and-forget, non-fatal) so expert Ask/summary stay in
    sync with user edits. Returns the refreshed product so the frontend can
    update its local state.
    """
    try:
        product, indexed_text = product_repo.update_artifact_content(
            db,
            product_id,
            artifact_id,
            pages=body.pages,
            page_id=body.page_id,
            content=body.content,
            generated_docs=body.generated_docs,
            raw_content=body.raw_content,
        )
    except ValueError as e:
        msg = str(e)
        status = 400 if "Provide one of" in msg else 404
        raise HTTPException(status_code=status, detail=msg)

    # Re-index the edited text into the per-product cognee dataset so the
    # expert agent / Ask recall user edits. Fire-and-forget; never fatal.
    if indexed_text and indexed_text.strip():
        try:
            from api.docgen import _index_in_background
            _index_in_background(indexed_text, f"prod_{product_id}")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Cognee re-index failed for artifact %s: %s", artifact_id, e)

    return product


__all__ = ["router"]
