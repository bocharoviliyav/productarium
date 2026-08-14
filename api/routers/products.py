"""Product / Codebase / Spec / Links CRUD router.

Endpoints (prefix ``/api/products``, tags ``products``):
- ``GET    /api/products``                                    — list products
- ``POST   /api/products``                                    — create product
- ``GET    /api/products/{product_id}``                       — get product
- ``PUT    /api/products/{product_id}``                       — update product
- ``DELETE /api/products/{product_id}``                       — delete product
- ``POST   /api/products/{product_id}/codebases``             — add codebase
- ``DELETE /api/products/{product_id}/codebases/{codebase_id}`` — delete codebase
- ``PUT    /api/products/{product_id}/codebases/{codebase_id}``  — update docs (WYSIWYG)
- ``POST   /api/products/{product_id}/specs``                 — add spec
- ``DELETE /api/products/{product_id}/specs/{spec_id}``       — delete spec
- ``PUT    /api/products/{product_id}/specs/{spec_id}``       — update spec content
- ``POST   /api/products/{product_id}/links``                 — add links
- ``DELETE /api/products/{product_id}/links/{links_id}``      — delete links
- ``PUT    /api/products/{product_id}/links/{links_id}``      — update links content

Thin layer: request parsing + cognee re-index handoff; all DB access lives in
``api.repositories.product_repo``.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db import get_db
from api.repositories import product_repo
from api.schemas import Codebase, Links, Product, Spec
from api.auth.deps import get_current_user
from api.models import UserORM

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/products", tags=["products"])


class CodebaseDocUpdate(BaseModel):
    """Partial update of a codebase's documentation (WYSIWYG saves).

    Exactly one of the doc shapes should be provided:
      - ``pages``               → replace the whole pages dict wholesale
      - ``page_id`` + ``content`` → upsert a single page's content field
      - ``generated_docs``      → replace the top-level generated_docs blob
    """
    generated_docs: Optional[str] = None
    page_id: Optional[str] = None
    content: Optional[str] = None
    pages: Optional[Dict[str, Any]] = None


class ContentUpdate(BaseModel):
    """Replace the raw ``content`` of a spec or links entity (authored directly)."""
    content: Optional[str] = None


@router.get("", response_model=list[Product])
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
    return {"message": "Product deleted successfully"}


# --- Codebases --------------------------------------------------------------
def _reindex(product_id: str, indexed_text: Optional[str], entity_id: str) -> None:
    """Re-index edited text into the per-product cognee dataset (fire-and-forget)."""
    if indexed_text and indexed_text.strip():
        try:
            from api.docgen import _index_in_background
            _index_in_background(indexed_text, f"prod_{product_id}")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Cognee re-index failed for entity %s: %s", entity_id, e)


@router.post("/{product_id}/codebases", response_model=Product)
async def add_codebase(product_id: str, codebase: Codebase, db: Session = Depends(get_db)):
    try:
        return product_repo.add_codebase(db, product_id, codebase)
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.delete("/{product_id}/codebases/{codebase_id}", response_model=Product)
async def delete_codebase(product_id: str, codebase_id: str, db: Session = Depends(get_db)):
    try:
        return product_repo.delete_codebase(db, product_id, codebase_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.put("/{product_id}/codebases/{codebase_id}", response_model=Product)
async def update_codebase_docs(
    product_id: str,
    codebase_id: str,
    body: CodebaseDocUpdate,
    db: Session = Depends(get_db),
):
    """Edit a codebase's generated documentation (WYSIWYG editor saves)."""
    try:
        product, indexed_text = product_repo.update_codebase_content(
            db,
            product_id,
            codebase_id,
            pages=body.pages,
            page_id=body.page_id,
            content=body.content,
            generated_docs=body.generated_docs,
        )
    except ValueError as e:
        msg = str(e)
        status = 400 if "Provide one of" in msg else 404
        raise HTTPException(status_code=status, detail=msg)
    _reindex(product_id, indexed_text, codebase_id)
    return product


# --- Specs ------------------------------------------------------------------
@router.post("/{product_id}/specs", response_model=Product)
async def add_spec(product_id: str, spec: Spec, db: Session = Depends(get_db)):
    try:
        return product_repo.add_spec(db, product_id, spec)
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.delete("/{product_id}/specs/{spec_id}", response_model=Product)
async def delete_spec(product_id: str, spec_id: str, db: Session = Depends(get_db)):
    try:
        return product_repo.delete_spec(db, product_id, spec_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.put("/{product_id}/specs/{spec_id}", response_model=Product)
async def update_spec(
    product_id: str,
    spec_id: str,
    body: ContentUpdate,
    db: Session = Depends(get_db),
):
    """Replace a spec's raw content (authored directly, no generation)."""
    try:
        product, indexed_text = product_repo.update_spec_content(
            db, product_id, spec_id, body.content
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")
    _reindex(product_id, indexed_text, spec_id)
    return product


# --- Links ------------------------------------------------------------------
@router.post("/{product_id}/links", response_model=Product)
async def add_links(product_id: str, links: Links, db: Session = Depends(get_db)):
    try:
        return product_repo.add_links(db, product_id, links)
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.delete("/{product_id}/links/{links_id}", response_model=Product)
async def delete_links(product_id: str, links_id: str, db: Session = Depends(get_db)):
    try:
        return product_repo.delete_links(db, product_id, links_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.put("/{product_id}/links/{links_id}", response_model=Product)
async def update_links(
    product_id: str,
    links_id: str,
    body: ContentUpdate,
    db: Session = Depends(get_db),
):
    """Replace a links collection's raw content."""
    try:
        product, indexed_text = product_repo.update_links_content(
            db, product_id, links_id, body.content
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Product not found")
    _reindex(product_id, indexed_text, links_id)
    return product


# --- Verification (item 5) — owner or admin -------------------------------
def _verify_entity(
    db: Session, product_id: str, entity_id: str, collection: str, user: UserORM
) -> Product:
    product = product_repo.load_product_orm(db, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    # Only the product owner or an admin may verify an entity.
    if user.role != "admin" and (not product.owner_id or product.owner_id != user.id):
        raise HTTPException(
            status_code=403,
            detail="Only the product owner or an admin can verify",
        )
    try:
        return product_repo.verify_child(db, product_id, entity_id, collection, user.id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Entity not found")


@router.post("/{product_id}/codebases/{codebase_id}/verify", response_model=Product)
async def verify_codebase(
    product_id: str, codebase_id: str,
    db: Session = Depends(get_db), user: UserORM = Depends(get_current_user),
):
    return _verify_entity(db, product_id, codebase_id, "codebases", user)


@router.post("/{product_id}/specs/{spec_id}/verify", response_model=Product)
async def verify_spec(
    product_id: str, spec_id: str,
    db: Session = Depends(get_db), user: UserORM = Depends(get_current_user),
):
    return _verify_entity(db, product_id, spec_id, "specs", user)


@router.post("/{product_id}/links/{links_id}/verify", response_model=Product)
async def verify_links(
    product_id: str, links_id: str,
    db: Session = Depends(get_db), user: UserORM = Depends(get_current_user),
):
    return _verify_entity(db, product_id, links_id, "links", user)


__all__ = ["router"]
