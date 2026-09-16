"""Product / Codebase / Spec / Links CRUD router.

Endpoints (prefix ``/api/products``, tags ``products``):
- ``GET    /api/products``                                    — list products (light rows, bare
  list + ``X-Total-Count``; the full object lives at GET /{id})
- ``POST   /api/products``                                    — create product
- ``GET    /api/products/{product_id}``                       — get product
- ``PUT    /api/products/{product_id}``                       — update product
- ``DELETE /api/products/{product_id}``                       — delete product
- ``POST   /api/products/{product_id}/codebases``             — add codebase
- ``DELETE /api/products/{product_id}/codebases/{codebase_id}`` — delete codebase
- ``PUT    /api/products/{product_id}/codebases/{codebase_id}``  — update docs (WYSIWYG)
- ``POST   /api/products/{product_id}/codebases/{codebase_id}/pages/{page_id}/verify``
  — verify a single documentation page (owner/admin)
- ``POST   /api/products/{product_id}/specs``                 — add spec
- ``DELETE /api/products/{product_id}/specs/{spec_id}``       — delete spec
- ``PUT    /api/products/{product_id}/specs/{spec_id}``       — update spec content
- ``POST   /api/products/{product_id}/links``                 — add links
- ``DELETE /api/products/{product_id}/links/{links_id}``      — delete links
- ``PUT    /api/products/{product_id}/links/{links_id}``      — update links content

Thin layer: request parsing + memory re-index handoff; all DB access lives in
``api.repositories.product_repo``.

Authorization (P0-2): reads require visible access (owner / grant /
viewer_global / manager / admin); writes require 'rw' (owner / rw-grant /
manager / admin). Product creation is restricted to admin|manager; the creator
becomes the owner when the payload has none.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Set

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db import get_db
from api.docgen.jobs import EntityBusyError
from api.models import ProductGrantORM, ProductORM, UserORM
from api.repositories import product_repo
from api.schemas import Codebase, Links, Product, ProductListItem, Spec
from api.auth.deps import get_current_user, require_product_access
from api.utils.repo_url import validate_repo_url

logger = logging.getLogger(__name__)

# All product CRUD endpoints require an authenticated user (a no-op when
# AUTH_PROVIDER=none). Object-level checks (owner/admin) apply on top where
# they matter — e.g. the ``verify`` endpoints below.
router = APIRouter(
    prefix="/api/products",
    tags=["products"],
    dependencies=[Depends(get_current_user)],
)


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


@router.get("", response_model=list[ProductListItem])
async def list_products(
    response: Response,
    db: Session = Depends(get_db),
    user: UserORM = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Light product listing (P1-16) in the baseline bare-list shape.

    Rows carry SQL-counted child totals only — no child entities, no
    generated_docs/pages/content (the full object is served by GET /{id}).
    Pagination via ``limit``/``offset`` query params; the visibility-
    filtered total rides in the ``X-Total-Count`` header so the body stays
    a plain JSON array (backward-compatible contract — no envelope).
    """
    visible = _visible_product_ids(db, user)
    items, total = product_repo.list_products_light(
        db, product_ids=visible, limit=limit, offset=offset
    )
    response.headers["X-Total-Count"] = str(total)
    return items


def _visible_product_ids(db: Session, user: UserORM) -> Optional[Set[str]]:
    """None = no restriction; otherwise the set of product ids the user sees."""
    if user.role in ("admin", "manager", "viewer_global"):
        return None
    owned = {
        row[0]
        for row in db.query(ProductORM.id)
        .filter(ProductORM.owner_id == user.id)
        .all()
    }
    granted = {
        row[0]
        for row in db.query(ProductGrantORM.product_id)
        .filter(ProductGrantORM.user_id == user.id)
        .all()
    }
    return owned | granted


def _guard_no_raw_dsn(result: Product, request: Product) -> None:
    """Refuse to return a product payload containing a RAW database DSN.

    Same semantics as the databases router's ``_assert_no_raw_dsn`` (kept
    local per the repo's router-isolation convention): the full-product
    POST/PUT accepts ``databases[].dsn`` and ``_database_orm_from_pydantic``
    masks it on persistence — this guard is the defense-in-depth check on
    the RESPONSE (review #4 HIGH). Password-less DSNs legitimately mask to
    themselves and never trip the guard.
    """
    from api.docgen.verification import mask_dsn

    raw_dsns = [
        (d.dsn or "").strip() for d in (request.databases or []) if (d.dsn or "").strip()
    ]
    if not raw_dsns:
        return
    leaking = [raw for raw in raw_dsns if mask_dsn(raw) != raw]
    if not leaking:
        return
    try:
        payload = repr(result.model_dump())
    except Exception:  # pragma: no cover - defensive
        return
    for raw in leaking:
        if raw in payload:
            logger.error(
                "products router: raw DSN leaked into the product payload; "
                "refusing to return it"
            )
            raise HTTPException(status_code=500, detail="Internal masking error")


@router.post("", response_model=Product)
async def create_product(
    product: Product,
    db: Session = Depends(get_db),
    user: UserORM = Depends(get_current_user),
):
    """Create a product (admin|manager only, P0-2).

    When the payload carries no ``owner_id`` the creator becomes the owner.
    """
    if user.role not in ("admin", "manager"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins and managers can create products",
        )
    if not product.owner_id:
        product = product.model_copy(update={"owner_id": user.id})
    p_orm = product_repo.upsert_product(db, product)
    result = product_repo.orm_to_product(p_orm)
    _guard_no_raw_dsn(result, product)
    return result


@router.get("/{product_id}", response_model=Product)
async def get_product(
    product_id: str,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("ro")),
):
    p_orm = product_repo.load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product_repo.orm_to_product(p_orm)


@router.put("/{product_id}", response_model=Product)
async def update_product(
    product_id: str,
    product: Product,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
):
    # Preserve previous overwrite semantics: the body Product is saved as-is.
    # (Server-owned verified flags and stored tokens survive the replace —
    # see product_repo.upsert_product.)
    # P1-18: the path product_id wins over any body id, so a stale/mismatched
    # body can never spawn a second product.
    if product.id != product_id:
        product = product.model_copy(update={"id": product_id})
    try:
        p_orm = product_repo.upsert_product(db, product)
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    result = product_repo.orm_to_product(p_orm)
    _guard_no_raw_dsn(result, product)
    return result


@router.delete("/{product_id}")
async def delete_product(
    product_id: str,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
):
    product_repo.delete_product(db, product_id)
    return {"message": "Product deleted successfully"}


# --- Codebases --------------------------------------------------------------
def _reindex(
    product_id: str,
    indexed_text: Optional[str],
    entity_id: str,
    *,
    source_type: str = "codebase",
) -> None:
    """Re-index edited text into the per-product memory backend (fire-and-forget).

    ``source_id`` = entity id so the pgvector upsert deletes the previous chunks
    for that entity before re-inserting.
    """
    if indexed_text and indexed_text.strip():
        try:
            from api.docgen import _index_in_background
            _index_in_background(
                indexed_text, f"prod_{product_id}",
                source_type=source_type, source_id=entity_id,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Memory re-index failed for entity %s: %s", entity_id, e)


@router.post("/{product_id}/codebases", response_model=Product)
async def add_codebase(
    product_id: str,
    codebase: Codebase,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
):
    # P0-1: reject dangerous clone sources at the CRUD boundary, before any
    # git invocation (ext::, file://, ssh://, arbitrary local paths, ...).
    if codebase.repo_url:
        try:
            validate_repo_url(codebase.repo_url)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    try:
        return product_repo.add_codebase(db, product_id, codebase)
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.delete("/{product_id}/codebases/{codebase_id}", response_model=Product)
async def delete_codebase(
    product_id: str,
    codebase_id: str,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
):
    try:
        return product_repo.delete_codebase(db, product_id, codebase_id)
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.put("/{product_id}/codebases/{codebase_id}", response_model=Product)
async def update_codebase_docs(
    product_id: str,
    codebase_id: str,
    body: CodebaseDocUpdate,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
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
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        msg = str(e)
        status = 400 if "Provide one of" in msg else 404
        raise HTTPException(status_code=status, detail=msg)
    _reindex(product_id, indexed_text, codebase_id, source_type="codebase")
    return product


# --- Specs ------------------------------------------------------------------
@router.post("/{product_id}/specs", response_model=Product)
async def add_spec(
    product_id: str,
    spec: Spec,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
):
    try:
        return product_repo.add_spec(db, product_id, spec)
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.delete("/{product_id}/specs/{spec_id}", response_model=Product)
async def delete_spec(
    product_id: str,
    spec_id: str,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
):
    try:
        return product_repo.delete_spec(db, product_id, spec_id)
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.put("/{product_id}/specs/{spec_id}", response_model=Product)
async def update_spec(
    product_id: str,
    spec_id: str,
    body: ContentUpdate,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
):
    """Replace a spec's raw content (authored directly, no generation)."""
    try:
        product, indexed_text = product_repo.update_spec_content(
            db, product_id, spec_id, body.content
        )
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Product not found")
    _reindex(product_id, indexed_text, spec_id, source_type="spec")
    return product


# --- Links ------------------------------------------------------------------
@router.post("/{product_id}/links", response_model=Product)
async def add_links(
    product_id: str,
    links: Links,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
):
    try:
        return product_repo.add_links(db, product_id, links)
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.delete("/{product_id}/links/{links_id}", response_model=Product)
async def delete_links(
    product_id: str,
    links_id: str,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
):
    try:
        return product_repo.delete_links(db, product_id, links_id)
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.put("/{product_id}/links/{links_id}", response_model=Product)
async def update_links(
    product_id: str,
    links_id: str,
    body: ContentUpdate,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
):
    """Replace a links collection's raw content."""
    try:
        product, indexed_text = product_repo.update_links_content(
            db, product_id, links_id, body.content
        )
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Product not found")
    _reindex(product_id, indexed_text, links_id)
    return product


# --- Verification (item 5) — owner or admin -------------------------------
def _verify_entity(
    db: Session,
    product_id: str,
    entity_id: str,
    collection: str,
    user: UserORM,
    page_id: Optional[str] = None,
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
        if page_id is None:
            return product_repo.verify_child(
                db, product_id, entity_id, collection, user.id
            )
        return product_repo.verify_page(
            db, product_id, entity_id, collection, page_id, user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e) or "Entity not found")


@router.post("/{product_id}/codebases/{codebase_id}/verify", response_model=Product)
async def verify_codebase(
    product_id: str, codebase_id: str,
    db: Session = Depends(get_db), user: UserORM = Depends(get_current_user),
):
    return _verify_entity(db, product_id, codebase_id, "codebases", user)


@router.post(
    "/{product_id}/codebases/{codebase_id}/pages/{page_id}/verify",
    response_model=Product,
)
async def verify_codebase_page(
    product_id: str, codebase_id: str, page_id: str,
    db: Session = Depends(get_db), user: UserORM = Depends(get_current_user),
):
    """Verify a single documentation page (not the whole codebase artifact)."""
    return _verify_entity(
        db, product_id, codebase_id, "codebases", user, page_id=page_id
    )


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
