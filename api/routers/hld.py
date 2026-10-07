"""HLD router — high-level design generation, status and verification.

Endpoints (prefix ``/api/products``, tags ``hld``):
- ``POST /{product_id}/hld/generate``          (rw; auto-creates the row, 202)
- ``GET  /{product_id}/hld/{hld_id}/generate/status``
- ``POST /{product_id}/hld/{hld_id}/verify``   (owner/admin)
- ``POST /{product_id}/hld/{hld_id}/pages/{page_id}/verify``

Cancellation and page/versions reuse the generic segment routes
(``{segment}=hlds`` in the docgen and doc-versions routers).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.auth.deps import get_current_user, require_product_access
from api.db import get_db
from api.models import ProductORM, UserORM
from api.routers.docgen import GenerateDocRequest, _get_status, _start_generate
from api.routers.products import _verify_entity
from api.schemas import Product

router = APIRouter(
    prefix="/api/products",
    tags=["hld"],
    dependencies=[Depends(get_current_user)],
)


@router.post("/{product_id}/hld/generate")
async def generate_hld(
    product_id: str,
    request_data: GenerateDocRequest,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
    user: UserORM = Depends(get_current_user),
):
    """Start the HLD generation job (202 + job_id; dedups an in-flight job)."""
    return _start_generate(
        db, product_id, "hld", f"hld_{product_id}", request_data, user.id
    )


@router.get("/{product_id}/hld/{hld_id}/generate/status")
async def get_hld_docgen_status(
    product_id: str,
    hld_id: str,
    job_id: str = Query(..., description="Docgen job id returned by the generate endpoint"),
    _product: ProductORM = Depends(require_product_access("ro")),
):
    return _get_status(product_id, "hld", hld_id, job_id)


@router.post("/{product_id}/hld/{hld_id}/verify", response_model=Product)
async def verify_hld(
    product_id: str,
    hld_id: str,
    db: Session = Depends(get_db),
    user: UserORM = Depends(get_current_user),
    light: bool = Query(False),
):
    return _verify_entity(db, product_id, hld_id, "hld", user, light=light)


@router.post(
    "/{product_id}/hld/{hld_id}/pages/{page_id}/verify",
    response_model=Product,
)
async def verify_hld_page(
    product_id: str,
    hld_id: str,
    page_id: str,
    db: Session = Depends(get_db),
    user: UserORM = Depends(get_current_user),
    light: bool = Query(False),
):
    """Verify a single HLD page (owner/admin); flags reset on content change."""
    return _verify_entity(
        db, product_id, hld_id, "hld", user, page_id=page_id, light=light
    )


__all__ = ["router"]
