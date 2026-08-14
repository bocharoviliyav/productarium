"""Product/Codebase/Spec/Links repository — ORM<->Pydantic mapping + persistence.

All Product + child-entity DB access (load, upsert, add/remove, content update)
lives here. No FastAPI dependencies — pure SQLAlchemy.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, List, Optional, Tuple

from sqlalchemy.orm import Session, selectinload

from api.db import get_db  # re-exported so routers import it from the repo
from api.models import (
    CodebaseORM,
    LinksORM,
    ProductORM,
    SpecORM,
)
from api.schemas import Codebase, Links, Product, Spec

logger = logging.getLogger(__name__)


# --- ORM<->Pydantic mapping -------------------------------------------------
def _codebase_orm_from_pydantic(c: Codebase) -> CodebaseORM:
    return CodebaseORM(
        id=c.id,
        name=c.name,
        repo_url=c.repo_url,
        repo_type=c.repo_type,
        token=c.token,
        generated_docs=c.generated_docs,
        pages=c.pages,
        verified=c.verified,
        verified_by=c.verified_by,
        verified_at=c.verified_at,
        source=c.source or "manual",
    )


def _spec_orm_from_pydantic(s: Spec) -> SpecORM:
    return SpecORM(
        id=s.id,
        name=s.name,
        kind=s.kind or "openapi",
        content=s.content,
        verified=s.verified,
        verified_by=s.verified_by,
        verified_at=s.verified_at,
        source=s.source or "manual",
    )


def _links_orm_from_pydantic(l: Links) -> LinksORM:
    return LinksORM(
        id=l.id,
        name=l.name,
        content=l.content,
        verified=l.verified,
        verified_by=l.verified_by,
        verified_at=l.verified_at,
        source=l.source or "manual",
    )


def orm_to_product(p_orm: ProductORM) -> Product:
    """Convert a ProductORM (with children eagerly loaded) to the Pydantic Product."""
    return Product(
        id=p_orm.id,
        name=p_orm.name,
        description=p_orm.description,
        summary=p_orm.summary,
        owner_id=p_orm.owner_id,
        codebases=[
            Codebase(
                id=c.id,
                name=c.name,
                repo_url=c.repo_url,
                repo_type=c.repo_type,
                token=c.token,
                generated_docs=c.generated_docs,
                pages=c.pages,
                verified=c.verified,
                verified_by=c.verified_by,
                verified_at=c.verified_at,
                source=c.source,
            )
            for c in p_orm.codebases
        ],
        specs=[
            Spec(
                id=s.id,
                name=s.name,
                kind=s.kind,
                content=s.content,
                verified=s.verified,
                verified_by=s.verified_by,
                verified_at=s.verified_at,
                source=s.source,
            )
            for s in p_orm.specs
        ],
        links=[
            Links(
                id=l.id,
                name=l.name,
                content=l.content,
                verified=l.verified,
                verified_by=l.verified_by,
                verified_at=l.verified_at,
                source=l.source,
            )
            for l in p_orm.links
        ],
    )


# --- Product queries --------------------------------------------------------
def _load_options():
    return (
        selectinload(ProductORM.codebases),
        selectinload(ProductORM.specs),
        selectinload(ProductORM.links),
    )


def load_product_orm(db: Session, product_id: str) -> Optional[ProductORM]:
    """Fetch a single ProductORM with its codebases/specs/links eagerly loaded."""
    q = db.query(ProductORM).filter(ProductORM.id == product_id)
    for opt in _load_options():
        q = q.options(opt)
    return q.first()


def list_products(db: Session) -> List[Product]:
    """List all products with children eagerly loaded, as Pydantic models."""
    q = db.query(ProductORM)
    for opt in _load_options():
        q = q.options(opt)
    return [orm_to_product(p) for p in q.all()]


def upsert_product(db: Session, product: Product) -> ProductORM:
    """Insert or update a Product and fully replace its codebases/specs/links.

    Mirrors the previous JSON overwrite semantics (full replace of the child
    lists) so POST/PUT stay drop-in compatible.
    """
    p_orm = db.get(ProductORM, product.id)
    if p_orm is None:
        p_orm = ProductORM(
            id=product.id,
            name=product.name,
            description=product.description,
            summary=product.summary,
            owner_id=product.owner_id,
        )
        db.add(p_orm)
    else:
        p_orm.name = product.name
        p_orm.description = product.description
        p_orm.summary = product.summary
        p_orm.owner_id = product.owner_id

    for model in (CodebaseORM, SpecORM, LinksORM):
        db.query(model).filter(model.product_id == product.id).delete(
            synchronize_session=False
        )
    db.flush()
    for c in product.codebases:
        orm = _codebase_orm_from_pydantic(c)
        orm.product_id = product.id
        db.add(orm)
    for s in product.specs:
        orm = _spec_orm_from_pydantic(s)
        orm.product_id = product.id
        db.add(orm)
    for l in product.links:
        orm = _links_orm_from_pydantic(l)
        orm.product_id = product.id
        db.add(orm)

    db.commit()
    db.refresh(p_orm)
    return p_orm


def delete_product(db: Session, product_id: str) -> None:
    """Delete a product (children cascade). No-op if missing."""
    p_orm = db.get(ProductORM, product_id)
    if p_orm is not None:
        db.delete(p_orm)
        db.commit()


# --- Per-type add / delete --------------------------------------------------
def _add_child(db: Session, product_id: str, orm, collection: str) -> Product:
    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    existing = next((x for x in getattr(p_orm, collection) if x.id == orm.id), None)
    if existing is not None:
        getattr(p_orm, collection).remove(existing)
        db.flush()
    getattr(p_orm, collection).append(orm)
    db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm)


def _delete_child(db: Session, product_id: str, entity_id: str, model, collection: str) -> Product:
    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    existing = next((x for x in getattr(p_orm, collection) if x.id == entity_id), None)
    if existing is not None:
        getattr(p_orm, collection).remove(existing)
        db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm)


def add_codebase(db: Session, product_id: str, codebase: Codebase) -> Product:
    orm = _codebase_orm_from_pydantic(codebase)
    orm.product_id = product_id
    return _add_child(db, product_id, orm, "codebases")


def add_spec(db: Session, product_id: str, spec: Spec) -> Product:
    orm = _spec_orm_from_pydantic(spec)
    orm.product_id = product_id
    return _add_child(db, product_id, orm, "specs")


def add_links(db: Session, product_id: str, links: Links) -> Product:
    orm = _links_orm_from_pydantic(links)
    orm.product_id = product_id
    return _add_child(db, product_id, orm, "links")


def delete_codebase(db: Session, product_id: str, codebase_id: str) -> Product:
    return _delete_child(db, product_id, codebase_id, CodebaseORM, "codebases")


def delete_spec(db: Session, product_id: str, spec_id: str) -> Product:
    return _delete_child(db, product_id, spec_id, SpecORM, "specs")


def delete_links(db: Session, product_id: str, links_id: str) -> Product:
    return _delete_child(db, product_id, links_id, LinksORM, "links")


# --- Content updates (WYSIWYG saves) ----------------------------------------
def update_codebase_content(
    db: Session,
    product_id: str,
    codebase_id: str,
    *,
    pages: Optional[dict] = None,
    page_id: Optional[str] = None,
    content: Optional[str] = None,
    generated_docs: Optional[str] = None,
) -> Tuple[Product, Optional[str]]:
    """Apply one of the WYSIWYG edit shapes to a codebase's docs.

    Returns (product, indexed_text) where indexed_text is what should be
    re-indexed into cognee (may be None). Raises ValueError if product or
    codebase is missing, or if no edit shape was provided.
    """
    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    codebase = next((c for c in p_orm.codebases if c.id == codebase_id), None)
    if codebase is None:
        raise ValueError("Codebase not found")

    indexed_text: Optional[str] = None

    if pages is not None:
        codebase.pages = pages
        indexed_text = json.dumps(pages, ensure_ascii=False)
    elif page_id is not None and content is not None:
        current = codebase.pages if isinstance(codebase.pages, dict) else {}
        page = current.get(page_id)
        if page is None:
            current[page_id] = {
                "id": page_id,
                "title": page_id,
                "content": content,
                "filePaths": [],
                "importance": "medium",
                "relatedPages": [],
            }
        else:
            page["content"] = content
            current[page_id] = page
        codebase.pages = current
        indexed_text = content
    elif generated_docs is not None:
        codebase.generated_docs = generated_docs
        indexed_text = generated_docs
    else:
        raise ValueError(
            "Provide one of: pages, (page_id + content), or generated_docs"
        )

    db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm), indexed_text


def update_spec_content(
    db: Session, product_id: str, spec_id: str, content: Optional[str]
) -> Tuple[Product, Optional[str]]:
    """Replace a spec's raw content (authored directly, no generation)."""
    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    spec = next((s for s in p_orm.specs if s.id == spec_id), None)
    if spec is None:
        raise ValueError("Spec not found")
    spec.content = content
    db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm), content


def update_links_content(
    db: Session, product_id: str, links_id: str, content: Optional[str]
) -> Tuple[Product, Optional[str]]:
    """Replace a links collection's raw content (JSON array of {url, description})."""
    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    links = next((l for l in p_orm.links if l.id == links_id), None)
    if links is None:
        raise ValueError("Links not found")
    links.content = content
    db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm), content


# --- Verification (item 5) --------------------------------------------------
def verify_child(
    db: Session, product_id: str, entity_id: str, collection: str, user_id: str
) -> Product:
    """Mark a codebase/spec/links entity as verified by ``user_id``."""
    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    entity = next((x for x in getattr(p_orm, collection) if x.id == entity_id), None)
    if entity is None:
        raise ValueError("Entity not found")
    entity.verified = True
    entity.verified_by = user_id
    entity.verified_at = datetime.utcnow()
    db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm)
