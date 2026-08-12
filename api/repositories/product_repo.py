"""Product/Artifact repository — ORM<->Pydantic mapping + persistence.

Extracted from the former ``api/api.py`` monolith so routers stay thin.
All Product/Artifact DB access (load, upsert, artifact add/remove, type
normalization) lives here. No FastAPI dependencies — pure SQLAlchemy.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from sqlalchemy.orm import Session, selectinload

from api.db import get_db  # re-exported so routers import it from the repo
from api.models import (
    ArtifactORM,
    ProductORM,
    LEGACY_ARTIFACT_TYPE_MAP,
)
from api.schemas import Artifact, Product

logger = logging.getLogger(__name__)


def normalize_artifact_type(a: Artifact):
    """Map a (possibly legacy) artifact type to the new (type, kind) pair.

    Legacy openapi/asyncapi -> spec (+kind), testcase -> documentation (+kind).
    New types are passed through; an explicit kind on the request wins.
    """
    if a.type in LEGACY_ARTIFACT_TYPE_MAP:
        new_type, default_kind = LEGACY_ARTIFACT_TYPE_MAP[a.type]
        return new_type, a.kind or default_kind
    return a.type, a.kind


def artifact_orm_from_pydantic(a: Artifact) -> ArtifactORM:
    """Build a new (transient) ArtifactORM from a Pydantic Artifact."""
    norm_type, kind = normalize_artifact_type(a)
    return ArtifactORM(
        id=a.id,
        name=a.name,
        type=norm_type,
        kind=kind,
        repo_url=a.repo_url,
        repo_type=a.repo_type,
        token=a.token,
        content=a.content,
        allure_url=a.allure_url,
        generated_docs=a.generated_docs,
        pages=a.pages,
        verified=a.verified,
        verified_by=a.verified_by,
        verified_at=a.verified_at,
        source=a.source or "manual",
    )


def orm_to_product(p_orm: ProductORM) -> Product:
    """Convert a ProductORM (with loaded artifacts) to the Pydantic Product.

    Field names/shapes match the previous JSON-file schema exactly so the
    frontend and Phase B consumers stay compatible. created_at/updated_at are
    intentionally NOT exposed in the public response shape.
    """
    return Product(
        id=p_orm.id,
        name=p_orm.name,
        description=p_orm.description,
        summary=p_orm.summary,
        owner_id=p_orm.owner_id,
        artifacts=[
            Artifact(
                id=a.id,
                name=a.name,
                type=a.type,
                kind=a.kind,
                repo_url=a.repo_url,
                repo_type=a.repo_type,
                token=a.token,
                content=a.content,
                allure_url=a.allure_url,
                generated_docs=a.generated_docs,
                pages=a.pages,
                verified=a.verified,
                verified_by=a.verified_by,
                verified_at=a.verified_at,
                source=a.source,
            )
            for a in p_orm.artifacts
        ],
    )


def load_product_orm(db: Session, product_id: str) -> Optional[ProductORM]:
    """Fetch a single ProductORM with its artifacts eagerly loaded."""
    return (
        db.query(ProductORM)
        .options(selectinload(ProductORM.artifacts))
        .filter(ProductORM.id == product_id)
        .first()
    )


def list_products(db: Session) -> List[Product]:
    """List all products with artifacts eagerly loaded, as Pydantic models."""
    products = (
        db.query(ProductORM)
        .options(selectinload(ProductORM.artifacts))
        .all()
    )
    return [orm_to_product(p) for p in products]


def upsert_product(db: Session, product: Product) -> ProductORM:
    """Insert or update a Product and fully replace its artifacts.

    Mirrors the previous JSON ``save_product`` overwrite semantics (full
    replace of the artifacts list) so POST/PUT stay drop-in compatible.
    Existing artifact rows are deleted and flushed before the new ones are
    inserted to avoid PK collisions within a single flush.
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

    db.query(ArtifactORM).filter(
        ArtifactORM.product_id == product.id
    ).delete(synchronize_session=False)
    db.flush()
    for a in product.artifacts:
        new_a = artifact_orm_from_pydantic(a)
        new_a.product_id = product.id
        db.add(new_a)

    db.commit()
    db.refresh(p_orm)
    return p_orm


def add_artifact(db: Session, product_id: str, artifact: Artifact) -> Product:
    """Add (or dedupe-replace) an artifact on a product. Returns the product."""
    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    existing = next((a for a in p_orm.artifacts if a.id == artifact.id), None)
    if existing is not None:
        p_orm.artifacts.remove(existing)  # cascade delete-orphan
        db.flush()  # flush DELETE before INSERT to avoid PK collision
    p_orm.artifacts.append(artifact_orm_from_pydantic(artifact))
    db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm)


def delete_artifact(db: Session, product_id: str, artifact_id: str) -> Product:
    """Remove an artifact from a product. Returns the refreshed product."""
    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    existing = next((a for a in p_orm.artifacts if a.id == artifact_id), None)
    if existing is not None:
        p_orm.artifacts.remove(existing)  # cascade delete-orphan
        db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm)


def update_artifact_content(
    db: Session,
    product_id: str,
    artifact_id: str,
    *,
    pages=None,
    page_id: Optional[str] = None,
    content: Optional[str] = None,
    generated_docs: Optional[str] = None,
    raw_content: Optional[str] = None,
):
    """Apply one of the WYSIWYG edit shapes to an artifact.

    Returns (product, indexed_text) where indexed_text is what should be
    re-indexed into cognee (may be None). Raises ValueError if product or
    artifact is missing, or if no edit shape was provided.
    """
    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    artifact = next((a for a in p_orm.artifacts if a.id == artifact_id), None)
    if artifact is None:
        raise ValueError("Artifact not found")

    indexed_text: Optional[str] = None

    if pages is not None:
        # Wholesale replace (e.g. a future full-document editor).
        artifact.pages = pages
        indexed_text = json.dumps(pages, ensure_ascii=False)
    elif page_id is not None and content is not None:
        # Upsert a single page's content. ``pages`` is a JSON column persisted
        # as a dict keyed by page_id; tolerate None / non-dict by rebuilding.
        current = artifact.pages if isinstance(artifact.pages, dict) else {}
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
        artifact.pages = current
        indexed_text = content
    elif generated_docs is not None:
        artifact.generated_docs = generated_docs
        indexed_text = generated_docs
    elif raw_content is not None:
        # Spec / links artifacts are authored directly into ``content`` (no
        # generation step). Replacing it keeps the structured viewer in sync;
        # the new text is re-indexed so expert Ask recall stays current.
        artifact.content = raw_content
        indexed_text = raw_content
    else:
        raise ValueError(
            "Provide one of: pages, (page_id + content), generated_docs, or raw_content"
        )

    db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm), indexed_text


def delete_product(db: Session, product_id: str) -> None:
    """Delete a product (artifacts cascade). No-op if missing."""
    p_orm = db.get(ProductORM, product_id)
    if p_orm is not None:
        db.delete(p_orm)
        db.commit()
