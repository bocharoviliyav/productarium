"""Product/Codebase/Spec/Links repository — ORM<->Pydantic mapping + persistence.

All Product + child-entity DB access (load, upsert, add/remove, content update)
lives here. No FastAPI dependencies — pure SQLAlchemy.

Security invariants (P0):

- Codebase git tokens are WRITE-ONLY: a plaintext token from a client payload
  is encrypted (Fernet) before storage and never serialized back — responses
  expose only ``has_token``. An empty token on update keeps the stored one.
- The verified triple (verified/verified_by/verified_at) is SERVER-OWNED: it is
  never copied from client payloads; full-replace upserts preserve it by id,
  and only the verify endpoints (owner/admin) mutate it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from api.config.settings import decrypt_secret, encrypt_secret, is_encrypted_secret
from api.db import get_db  # re-exported so routers import it from the repo
from api.models import (
    CodebaseORM,
    DatabaseORM,
    DocVersionORM,
    LinksORM,
    ProductORM,
    SpecORM,
)
from api.repositories.doc_version_repo import (
    append_version,
    ensure_baseline_version,
)
from api.schemas import Codebase, Database, Links, Product, ProductListItem, Spec

logger = logging.getLogger(__name__)


# --- Git token helpers (P0-2) -----------------------------------------------
def _encrypt_stored_token(plaintext: Optional[str]) -> Optional[str]:
    """Encrypt a git token for storage; None/empty stays None.

    Falls back to storing the plaintext (with a warning) only when Fernet is
    unavailable — same policy as the settings store.
    """
    if not plaintext:
        return None
    encrypted = encrypt_secret(plaintext)
    if encrypted is None:
        logger.warning(
            "Fernet unavailable; storing a codebase git token in plaintext"
        )
        return plaintext
    return encrypted


def get_codebase_token(c: Any) -> Optional[str]:
    """Plaintext git token for a codebase (ORM row or object with ``.token``).

    Decrypts Fernet ciphertext; legacy plaintext values (pre-migration rows)
    pass through unchanged. Never raises, never logs the value.
    """
    stored = getattr(c, "token", None)
    if not stored:
        return None
    if is_encrypted_secret(stored):
        return decrypt_secret(stored)
    return stored


def migrate_plaintext_tokens(db: Optional[Session] = None) -> int:
    """Lazy migration (P0-2): encrypt any legacy plaintext codebase tokens.

    Called once at startup (non-fatal). Rows already storing Fernet ciphertext
    (``gAAAA…``) are skipped. Returns the number of migrated rows.
    """
    try:
        if db is None:
            from api.db import SessionLocal

            with SessionLocal() as session:
                return migrate_plaintext_tokens(session)
        migrated = 0
        rows = (
            db.query(CodebaseORM)
            .filter(CodebaseORM.token.isnot(None), CodebaseORM.token != "")
            .all()
        )
        for row in rows:
            if row.token and not is_encrypted_secret(row.token):
                encrypted = encrypt_secret(row.token)
                if encrypted is not None:
                    row.token = encrypted
                    migrated += 1
        if migrated:
            db.commit()
            logger.info("Encrypted %d legacy plaintext codebase token(s).", migrated)
        return migrated
    except Exception as e:  # non-fatal by contract
        logger.warning("Codebase token migration failed: %s", e)
        return 0


# --- ORM<->Pydantic mapping -------------------------------------------------
def _codebase_orm_from_pydantic(
    c: Codebase, *, stored_token: Optional[str] = None
) -> CodebaseORM:
    """Build a CodebaseORM from a client payload.

    ``stored_token`` is the already-encrypted value to persist — callers
    resolve it via :func:`_resolved_stored_token` (empty payload token = keep
    the existing stored value). The verified triple is intentionally NOT
    copied from the payload (P0-5: server-owned).
    """
    return CodebaseORM(
        id=c.id,
        name=c.name,
        repo_url=c.repo_url,
        repo_type=c.repo_type,
        token=stored_token,
        generated_docs=c.generated_docs,
        pages=c.pages,
        source=c.source or "manual",
    )


def _resolved_stored_token(
    payload_token: Optional[str], existing_token: Optional[str]
) -> Optional[str]:
    """Merge rule for the write-only token (P0-2):

    non-empty payload token → its ciphertext; empty/None → keep the existing
    stored value (also None for brand-new entities).
    """
    raw = (payload_token or "").strip() if payload_token is not None else ""
    if raw:
        return _encrypt_stored_token(raw)
    return existing_token or None


def _spec_orm_from_pydantic(s: Spec) -> SpecORM:
    # verified triple is server-owned (P0-5) — not copied from the payload
    return SpecORM(
        id=s.id,
        name=s.name,
        kind=s.kind or "openapi",
        content=s.content,
        source=s.source or "manual",
    )


def _links_orm_from_pydantic(l: Links) -> LinksORM:
    # verified triple is server-owned (P0-5) — not copied from the payload
    return LinksORM(
        id=l.id,
        name=l.name,
        content=l.content,
        source=l.source or "manual",
    )


def _database_orm_from_pydantic(d: Database) -> DatabaseORM:
    """Build a DatabaseORM from the API model.

    Secret hygiene: a raw ``dsn`` is masked via ``mask_dsn`` and ONLY the
    masked form reaches the ORM (``dsn_masked``); a provided ``dsn_masked``
    (round-trip echo) is preferred as-is, then re-masked defensively so a
    client-crafted "dsn_masked" containing credentials cannot smuggle a
    secret into storage either. The PRESET path (``db_type``) never passes
    through here — the databases router builds its own DatabaseORM with
    ``dsn_masked=None`` (the raw DSN lives only in the preset server's
    encrypted env); ``db_type`` is still copied for callers that send it
    without a preset flow (it is then informational only).
    """
    from api.docgen.verification import mask_dsn

    dsn_masked = d.dsn_masked if d.dsn_masked is not None else d.dsn
    return DatabaseORM(
        id=d.id,
        name=d.name,
        db_type=d.db_type,
        dsn_masked=mask_dsn(dsn_masked) if dsn_masked else None,
        mcp_server_id=d.mcp_server_id,
        generated_docs=d.generated_docs,
        pages=d.pages,
        verified=d.verified,
        verified_by=d.verified_by,
        verified_at=d.verified_at,
        source=d.source or "manual",
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
                # P0-2: the stored token (ciphertext) is NEVER exposed; only
                # the boolean fact that one exists.
                has_token=bool(c.token),
                generated_docs=c.generated_docs,
                pages=c.pages,
                current_version=c.current_version,
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
                current_version=s.current_version,
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
        databases=[
            Database(
                id=d.id,
                name=d.name,
                db_type=d.db_type,
                dsn_masked=d.dsn_masked,
                mcp_server_id=d.mcp_server_id,
                generated_docs=d.generated_docs,
                pages=d.pages,
                current_version=d.current_version,
                verified=d.verified,
                verified_by=d.verified_by,
                verified_at=d.verified_at,
                source=d.source,
            )
            for d in p_orm.databases
        ],
    )


# --- Product queries --------------------------------------------------------
def _load_options():
    return (
        selectinload(ProductORM.codebases),
        selectinload(ProductORM.specs),
        selectinload(ProductORM.links),
        selectinload(ProductORM.databases),
    )


def load_product_orm(db: Session, product_id: str) -> Optional[ProductORM]:
    """Fetch a single ProductORM with its codebases/specs/links eagerly loaded."""
    q = db.query(ProductORM).filter(ProductORM.id == product_id)
    for opt in _load_options():
        q = q.options(opt)
    return q.first()


def _count_subquery(model, *, verified: bool = False):
    """Correlated ``SELECT count(*)`` subquery over a child table (P1-16).

    Counted in SQL — child rows (and their Text payloads) are never loaded.
    """
    from sqlalchemy import select as _select

    sq = _select(func.count()).select_from(model).where(
        model.product_id == ProductORM.id
    )
    if verified:
        sq = sq.where(model.verified.is_(True))
    return sq.scalar_subquery()


def list_products_light(
    db: Session,
    product_ids: Optional[Any] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[ProductListItem], int]:
    """Light paginated product listing (P1-16): counters in SQL, no Text fields.

    Serves the bare-list contract: the caller returns these rows as a plain
    JSON array and carries ``total`` out-of-band (``X-Total-Count`` header).
    ``total`` is the filtered count BEFORE pagination. ``product_ids``
    restricts the listing to visible products (None = no restriction;
    empty iterable -> ([], 0) without querying).
    """
    base = db.query(ProductORM)
    if product_ids is not None:
        ids = list(product_ids)
        if not ids:
            return [], 0
        base = base.filter(ProductORM.id.in_(ids))
    total = base.count()
    rows = (
        base.with_entities(
            ProductORM.id,
            ProductORM.name,
            ProductORM.description,
            ProductORM.summary,
            ProductORM.owner_id,
            ProductORM.created_at,
            ProductORM.updated_at,
            _count_subquery(CodebaseORM).label("codebases_count"),
            _count_subquery(CodebaseORM, verified=True).label("verified_codebases"),
            _count_subquery(SpecORM).label("specs_count"),
            _count_subquery(SpecORM, verified=True).label("verified_specs"),
            _count_subquery(LinksORM).label("links_count"),
            _count_subquery(LinksORM, verified=True).label("verified_links"),
            _count_subquery(DatabaseORM).label("databases_count"),
            _count_subquery(DatabaseORM, verified=True).label("verified_databases"),
        )
        .order_by(ProductORM.created_at, ProductORM.id)
        .offset(max(0, offset))
        .limit(max(1, limit))
        .all()
    )
    items = [
        ProductListItem(
            id=r.id,
            name=r.name,
            description=r.description or "",
            summary=r.summary,
            owner_id=r.owner_id,
            created_at=r.created_at,
            updated_at=r.updated_at,
            codebases_count=r.codebases_count or 0,
            verified_codebases=r.verified_codebases or 0,
            specs_count=r.specs_count or 0,
            verified_specs=r.verified_specs or 0,
            links_count=r.links_count or 0,
            verified_links=r.verified_links or 0,
            databases_count=r.databases_count or 0,
            verified_databases=r.verified_databases or 0,
        )
        for r in rows
    ]
    return items, total


def _entity_lock(entity_type: str, entity_id: str):
    """Context manager serializing API writes with running docgen jobs.

    Lazy import so ``product_repo`` stays free of docgen-package imports at
    module load (the docgen package lazily imports this module at runtime).
    Raises ``EntityBusyError`` (mapped to HTTP 409 by the routers) when a
    docgen job holds the lock for this entity.
    """
    from api.docgen.jobs import lock_for_entity  # lazy: avoid import cycle weight

    return lock_for_entity(entity_type, entity_id)


_ENTITY_TYPE_BY_COLLECTION = {
    "codebases": "codebase",
    "specs": "spec",
    "links": "links",
    "databases": "database",
}


def upsert_product(db: Session, product: Product) -> ProductORM:
    """Insert or update a Product with PER-ID child upserts (P1-19).

    Children in the payload are updated in place when a row with the same id
    exists, inserted when new, and rows absent from the payload are deleted
    (same end state as the old delete-all+reinsert, but rows are never deleted
    and re-created under their id, so a concurrent docgen commit to the same
    row cannot be wiped). Server-owned state survives: the verified triple
    stays untouched on existing rows (P0-5) and codebase tokens follow the
    write-only merge rule (P0-2).

    The whole operation runs under the entity locks of every touched child
    (existing ∪ payload ids, acquired in sorted order — deadlock-free) so a
    concurrently running docgen job for any of these entities either finishes
    first or we fail fast with ``EntityBusyError`` (HTTP 409 at the router).
    """
    import contextlib

    existing_ids = {
        "codebase": {
            row[0] for row in db.query(CodebaseORM.id).filter(
                CodebaseORM.product_id == product.id
            )
        },
        "spec": {
            row[0] for row in db.query(SpecORM.id).filter(
                SpecORM.product_id == product.id
            )
        },
        "links": {
            row[0] for row in db.query(LinksORM.id).filter(
                LinksORM.product_id == product.id
            )
        },
    }
    payload_ids = {
        "codebase": {c.id for c in product.codebases},
        "spec": {s.id for s in product.specs},
        "links": {l.id for l in product.links},
    }
    # Fixed sorted order keeps concurrent multi-lock acquisitions deadlock-free.
    keys = sorted(
        f"{etype}:{eid}"
        for etype in ("codebase", "spec", "links")
        for eid in existing_ids[etype] | payload_ids[etype]
    )
    with contextlib.ExitStack() as stack:
        for key in keys:
            etype, _, eid = key.partition(":")
            stack.enter_context(_entity_lock(etype, eid))
        return _upsert_product_locked(db, product)


def _upsert_product_locked(db: Session, product: Product) -> ProductORM:
    """Per-id child upsert — caller must already hold the entity locks."""
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

    # Snapshot server-owned state per (collection, id) before the full replace
    # (plain values — the rows themselves are deleted below). P0-5: the
    # verified triple is server-owned; P0-2: codebase tokens follow the
    # write-only merge rule.
    saved: dict = {}
    for model, coll in (
        (CodebaseORM, "codebases"),
        (SpecORM, "specs"),
        (LinksORM, "links"),
    ):
        for row in db.query(model).filter(model.product_id == product.id).all():
            saved[(coll, row.id)] = {
                "verified": row.verified,
                "verified_by": row.verified_by,
                "verified_at": row.verified_at,
                # Only CodebaseORM carries a git token column.
                "token": row.token if model is CodebaseORM else None,
                # Version pointer is server-owned (doc_versions history).
                "current_version": getattr(row, "current_version", None),
            }

    # Verified state is server-owned (review #4): capture the STORED
    # verification triple per database id BEFORE the child replace, so the
    # re-insert below preserves it for existing artifacts and forces False
    # for new ones — a client cannot grant verification through the payload
    # (only the owner/admin verify endpoint can).
    prev_db_verified = {
        d.id: (d.verified, d.verified_by, d.verified_at)
        for d in p_orm.databases
    }
    prev_db_version = {d.id: d.current_version for d in p_orm.databases}

    for model in (CodebaseORM, SpecORM, LinksORM, DatabaseORM):
        db.query(model).filter(model.product_id == product.id).delete(
            synchronize_session=False
        )
    db.flush()
    for c in product.codebases:
        prev = saved.get(("codebases", c.id))
        orm = _codebase_orm_from_pydantic(
            c,
            stored_token=_resolved_stored_token(
                c.token, prev.get("token") if prev else None
            ),
        )
        _restore_server_state(orm, prev)
        orm.product_id = product.id
        db.add(orm)
    for s in product.specs:
        prev = saved.get(("specs", s.id))
        orm = _spec_orm_from_pydantic(s)
        _restore_server_state(orm, prev)
        orm.product_id = product.id
        db.add(orm)
    for l in product.links:
        prev = saved.get(("links", l.id))
        orm = _links_orm_from_pydantic(l)
        _restore_server_state(orm, prev)
        orm.product_id = product.id
        db.add(orm)
    for d in product.databases:
        orm = _database_orm_from_pydantic(d)
        verified, verified_by, verified_at = prev_db_verified.get(
            d.id, (False, None, None)
        )
        orm.verified, orm.verified_by, orm.verified_at = (
            verified, verified_by, verified_at,
        )
        orm.current_version = prev_db_version.get(d.id)
        orm.product_id = product.id
        db.add(orm)

    db.commit()
    db.refresh(p_orm)
    return p_orm


def delete_product(db: Session, product_id: str) -> None:
    """Delete a product (children + doc version history cascade). No-op if missing."""
    p_orm = db.get(ProductORM, product_id)
    if p_orm is not None:
        db.query(DocVersionORM).filter(
            DocVersionORM.product_id == product_id
        ).delete(synchronize_session=False)
        db.delete(p_orm)
        db.commit()


def _restore_server_state(orm, prev) -> None:
    """Restore server-owned fields from the previous state with the same id.

    ``prev`` is either a plain-value snapshot dict (upsert full replace) or a
    live ORM row (_add_child replace). P0-5: verified/verified_by/verified_at
    are owned by the verify endpoints — a client re-sending an entity cannot
    reset or forge them.
    """
    if prev is None:
        return
    if isinstance(prev, dict):
        orm.verified = prev.get("verified") or False
        orm.verified_by = prev.get("verified_by")
        orm.verified_at = prev.get("verified_at")
        if hasattr(orm, "current_version"):
            orm.current_version = prev.get("current_version")
    else:
        orm.verified = prev.verified
        orm.verified_by = prev.verified_by
        orm.verified_at = prev.verified_at
        if hasattr(orm, "current_version"):
            orm.current_version = getattr(prev, "current_version", None)


# --- Per-type add / delete --------------------------------------------------
def _add_child(db: Session, product_id: str, orm, collection: str) -> Product:
    with _entity_lock(_ENTITY_TYPE_BY_COLLECTION[collection], orm.id):
        p_orm = load_product_orm(db, product_id)
        if p_orm is None:
            raise ValueError("Product not found")
        existing = next((x for x in getattr(p_orm, collection) if x.id == orm.id), None)
        if existing is not None:
            _restore_server_state(orm, existing)
            if isinstance(orm, CodebaseORM) and not orm.token:
                # write-only token merge: empty payload token keeps the stored one
                orm.token = existing.token
            getattr(p_orm, collection).remove(existing)
            db.flush()
        getattr(p_orm, collection).append(orm)
        db.commit()
        db.refresh(p_orm)
        return orm_to_product(p_orm)


def _delete_child(db: Session, product_id: str, entity_id: str, model, collection: str) -> Product:
    with _entity_lock(_ENTITY_TYPE_BY_COLLECTION[collection], entity_id):
        p_orm = load_product_orm(db, product_id)
        if p_orm is None:
            raise ValueError("Product not found")
        existing = next((x for x in getattr(p_orm, collection) if x.id == entity_id), None)
        if existing is not None:
            getattr(p_orm, collection).remove(existing)
            db.query(DocVersionORM).filter(
                DocVersionORM.entity_id == entity_id
            ).delete(synchronize_session=False)
            db.commit()
        db.refresh(p_orm)
        return orm_to_product(p_orm)


def add_codebase(db: Session, product_id: str, codebase: Codebase) -> Product:
    existing_token = (
        db.query(CodebaseORM.token)
        .filter(CodebaseORM.product_id == product_id, CodebaseORM.id == codebase.id)
        .scalar()
    )
    orm = _codebase_orm_from_pydantic(
        codebase,
        stored_token=_resolved_stored_token(codebase.token, existing_token),
    )
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


def add_database(db: Session, product_id: str, database: Database) -> Product:
    orm = _database_orm_from_pydantic(database)
    orm.product_id = product_id
    return _add_child(db, product_id, orm, "databases")


def delete_codebase(db: Session, product_id: str, codebase_id: str) -> Product:
    return _delete_child(db, product_id, codebase_id, CodebaseORM, "codebases")


def delete_spec(db: Session, product_id: str, spec_id: str) -> Product:
    return _delete_child(db, product_id, spec_id, SpecORM, "specs")


def delete_links(db: Session, product_id: str, links_id: str) -> Product:
    return _delete_child(db, product_id, links_id, LinksORM, "links")


def delete_database(db: Session, product_id: str, database_id: str) -> Product:
    return _delete_child(db, product_id, database_id, DatabaseORM, "databases")


# --- Content updates (WYSIWYG saves) ----------------------------------------
# Per-page verification flags (server-owned; ride inside the pages JSON dict).
_PAGE_VERIFY_KEYS = ("verified", "verified_by", "verified_at")


def _strip_page_flags(pages: dict) -> dict:
    """Drop client-supplied per-page verification flags (server-owned)."""
    out = {}
    for pid, page in pages.items():
        if isinstance(page, dict) and any(k in page for k in _PAGE_VERIFY_KEYS):
            page = {k: v for k, v in page.items() if k not in _PAGE_VERIFY_KEYS}
        out[pid] = page
    return out


def update_database_content(
    db: Session,
    product_id: str,
    database_id: str,
    *,
    pages: Optional[dict] = None,
    page_id: Optional[str] = None,
    content: Optional[str] = None,
    generated_docs: Optional[str] = None,
) -> Tuple[Product, Optional[str]]:
    """Apply one of the WYSIWYG edit shapes to a database's docs.

    Same shapes/semantics as ``update_codebase_content`` (the artifact viewer
    reuses the codebase editor for databases): ``pages`` wholesale, a single
    page upsert via ``page_id`` + ``content``, or the whole ``generated_docs``
    blob. Returns (product, indexed_text) for memory re-indexing; raises
    ValueError when the product/database is missing or no shape was provided.
    """
    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    database = next((d for d in p_orm.databases if d.id == database_id), None)
    if database is None:
        raise ValueError("Database not found")

    ensure_baseline_version(db, "database", database)
    indexed_text: Optional[str] = None

    if pages is not None:
        pages = _strip_page_flags(pages)
        database.pages = pages
        indexed_text = json.dumps(pages, ensure_ascii=False)
    elif page_id is not None and content is not None:
        # Copy-on-write: SQLAlchemy does NOT track in-place mutations of a
        # JSON column, so mutating the loaded dict in place would silently
        # persist nothing — build fresh dicts so the assignment marks the
        # column dirty.
        current = dict(database.pages) if isinstance(database.pages, dict) else {}
        page = dict(current.get(page_id) or {})
        if page:
            # Verification binds to exact content: a changed page resets it.
            if page.get("content") != content:
                for key in _PAGE_VERIFY_KEYS:
                    page.pop(key, None)
            page["content"] = content
        else:
            page = {
                "id": page_id,
                "title": page_id,
                "content": content,
                "filePaths": [],
                "importance": "medium",
                "relatedPages": [],
            }
        current[page_id] = page
        database.pages = current
        indexed_text = content
    elif generated_docs is not None:
        database.generated_docs = generated_docs
        indexed_text = generated_docs
    else:
        raise ValueError(
            "Provide one of: pages, (page_id + content), or generated_docs"
        )

    append_version(db, "database", database, source="edit")
    db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm), indexed_text


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
    re-indexed into the memory backend (may be None). Raises ValueError if
    product or codebase is missing, or if no edit shape was provided.
    """
    with _entity_lock("codebase", codebase_id):
        p_orm = load_product_orm(db, product_id)
        if p_orm is None:
            raise ValueError("Product not found")
        codebase = next((c for c in p_orm.codebases if c.id == codebase_id), None)
        if codebase is None:
            raise ValueError("Codebase not found")

        ensure_baseline_version(db, "codebase", codebase)
        indexed_text: Optional[str] = None

        if pages is not None:
            pages = _strip_page_flags(pages)
            codebase.pages = pages
            indexed_text = json.dumps(pages, ensure_ascii=False)
        elif page_id is not None and content is not None:
            # Copy-on-write (same as update_database_content): in-place mutation
            # of the loaded JSON dict would silently persist nothing.
            current = dict(codebase.pages) if isinstance(codebase.pages, dict) else {}
            page = dict(current.get(page_id) or {})
            if page:
                # Verification binds to exact content: a changed page resets it.
                if page.get("content") != content:
                    for key in _PAGE_VERIFY_KEYS:
                        page.pop(key, None)
                page["content"] = content
            else:
                page = {
                    "id": page_id,
                    "title": page_id,
                    "content": content,
                    "filePaths": [],
                    "importance": "medium",
                    "relatedPages": [],
                }
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

        append_version(db, "codebase", codebase, source="edit")
        db.commit()
        db.refresh(p_orm)
        return orm_to_product(p_orm), indexed_text


def update_spec_content(
    db: Session, product_id: str, spec_id: str, content: Optional[str]
) -> Tuple[Product, Optional[str]]:
    """Replace a spec's raw content (authored directly, no generation)."""
    with _entity_lock("spec", spec_id):
        p_orm = load_product_orm(db, product_id)
        if p_orm is None:
            raise ValueError("Product not found")
        spec = next((s for s in p_orm.specs if s.id == spec_id), None)
        if spec is None:
            raise ValueError("Spec not found")
        ensure_baseline_version(db, "spec", spec)
        spec.content = content
        append_version(db, "spec", spec, source="edit")
        db.commit()
        db.refresh(p_orm)
        return orm_to_product(p_orm), content


def update_links_content(
    db: Session, product_id: str, links_id: str, content: Optional[str]
) -> Tuple[Product, Optional[str]]:
    """Replace a links collection's raw content (JSON array of {url, description})."""
    with _entity_lock("links", links_id):
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


def update_database_meta(
    db: Session,
    product_id: str,
    database_id: str,
    *,
    name: Optional[str] = None,
    dsn: Optional[str] = None,
    dsn_masked: Optional[str] = None,
    mcp_server_id: Optional[str] = None,
) -> Product:
    """Update a database artifact's metadata (name / DSN / MCP server pin).

    A raw ``dsn`` is masked via ``mask_dsn`` before persistence; only the
    masked form is stored (same contract as creation). ``mcp_server_id`` may
    be cleared by passing the empty string.
    """
    from api.docgen.verification import mask_dsn

    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    database = next((d for d in p_orm.databases if d.id == database_id), None)
    if database is None:
        raise ValueError("Database not found")
    if name is not None:
        database.name = name
    if dsn is not None or dsn_masked is not None:
        raw = dsn if dsn is not None else dsn_masked
        database.dsn_masked = mask_dsn(raw) if (raw or "").strip() else None
    if mcp_server_id is not None:
        database.mcp_server_id = mcp_server_id or None
    db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm)


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


def verify_page(
    db: Session,
    product_id: str,
    entity_id: str,
    collection: str,
    page_id: str,
    user_id: str,
) -> Product:
    """Mark one documentation page (codebases/databases pages JSON) verified.

    Copy-on-write on the pages dict — in-place JSON mutation is not tracked.
    Verification binds to content: edits and regenerations that change the
    page drop the flags (see ``update_*_content`` and docgen persist).
    """
    p_orm = load_product_orm(db, product_id)
    if p_orm is None:
        raise ValueError("Product not found")
    entity = next((x for x in getattr(p_orm, collection) if x.id == entity_id), None)
    if entity is None:
        raise ValueError("Entity not found")
    pages = entity.pages if isinstance(entity.pages, dict) else {}
    if page_id not in pages:
        raise ValueError("Page not found")
    current = dict(pages)
    page = dict(current[page_id])
    page.update(
        verified=True,
        verified_by=user_id,
        verified_at=datetime.utcnow().isoformat(),
    )
    current[page_id] = page
    entity.pages = current
    db.commit()
    db.refresh(p_orm)
    return orm_to_product(p_orm)
