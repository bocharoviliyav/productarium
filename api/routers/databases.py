"""Database artifact router — per-type CRUD + reverse-engineering (Wave E).

Endpoints (prefix ``/api/products``, tags ``databases``):

- ``POST   /api/products/{product_id}/databases``                     — add database
- ``DELETE /api/products/{product_id}/databases/{database_id}``       — delete database
- ``PUT    /api/products/{product_id}/databases/{database_id}``       — edit docs / update meta
- ``POST   /api/products/{product_id}/databases/{database_id}/verify``— verify (owner/admin)
- ``POST   /api/products/{product_id}/databases/{database_id}/generate``
    Start the MCP reverse-engineering flow (202 + job_id; poll via status).
- ``GET    /api/products/{product_id}/databases/{database_id}/generate/status?job_id=``

Two add flows (``POST``):

- **Preset flow** (``db_type`` in the payload — the hardcoded UI path):
  validate the DSN against the preset registry → REAL MCP connection check
  (launcher spawn + handshake + tools/list + one probe tool call,
  ``api/mcp/presets.py``) → on success atomically create a DEDICATED
  system-managed ``McpServerORM`` row (stdio, launcher fixed forever, DSN
  only in the Fernet-encrypted env, ``preset_key`` set) + the product
  binding + the ``DatabaseORM`` row with ``dsn_masked=None`` (the DSN is
  never stored or shown anywhere else). Preset databases are immutable in
  their connection settings (``dsn``/``mcp_server_id`` PUTs are rejected);
  deleting the database deletes the dedicated server row + bindings too.
- **Legacy flow** (no ``db_type``): unchanged behavior for API clients.

Secret hygiene: a raw ``dsn`` in the request body is masked via
``api.docgen.verification.mask_dsn`` BEFORE persistence; only ``dsn_masked``
is stored, logged, or returned (the raw DSN never reaches the ORM, the job
registry, or the response body).

``mcp_server_id`` (optional, legacy flow) pins the registry MCP server whose
introspection tools the reverse-engineering flow uses. When provided it must
reference an existing server that is BOUND and ENABLED for this product
(the same visibility rule the expert agent applies); NULL means "all bound
enabled servers". Preset databases always pin their dedicated server.
"""

from __future__ import annotations

import logging
import secrets as pysecrets
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.auth.deps import get_current_user
from api.db import get_db
from api.docgen.jobs import (
    EntityBusyError,
    _progress_snapshot,
    create_or_get_job,
    get_job,
    submit_job,
)
from api.mcp.manager import get_mcp_manager
from api.mcp.presets import (
    PresetConnectionError,
    PresetError,
    check_preset_connection,
    choose_launcher,
    get_preset,
    validate_dsn,
)
from api.mcp.secrets import encrypt_secret_dict
from api.models import (
    DatabaseORM,
    McpServerORM,
    ProductMcpServerORM,
    ProductORM,
    UserORM,
)
from api.docgen.verification import mask_dsn
from api.repositories import product_repo
from api.routers.products import _maybe_light
from api.schemas import Database, Product

logger = logging.getLogger(__name__)

# Same auth posture as the products router: all CRUD + generate endpoints
# require an authenticated user (no-op when AUTH_PROVIDER=none); the verify
# endpoint adds its owner/admin check on top.
router = APIRouter(
    prefix="/api/products",
    tags=["databases"],
    dependencies=[Depends(get_current_user)],
)


class DatabaseUpdate(BaseModel):
    """Partial update of a database artifact (WYSIWYG saves OR metadata).

    Doc-edit shapes (the artifact viewer's editor, same contract as codebases):
      - ``pages``                 → replace the whole pages dict wholesale
      - ``page_id`` + ``content`` → upsert a single page's content field
      - ``generated_docs``        → replace the top-level generated_docs blob
    Metadata shape (the product page's settings form):
      - ``name`` / ``dsn`` / ``mcp_server_id`` — see ``update_database_meta``.
    A doc-edit shape takes precedence when present.
    """

    # --- doc-edit shapes ---
    generated_docs: Optional[str] = None
    page_id: Optional[str] = None
    content: Optional[str] = None
    pages: Optional[Dict[str, Any]] = None

    # --- metadata shape ---
    name: Optional[str] = Field(default=None, max_length=256)
    # Raw DSN; masked via mask_dsn before persistence (never stored as-is).
    dsn: Optional[str] = None
    # MCP server pin; an empty string CLEARS the pin (back to all bound servers).
    mcp_server_id: Optional[str] = Field(default=None, max_length=64)

    def wants_doc_update(self) -> bool:
        return any(
            (
                self.pages is not None,
                self.page_id is not None and self.content is not None,
                self.generated_docs is not None,
            )
        )


class GenerateDatabaseDocsRequest(BaseModel):
    model: Optional[str] = None
    # DEPRECATED no-op: the generation language is controlled by the admin
    # ``generation.language`` setting and resolved when the job starts.
    language: Optional[str] = None


def _validate_mcp_server_pin(db: Session, product_id: str, mcp_server_id: str) -> None:
    """Ensure the pinned MCP server exists and is bound+enabled to the product.

    404 when the server id is unknown; 400 when the server exists but is not
    (or not yet) usable for this product — the reverse-engineering flow may
    only see the same MCP surface the expert agent sees.
    """
    server = db.query(McpServerORM).filter(McpServerORM.id == mcp_server_id).first()
    if server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    binding = (
        db.query(ProductMcpServerORM)
        .filter(
            ProductMcpServerORM.product_id == product_id,
            ProductMcpServerORM.mcp_server_id == mcp_server_id,
            ProductMcpServerORM.enabled.is_(True),
        )
        .first()
    )
    if binding is None or not server.enabled:
        raise HTTPException(
            status_code=400,
            detail=(
                "MCP server is not bound (or not enabled) for this product; "
                "bind it via POST /api/products/{id}/mcp first"
            ),
        )


# --- CRUD --------------------------------------------------------------------
@router.post("/{product_id}/databases", response_model=Product)
async def add_database(
    product_id: str, database: Database, db: Session = Depends(get_db),
    light: bool = Query(False),
):
    if database.db_type:
        product = await _add_preset_database(product_id, database, db)
        _assert_no_raw_dsn(product, database.dsn)
        return _maybe_light(product, light)
    if database.mcp_server_id:
        _validate_mcp_server_pin(db, product_id, database.mcp_server_id)
    # Verified state is server-owned: a client cannot grant verification at
    # creation (only the owner/admin ``verify`` endpoint can — review #4).
    fresh = database.model_copy(
        update={"verified": False, "verified_by": None, "verified_at": None}
    )
    try:
        product = product_repo.add_database(db, product_id, fresh)
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")
    # Defensive: the raw DSN must never appear in the serialized response.
    _assert_no_raw_dsn(product, database.dsn)
    return _maybe_light(product, light)


async def _add_preset_database(
    product_id: str, database: Database, db: Session
) -> Product:
    """The hardcoded preset flow: DSN → real MCP check → server+binding+DB.

    The connection check runs BEFORE any DB write (a failed check must not
    leave rows behind), then everything is created atomically in one commit
    under the database entity lock. The raw DSN is used only in memory: it
    goes into the dedicated server row's Fernet-encrypted env and nowhere
    else — the DatabaseORM row gets ``dsn_masked=None``.
    """
    spec = get_preset(database.db_type)
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown database type {database.db_type!r} "
                   "(see GET /api/db-presets)",
        )
    try:
        dsn = validate_dsn(spec, database.dsn or "")
    except PresetError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        await check_preset_connection(spec, dsn)
    except PresetConnectionError as e:
        # Sanitized already (no DSN); full detail is in the server log.
        raise HTTPException(status_code=400, detail=str(e))

    launcher = choose_launcher(spec, dsn)
    env = spec.build_env(dsn, docker=(launcher.kind == "docker"))
    server_id = f"mcp_{pysecrets.token_hex(16)}"
    binding_id = f"pmb_{pysecrets.token_hex(16)}"

    try:
        with product_repo._entity_lock("database", database.id):
            p_orm = product_repo.load_product_orm(db, product_id)
            if p_orm is None:
                raise HTTPException(status_code=404, detail="Product not found")
            existing = next(
                (d for d in p_orm.databases if d.id == database.id), None
            )
            if existing is not None:
                _drop_preset_server(db, existing)
                p_orm.databases.remove(existing)
                db.flush()
            db.add(McpServerORM(
                id=server_id,
                name=f"preset-{spec.key}-{database.id}",
                preset_key=spec.key,
                transport="stdio",
                command=launcher.command,
                args=list(launcher.args),
                env=encrypt_secret_dict(env),
                enabled=True,
                # The connection check just proved this server works.
                status="ok",
                status_checked_at=datetime.utcnow(),
            ))
            db.add(ProductMcpServerORM(
                id=binding_id,
                product_id=product_id,
                mcp_server_id=server_id,
                enabled=True,
            ))
            p_orm.databases.append(DatabaseORM(
                id=database.id,
                product_id=product_id,
                name=database.name,
                db_type=spec.key,
                dsn_masked=None,
                mcp_server_id=server_id,
                generated_docs=None,
                pages=None,
                verified=False,
                verified_by=None,
                verified_at=None,
                source="preset",
            ))
            db.commit()
        db.refresh(p_orm)
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Database id conflict")
    get_mcp_manager().invalidate(server_id)
    return product_repo.orm_to_product(p_orm)


def _drop_preset_server(db: Session, database_row: DatabaseORM) -> None:
    """Delete the DEDICATED preset server row (+ bindings) of a preset DB.

    No-op for legacy databases (no ``db_type`` / no pinned preset server).
    """
    server_id = getattr(database_row, "mcp_server_id", None)
    if not server_id:
        return
    server = db.query(McpServerORM).filter(McpServerORM.id == server_id).first()
    if server is not None and server.preset_key:
        # Drop the FK reference first so the unit-of-work deletes the
        # databases row (or the whole server) without relying on the
        # DB-level ON DELETE SET NULL.
        database_row.mcp_server_id = None
        db.delete(server)  # product bindings cascade (ORM + FK ON DELETE)
        get_mcp_manager().invalidate(server_id)


@router.delete("/{product_id}/databases/{database_id}", response_model=Product)
async def delete_database(
    product_id: str, database_id: str, db: Session = Depends(get_db),
    light: bool = Query(False),
):
    p_orm = product_repo.load_product_orm(db, product_id)
    if p_orm is not None:
        existing = next(
            (d for d in p_orm.databases if d.id == database_id), None
        )
        if existing is not None:
            _drop_preset_server(db, existing)
    try:
        return _maybe_light(
            product_repo.delete_database(db, product_id, database_id), light
        )
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=404, detail="Product not found")


@router.put("/{product_id}/databases/{database_id}", response_model=Product)
async def update_database(
    product_id: str,
    database_id: str,
    body: DatabaseUpdate,
    db: Session = Depends(get_db),
    light: bool = Query(False),
):
    """Edit a database artifact: generated docs (WYSIWYG) OR metadata.

    Doc-edit shapes mirror the codebase PUT (``pages`` / ``page_id`` +
    ``content`` / ``generated_docs``) and re-index the edited text into the
    product's memory backend (``source_type="database"``). Otherwise the
    request is a metadata update: name / DSN (masked on acceptance; only
    ``dsn_masked`` persists) / MCP server pin (``""`` clears it).

    Preset databases (``db_type`` set) are IMMUTABLE in their connection
    settings: ``dsn`` and ``mcp_server_id`` PUTs are rejected with 400 —
    the connection was validated at creation and is fixed forever; delete
    + re-add is the only way to change it. Name and doc edits stay allowed.
    """
    _reject_preset_connection_changes(db, product_id, database_id, body)
    if body.wants_doc_update():
        try:
            product, indexed_text = product_repo.update_database_content(
                db,
                product_id,
                database_id,
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
        _reindex(product_id, indexed_text, database_id, source_type="database")
        return _maybe_light(product, light)

    if body.mcp_server_id:
        _validate_mcp_server_pin(db, product_id, body.mcp_server_id)
    try:
        product = product_repo.update_database_meta(
            db,
            product_id,
            database_id,
            name=body.name,
            dsn=body.dsn,
            mcp_server_id=body.mcp_server_id,
        )
    except EntityBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError:
        raise HTTPException(status_code=404, detail="Database not found")
    _assert_no_raw_dsn(product, body.dsn)
    return _maybe_light(product, light)


def _reject_preset_connection_changes(
    db: Session, product_id: str, database_id: str, body: "DatabaseUpdate"
) -> None:
    """400 when a preset database's connection settings are being changed.

    Doc-edit shapes and ``name`` are fine; any attempt to touch ``dsn`` or
    ``mcp_server_id`` (including clearing the pin) is rejected.
    """
    if body.dsn is None and body.mcp_server_id is None:
        return
    row = (
        db.query(DatabaseORM)
        .filter(
            DatabaseORM.product_id == product_id,
            DatabaseORM.id == database_id,
        )
        .first()
    )
    if row is not None and row.db_type:
        raise HTTPException(
            status_code=400,
            detail=(
                "Preset database connection settings are immutable; "
                "delete and re-add the database to change them"
            ),
        )


def _reindex(
    product_id: str,
    indexed_text: Optional[str],
    entity_id: str,
    *,
    source_type: str = "database",
) -> None:
    """Re-index edited docs into the product memory backend (fire-and-forget).

    Same handoff as ``api.routers.products._reindex`` (kept local so the two
    routers stay independently testable).
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


def _assert_no_raw_dsn(product: Product, raw_dsn: Optional[str]) -> None:
    """Belt-and-braces leak guard for the serialized product payload.

    A DSN with NO detectable secret legitimately masks to itself
    (``postgres://app@host/db`` / ``postgres://localhost:5432/db``), so raw
    and masked being EQUAL is not an error (review #4: the old check 500'd
    on exactly those common forms). The guard fires only when masking DID
    redact something and the raw form still survives in the payload.
    """
    if not raw_dsn:
        return
    if mask_dsn(raw_dsn) == raw_dsn.strip():
        return  # no secret detected — the masked form legitimately equals raw
    try:
        payload = product.model_dump()
        if raw_dsn in repr(payload):
            logger.error(
                "databases router: raw DSN leaked into the product payload; "
                "refusing to return it"
            )
            raise HTTPException(status_code=500, detail="Internal masking error")
    except HTTPException:
        raise
    except Exception:  # pragma: no cover - defensive
        pass


# --- Verification (owner or admin) -------------------------------------------
@router.post("/{product_id}/databases/{database_id}/verify", response_model=Product)
async def verify_database(
    product_id: str,
    database_id: str,
    db: Session = Depends(get_db),
    user: UserORM = Depends(get_current_user),
    light: bool = Query(False),
):
    product = product_repo.load_product_orm(db, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    if user.role != "admin" and (
        not product.owner_id or product.owner_id != user.id
    ):
        raise HTTPException(
            status_code=403,
            detail="Only the product owner or an admin can verify",
        )
    try:
        return _maybe_light(
            product_repo.verify_child(db, product_id, database_id, "databases", user.id),
            light,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Entity not found")


@router.post(
    "/{product_id}/databases/{database_id}/pages/{page_id}/verify",
    response_model=Product,
)
async def verify_database_page(
    product_id: str,
    database_id: str,
    page_id: str,
    db: Session = Depends(get_db),
    user: UserORM = Depends(get_current_user),
    light: bool = Query(False),
):
    """Verify a single page of a database artifact's docs (owner/admin)."""
    product = product_repo.load_product_orm(db, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    if user.role != "admin" and (
        not product.owner_id or product.owner_id != user.id
    ):
        raise HTTPException(
            status_code=403,
            detail="Only the product owner or an admin can verify",
        )
    try:
        return _maybe_light(
            product_repo.verify_page(
                db, product_id, database_id, "databases", page_id, user.id
            ),
            light,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e) or "Entity not found")


# --- Reverse-engineering (202 + poll) ----------------------------------------
@router.post("/{product_id}/databases/{database_id}/generate")
async def generate_database_docs(
    product_id: str,
    database_id: str,
    request_data: GenerateDatabaseDocsRequest,
    db: Session = Depends(get_db),
):
    """Start the MCP reverse-engineering job (202 + job_id)."""
    p_orm = product_repo.load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")
    entity = next((d for d in p_orm.databases if d.id == database_id), None)
    if entity is None:
        raise HTTPException(status_code=404, detail="Database not found")

    # Fork H3/H4: dedup — a repeated POST re-attaches to the in-flight job
    # (same 202 + job_id) instead of racing a second reverse-engineering run.
    job_id, is_new = create_or_get_job(product_id, "database", database_id)
    if is_new:
        # ``language`` (deprecated request field) is passed through as-is;
        # the worker resolves the effective language from the admin setting
        # at job start (api.docgen.jobs._run_docgen_job_async).
        submit_job(
            job_id,
            product_id,
            "database",
            database_id,
            request_data.model,
            request_data.language,
        )
    job = get_job(job_id)
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": (job or {}).get("status", "queued"),
            "entity_type": "database",
            "entity_id": database_id,
            "reused": not is_new,
        },
    )


@router.get("/{product_id}/databases/{database_id}/generate/status")
async def get_database_docgen_status(
    product_id: str,
    database_id: str,
    job_id: str = Query(..., description="Docgen job id returned by the generate endpoint"),
):
    job = get_job(job_id)
    if (
        job is None
        or job.get("product_id") != product_id
        or job.get("entity_type") != "database"
        or job.get("entity_id") != database_id
    ):
        raise HTTPException(status_code=404, detail="Docgen job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": _progress_snapshot(job),
        "indexing_status": job.get("indexing_status", "idle"),
        "indexing_message": job.get("indexing_message", ""),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "docs_chars": job.get("docs_chars"),
    }


__all__ = ["router"]
