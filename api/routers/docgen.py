"""Docgen router — async documentation generation (202 + poll).

Endpoints (prefix ``/api/products``, tags ``docgen``):
- ``POST /api/products/{product_id}/codebases/{codebase_id}/generate``
- ``GET  /api/products/{product_id}/codebases/{codebase_id}/generate/status``
- ``POST /api/products/{product_id}/specs/{spec_id}/generate``
- ``GET  /api/products/{product_id}/specs/{spec_id}/generate/status``

The POST returns 202 + job_id immediately; the heavy work runs in a worker
thread (see ``api.docgen.jobs``). The status endpoint polls the job registry.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.auth.deps import (
    get_current_user,
    require_product_access,
    require_product_permission,
)
from api.db import get_db
from api.docgen.jobs import (
    _progress_snapshot,
    active_jobs_for_product,
    create_or_get_job,
    get_job,
    request_cancel_for_entity,
    submit_detect_job,
    submit_incremental_job,
    submit_job,
)
from api.models import HldORM, ProductORM, UserORM
from api.repositories import doc_version_repo, product_repo
from api.utils.rate_limit import enforce_user_rate_limit

logger = logging.getLogger(__name__)

# Generation is an authenticated action: starting a (potentially expensive)
# docgen job and polling its status both require a session (a no-op when
# AUTH_PROVIDER=none). SPEC-001: generate/cancel/regenerate need the docs
# permission; status/page reads need plain visibility.
router = APIRouter(
    prefix="/api/products",
    tags=["docgen"],
    dependencies=[Depends(get_current_user)],
)


class GenerateDocRequest(BaseModel):
    model: Optional[str] = None
    # DEPRECATED no-op: the generation language is controlled by the admin
    # ``generation.language`` setting (see api.prompts.get_generation_language)
    # and resolved when the job starts. The field is kept so older clients
    # sending it are not rejected.
    language: Optional[str] = None
    # Regenerate endpoint only: also force every descendant of the target
    # page (branch regeneration). Plain /generate ignores it.
    branch: bool = False


def _branch_page_ids(pages: Dict[str, Any], root: str) -> List[str]:
    """The page plus all descendants, via BFS over parent links."""
    children: Dict[str, List[str]] = {}
    for pid, page in pages.items():
        parent = (page or {}).get("parent")
        if parent and parent != pid:
            children.setdefault(parent, []).append(pid)
    out: List[str] = []
    seen = {root}
    queue = [root]
    while queue:
        pid = queue.pop(0)
        out.append(pid)
        for child in children.get(pid, ()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return out


def _submit_202(
    job_id: str, entity_type: str, entity_id: str, is_new: bool
) -> JSONResponse:
    """Shared 202 body for the generate/detect/incremental endpoints."""
    job = get_job(job_id)
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": (job or {}).get("status", "queued"),
            "entity_type": entity_type,
            "entity_id": entity_id,
            "reused": not is_new,
        },
    )


def _start_generate(
    db: Session, product_id: str, entity_type: str, entity_id: str,
    request_data: GenerateDocRequest, user_id: str,
    force_pages: Optional[List[str]] = None,
) -> JSONResponse:
    # P1-17: per-user token bucket on the expensive generate operation.
    # Checked before any DB work so an exhausted client is shed cheaply.
    enforce_user_rate_limit(
        user_id,
        setting_key="rate.docgen.per_user_hour",
        env_name="RATE_DOCGEN_PER_USER_HOUR",
        default_per_minute=10,
        window_seconds=3600.0,
    )

    # The docs/manage permission is enforced by the endpoint's dependency.
    p_orm = product_repo.load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")

    if entity_type == "hld" and p_orm.hld is None:
        # Pseudo-entity: auto-create the single row on first generate (the
        # worker loads it in a fresh session — commit BEFORE the job starts).
        try:
            p_orm.hld = HldORM(id=f"hld_{product_id}", product_id=product_id)
            db.commit()
        except IntegrityError:
            # Lost a concurrent first-generate race: adopt the winner's
            # row and let create_or_get_job dedup the in-flight job.
            db.rollback()
            p_orm.hld = db.query(HldORM).filter(
                HldORM.id == f"hld_{product_id}"
            ).one_or_none()
    entity = product_repo.find_entity(p_orm, entity_type, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.capitalize()} not found")

    # Fork H3/H4: a repeated POST for an entity with an in-flight job re-attaches
    # to that job (same 202 + job_id) instead of racing a duplicate generation.
    job_id, is_new = create_or_get_job(product_id, entity_type, entity_id)
    if is_new:
        # ``language`` (deprecated request field) is passed through as-is;
        # the worker resolves the effective language from the admin setting
        # at job start (api.docgen.jobs._run_docgen_job_async).
        submit_job(
            job_id, product_id, entity_type, entity_id,
            request_data.model, request_data.language, force_pages=force_pages,
        )
    return _submit_202(job_id, entity_type, entity_id, is_new)


@router.post("/{product_id}/codebases/{codebase_id}/generate")
async def generate_codebase_docs(
    product_id: str, codebase_id: str,
    request_data: GenerateDocRequest, db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_permission("docs")),
    user: UserORM = Depends(get_current_user),
):
    return _start_generate(db, product_id, "codebase", codebase_id, request_data, user.id)


@router.get("/{product_id}/codebases/{codebase_id}/generate/status")
async def get_codebase_docgen_status(
    product_id: str, codebase_id: str,
    job_id: str = Query(..., description="Docgen job id returned by the generate endpoint"),
    _product: ProductORM = Depends(require_product_access("ro")),
):
    return _get_status(product_id, "codebase", codebase_id, job_id)


@router.post("/{product_id}/specs/{spec_id}/generate")
async def generate_spec_docs(
    product_id: str, spec_id: str,
    request_data: GenerateDocRequest, db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_permission("docs")),
    user: UserORM = Depends(get_current_user),
):
    return _start_generate(db, product_id, "spec", spec_id, request_data, user.id)


@router.get("/{product_id}/specs/{spec_id}/generate/status")
async def get_spec_docgen_status(
    product_id: str, spec_id: str,
    job_id: str = Query(..., description="Docgen job id returned by the generate endpoint"),
    _product: ProductORM = Depends(require_product_access("ro")),
):
    return _get_status(product_id, "spec", spec_id, job_id)


_SEGMENT_TO_TYPE = {
    "codebases": "codebase",
    "specs": "spec",
    "databases": "database",
    "hlds": "hld",
    # Singular alias: the UI's entityPath("hld") builds sub-resource URLs
    # with the singular segment (cancel, page-content, versions).
    "hld": "hld",
}
_CANCEL_SEGMENTS = _SEGMENT_TO_TYPE


@router.post("/{product_id}/{segment}/{entity_id}/generate/cancel")
async def cancel_docgen_generation(
    product_id: str, segment: str, entity_id: str,
    _product: ProductORM = Depends(require_product_permission("docs")),
):
    """Cooperatively cancel the entity's ACTIVE docgen job (docs access —
    whoever may start a run may stop it).

    The worker aborts at its next checkpoint: NO new doc version is appended
    and the artifact is restored to its current version. Idempotent —
    ``cancelled: false`` when there was nothing to stop.
    """
    entity_type = _CANCEL_SEGMENTS.get(segment)
    if entity_type is None:
        raise HTTPException(status_code=404, detail="Unknown artifact type")
    job_id = request_cancel_for_entity(product_id, entity_type, entity_id)
    if job_id is None:
        return {"cancelled": False, "job_id": None}
    return {"cancelled": True, "job_id": job_id}


@router.post(
    "/{product_id}/{segment}/{entity_id}/pages/{page_id}/regenerate"
)
async def regenerate_docgen_page(
    product_id: str, segment: str, entity_id: str, page_id: str,
    request_data: GenerateDocRequest, db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_permission("docs")),
    user: UserORM = Depends(get_current_user),
):
    """Per-page regeneration (codebase / database artifacts).

    Re-runs generation for ONE page — or, with ``branch: true``, for the page
    and all its descendants: unchanged codebase units still come back via
    diff-reuse, the database flow merges only the forced pages at persist.
    The judge issues stored on each page feed the prompt as reviewer notes.
    The result lands as a NEW doc version (like any generation).
    """
    entity_type = _SEGMENT_TO_TYPE.get(segment)
    if entity_type not in ("codebase", "database"):
        raise HTTPException(
            status_code=400,
            detail="Per-page regeneration is only available for codebases and databases",
        )
    p_orm = product_repo.load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")
    entity = product_repo.find_entity(p_orm, entity_type, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.capitalize()} not found")
    if page_id not in (entity.pages or {}):
        raise HTTPException(status_code=404, detail="Page not found")
    force_pages = (
        _branch_page_ids(entity.pages or {}, page_id)
        if request_data.branch else [page_id]
    )
    return _start_generate(
        db, product_id, entity_type, entity_id, request_data, user.id,
        force_pages=force_pages,
    )


@router.get("/{product_id}/{segment}/{entity_id}/pages/{page_id}")
async def get_docgen_page(
    product_id: str, segment: str, entity_id: str, page_id: str,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("ro")),
    version: Optional[int] = Query(None, ge=1, description="Read from this archived doc version"),
):
    """Full page body (content + provenance + verify flags).

    Companion to the ``light`` product payloads: product endpoints return
    page metadata only, so the viewer lazy-loads the open page here —
    optionally from an archived doc version (``?version=N``).
    """
    entity_type = _SEGMENT_TO_TYPE.get(segment)
    if entity_type not in ("codebase", "database", "hld"):
        raise HTTPException(
            status_code=400,
            detail="Page content is only available for codebases, databases and HLD",
        )
    if version is None:
        p_orm = product_repo.load_product_orm(db, product_id)
        if p_orm is None:
            raise HTTPException(status_code=404, detail="Product not found")
        entity = product_repo.find_entity(p_orm, entity_type, entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail=f"{entity_type.capitalize()} not found")
        page, source = (entity.pages or {}).get(page_id), "current"
    else:
        row = doc_version_repo.get_version(db, entity_type, entity_id, version)
        # Version rows key on (entity_type, entity_id) alone — reject another
        # product's row (hld_{product_id} ids are deterministic/guessable).
        if row is None or row.product_id != product_id:
            raise HTTPException(status_code=404, detail="Version not found")
        page, source = (row.pages or {}).get(page_id), "archive"
    if not isinstance(page, dict):
        raise HTTPException(status_code=404, detail="Page not found")
    return {"page": page, "source": source}


# SPEC-010: detect / incremental update are codebase/spec/database only
# (HLD has no external source to diff against).
_DETECT_TYPES = ("codebase", "spec", "database")


def _detect_rate_limit(user_id: str) -> None:
    """Same per-user bucket as generate: a database detect re-walks the catalog."""
    enforce_user_rate_limit(
        user_id,
        setting_key="rate.docgen.per_user_hour",
        env_name="RATE_DOCGEN_PER_USER_HOUR",
        default_per_minute=10,
        window_seconds=3600.0,
    )


def _require_detectable_entity(
    db: Session, product_id: str, entity_type: str, entity_id: str
) -> None:
    """404-check the detect/incremental target (worker reloads it itself)."""
    p_orm = product_repo.load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")
    if product_repo.find_entity(p_orm, entity_type, entity_id) is None:
        raise HTTPException(status_code=404, detail=f"{entity_type.capitalize()} not found")


@router.post("/{product_id}/{segment}/{entity_id}/detect")
async def detect_docgen_changes(
    product_id: str, segment: str, entity_id: str,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_permission("docs")),
    user: UserORM = Depends(get_current_user),
):
    """SPEC-010 change detection (202 + poll): compares the stored source
    baseline with the live source (git remote tip / fresh catalog walk /
    content hash) and returns an impact plan in the job ``result``.
    """
    entity_type = _SEGMENT_TO_TYPE.get(segment)
    if entity_type not in _DETECT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Change detection is only available for codebases, specs and databases",
        )
    _detect_rate_limit(user.id)
    _require_detectable_entity(db, product_id, entity_type, entity_id)
    job_id, is_new = submit_detect_job(product_id, entity_type, entity_id)
    return _submit_202(job_id, entity_type, entity_id, is_new)


@router.get("/{product_id}/{segment}/{entity_id}/detect/status")
async def get_docgen_detect_status(
    product_id: str, segment: str, entity_id: str,
    job_id: str = Query(..., description="Detect job id returned by the detect endpoint"),
    _product: ProductORM = Depends(require_product_access("ro")),
):
    entity_type = _SEGMENT_TO_TYPE.get(segment)
    if entity_type not in _DETECT_TYPES:
        raise HTTPException(status_code=400, detail="Unknown artifact type")
    return _get_status(product_id, f"detect:{entity_type}", entity_id, job_id)


@router.post("/{product_id}/{segment}/{entity_id}/incremental")
async def incremental_docgen_update(
    product_id: str, segment: str, entity_id: str,
    request_data: GenerateDocRequest, db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_permission("docs")),
    user: UserORM = Depends(get_current_user),
):
    """SPEC-010 incremental update (202 + poll, same status endpoint as
    generate): the worker re-detects at run start — unchanged → success
    WITHOUT a new version; changed → only the impacted pages regenerate
    (empty impact / spec → full regeneration). The result summary rides in
    the job ``result``. Full regeneration via ``/generate`` stays available.
    """
    entity_type = _SEGMENT_TO_TYPE.get(segment)
    if entity_type not in _DETECT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Incremental updates are only available for codebases, specs and databases",
        )
    _detect_rate_limit(user.id)
    _require_detectable_entity(db, product_id, entity_type, entity_id)
    job_id, is_new = submit_incremental_job(
        product_id, entity_type, entity_id,
        request_data.model, request_data.language,
    )
    return _submit_202(job_id, entity_type, entity_id, is_new)


@router.get("/{product_id}/docgen/active")
async def get_active_docgen_jobs(
    product_id: str,
    _product: ProductORM = Depends(require_product_access("ro")),
):
    """Active (queued/running) docgen jobs for the product (visible access).

    Lets the UI restore in-flight generation animations after a page
    reload/navigation (the job registry is in-memory and single-process: a
    backend restart kills the jobs, so there is nothing durable to poll).
    """
    return active_jobs_for_product(product_id)


def _get_status(product_id: str, entity_type: str, entity_id: str, job_id: str) -> dict:
    job = get_job(job_id)
    if (
        job is None
        or job.get("product_id") != product_id
        or job.get("entity_type") != entity_type
        or job.get("entity_id") != entity_id
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
        "result": job.get("result"),
    }


__all__ = ["router"]
