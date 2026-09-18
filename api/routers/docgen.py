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
from sqlalchemy.orm import Session

from api.auth.deps import get_current_user, require_product_access
from api.db import get_db
from api.docgen.jobs import (
    _progress_snapshot,
    active_jobs_for_product,
    create_or_get_job,
    get_job,
    request_cancel_for_entity,
    submit_job,
)
from api.models import ProductORM, UserORM
from api.repositories import product_repo
from api.utils.rate_limit import enforce_user_rate_limit

logger = logging.getLogger(__name__)

# Generation is an authenticated action: starting a (potentially expensive)
# docgen job and polling its status both require a session (a no-op when
# AUTH_PROVIDER=none).
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

    # Access (rw) is enforced by the endpoint's require_product_access dep.
    p_orm = product_repo.load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")

    collection = p_orm.codebases if entity_type == "codebase" else p_orm.specs
    entity = next((e for e in collection if e.id == entity_id), None)
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


@router.post("/{product_id}/codebases/{codebase_id}/generate")
async def generate_codebase_docs(
    product_id: str, codebase_id: str,
    request_data: GenerateDocRequest, db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_access("rw")),
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
    _product: ProductORM = Depends(require_product_access("rw")),
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
}
_CANCEL_SEGMENTS = _SEGMENT_TO_TYPE


@router.post("/{product_id}/{segment}/{entity_id}/generate/cancel")
async def cancel_docgen_generation(
    product_id: str, segment: str, entity_id: str,
    _product: ProductORM = Depends(require_product_access("rw")),
):
    """Cooperatively cancel the entity's ACTIVE docgen job.

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
    _product: ProductORM = Depends(require_product_access("rw")),
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
    collection = (
        p_orm.codebases if entity_type == "codebase" else p_orm.databases
    )
    entity = next((e for e in collection if e.id == entity_id), None)
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


@router.get("/{product_id}/docgen/active")
async def get_active_docgen_jobs(
    product_id: str, db: Session = Depends(get_db),
):
    """Active (queued/running) docgen jobs for the product.

    Lets the UI restore in-flight generation animations after a page
    reload/navigation (the job registry is in-memory and single-process: a
    backend restart kills the jobs, so there is nothing durable to poll).
    """
    if product_repo.load_product_orm(db, product_id) is None:
        raise HTTPException(status_code=404, detail="Product not found")
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
    }


__all__ = ["router"]
