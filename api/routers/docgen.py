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
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db import get_db
from api.docgen.jobs import create_job, get_job, submit_job
from api.repositories import product_repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/products", tags=["docgen"])


class GenerateDocRequest(BaseModel):
    model: Optional[str] = None
    language: Optional[str] = "ru"


def _start_generate(
    db: Session, product_id: str, entity_type: str, entity_id: str,
    request_data: GenerateDocRequest,
) -> JSONResponse:
    p_orm = product_repo.load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")

    collection = p_orm.codebases if entity_type == "codebase" else p_orm.specs
    entity = next((e for e in collection if e.id == entity_id), None)
    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.capitalize()} not found")

    job_id = create_job(product_id, entity_type, entity_id)
    submit_job(
        job_id, product_id, entity_type, entity_id,
        request_data.model, request_data.language or "ru",
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued", "entity_type": entity_type, "entity_id": entity_id},
    )


@router.post("/{product_id}/codebases/{codebase_id}/generate")
async def generate_codebase_docs(
    product_id: str, codebase_id: str,
    request_data: GenerateDocRequest, db: Session = Depends(get_db),
):
    return _start_generate(db, product_id, "codebase", codebase_id, request_data)


@router.get("/{product_id}/codebases/{codebase_id}/generate/status")
async def get_codebase_docgen_status(
    product_id: str, codebase_id: str,
    job_id: str = Query(..., description="Docgen job id returned by the generate endpoint"),
):
    return _get_status(product_id, "codebase", codebase_id, job_id)


@router.post("/{product_id}/specs/{spec_id}/generate")
async def generate_spec_docs(
    product_id: str, spec_id: str,
    request_data: GenerateDocRequest, db: Session = Depends(get_db),
):
    return _start_generate(db, product_id, "spec", spec_id, request_data)


@router.get("/{product_id}/specs/{spec_id}/generate/status")
async def get_spec_docgen_status(
    product_id: str, spec_id: str,
    job_id: str = Query(..., description="Docgen job id returned by the generate endpoint"),
):
    return _get_status(product_id, "spec", spec_id, job_id)


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
        "indexing_status": job.get("indexing_status", "idle"),
        "indexing_message": job.get("indexing_message", ""),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "docs_chars": job.get("docs_chars"),
    }


__all__ = ["router"]
