"""Docgen router — async artifact documentation generation (202 + poll).

Endpoints (prefix ``/api/products``, tags ``docgen``):
- ``POST /api/products/{product_id}/artifacts/{artifact_id}/generate``
- ``GET  /api/products/{product_id}/artifacts/{artifact_id}/generate/status``

The POST returns 202 + job_id immediately; the heavy work runs in a worker
thread (see ``api.docgen.jobs``). The status endpoint polls the job registry.
"""

from __future__ import annotations

import logging
import os
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
    # Provider/model default to the OpenAI-compatible local server (LM Studio /
    # llama.cpp / vLLM). Override via env or per-request. Ollama is still
    # supported when explicitly requested, but is no longer the default because
    # the default local server in this deployment is LM Studio (:1234).
    provider: Optional[str] = os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local")
    model: Optional[str] = os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b")
    language: Optional[str] = "ru"


@router.post("/{product_id}/artifacts/{artifact_id}/generate")
async def generate_artifact_docs(
    product_id: str,
    artifact_id: str,
    request_data: GenerateDocRequest,
    db: Session = Depends(get_db),
):
    """Start asynchronous artifact documentation generation.

    Returns ``202`` with ``{job_id, status: "queued"}`` immediately. The heavy
    work (git clone, file read, RLM) runs in a worker thread with its own event
    loop, so the main FastAPI loop is never blocked and the Next.js proxy never
    sees a long-held connection (which previously caused ECONNRESET). Poll
    ``GET .../generate/status?job_id=...`` for the result.
    """
    p_orm = product_repo.load_product_orm(db, product_id)
    if p_orm is None:
        raise HTTPException(status_code=404, detail="Product not found")
    artifact = next((a for a in p_orm.artifacts if a.id == artifact_id), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    job_id = create_job(product_id, artifact_id)
    submit_job(
        job_id,
        product_id,
        artifact_id,
        request_data.provider or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local"),
        request_data.model,
        request_data.language or "ru",
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued", "artifact_id": artifact_id},
    )


@router.get("/{product_id}/artifacts/{artifact_id}/generate/status")
async def get_docgen_status(
    product_id: str,
    artifact_id: str,
    job_id: str = Query(..., description="Docgen job id returned by the generate endpoint"),
):
    """Poll the status of an asynchronous docgen job and cognee indexing."""
    job = get_job(job_id)
    if (
        job is None
        or job.get("product_id") != product_id
        or job.get("artifact_id") != artifact_id
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
