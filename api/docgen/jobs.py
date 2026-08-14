"""Async documentation generation (202 + poll) job registry.

Long-running doc generation (git clone, file read, RLM bootstrap) is offloaded
to a dedicated ThreadPoolExecutor. The POST returns 202 + job_id immediately so
the Next.js proxy never holds a long connection (which caused ECONNRESET). Each
worker thread runs its OWN event loop (the docgen pipeline is async) with its
OWN SQLAlchemy session, so the main FastAPI event loop is never blocked and
request-scoped sessions are not shared across threads.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from sqlalchemy.orm import selectinload

from api.db import SessionLocal
from api.models import ProductORM

logger = logging.getLogger(__name__)

# --- Job registry + executor (module-level singletons) -----------------------
_docgen_jobs: Dict[str, Dict[str, Any]] = {}
_DocgenMaxWorkers = int(os.environ.get("DOCGEN_MAX_WORKERS", "2"))
_docgen_executor = ThreadPoolExecutor(
    max_workers=_DocgenMaxWorkers, thread_name_prefix="docgen"
)


def _docgen_prune_old_jobs(max_age_seconds: int = 3600) -> None:
    """Drop finished jobs older than ``max_age_seconds`` to bound memory."""
    cutoff = time.time() - max_age_seconds
    stale = [
        jid for jid, j in _docgen_jobs.items()
        if j.get("finished_at") and j["finished_at"] < cutoff
    ]
    for jid in stale:
        _docgen_jobs.pop(jid, None)


def create_job(product_id: str, entity_type: str, entity_id: str) -> str:
    """Register a new queued job and return its id."""
    _docgen_prune_old_jobs()
    job_id = uuid.uuid4().hex
    _docgen_jobs[job_id] = {
        "job_id": job_id,
        "product_id": product_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "status": "queued",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "docs_chars": None,
    }
    return job_id


def submit_job(
    job_id: str,
    product_id: str,
    entity_type: str,
    entity_id: str,
    model: Optional[str],
    language: str,
) -> None:
    """Submit the job to the worker thread pool."""
    _docgen_executor.submit(
        _run_docgen_job,
        job_id, product_id, entity_type, entity_id, model, language,
    )
    logger.info("Submitted docgen job %s for %s %s", job_id, entity_type, entity_id)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Look up a job by id (or None if unknown)."""
    return _docgen_jobs.get(job_id)


# --- Worker-thread docgen pipeline -------------------------------------------

async def _run_docgen_job_async(
    job_id: str,
    product_id: str,
    entity_type: str,
    entity_id: str,
    model: Optional[str],
    language: str,
) -> None:
    """Async body of a docgen job: loads the entity in a FRESH DB session
    (the request session is closed by now), generates docs, commits, and
    records the outcome in the job registry. Runs inside the worker thread's
    own event loop."""
    job = _docgen_jobs[job_id]
    job["status"] = "running"
    job["indexing_status"] = "idle"
    job["indexing_message"] = "Генерация документации..."
    job["started_at"] = time.time()
    db = SessionLocal()
    try:
        p_orm = (
            db.query(ProductORM)
            .options(
                selectinload(ProductORM.codebases),
                selectinload(ProductORM.specs),
            )
            .filter(ProductORM.id == product_id)
            .first()
        )
        if p_orm is None:
            raise ValueError("Product not found")

        if entity_type == "codebase":
            entity = next((c for c in p_orm.codebases if c.id == entity_id), None)
            if entity is None:
                raise ValueError("Codebase not found")
            from api.docgen.codebase import generate_codebase_docs
            docs = await generate_codebase_docs(
                entity, p_orm, model=model,
                language=language or "ru",
            )
        elif entity_type == "spec":
            entity = next((s for s in p_orm.specs if s.id == entity_id), None)
            if entity is None:
                raise ValueError("Spec not found")
            spec_kind = (getattr(entity, "spec_kind", "") or "").lower()
            from api.docgen.spec import generate_openapi_docs, generate_asyncapi_docs
            if spec_kind == "asyncapi":
                docs = await generate_asyncapi_docs(
                    entity, p_orm, model=model,
                    language=language or "ru",
                )
            else:
                docs = await generate_openapi_docs(
                    entity, p_orm, model=model,
                    language=language or "ru",
                )
        else:
            raise ValueError(f"Unsupported docgen entity_type: {entity_type}")

        db.commit()
        job["status"] = "succeeded"
        # Display is decoupled from the knowledge graph: docs are already
        # committed, so the job is a success regardless of how long cognee
        # cognify takes (it can run 20-30 min and is handed off to the main
        # event loop, NOT gated on the worker thread).
        job["indexing_status"] = "succeeded"
        job["indexing_message"] = "Документы сгенерированы. Граф знаний обновляется в фоне."
        job["finished_at"] = time.time()
        job["docs_chars"] = len(docs or "")
        logger.info("Docgen job %s succeeded for %s %s", job_id, entity_type, entity_id)
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        job["status"] = "failed"
        job["indexing_status"] = "failed"
        job["indexing_message"] = f"Ошибка генерации документации: {e}"
        job["error"] = str(e)
        job["finished_at"] = time.time()
        logger.error("Docgen job %s failed: %s", job_id, e, exc_info=True)
    finally:
        try:
            db.close()
        except Exception:
            pass


def _resolve_indexing_drain_seconds() -> float:
    """Best-effort ceiling for the worker-loop indexing drain."""
    from api.config.timeout import resolve_docgen_indexing_drain_seconds
    return resolve_docgen_indexing_drain_seconds()


def _run_docgen_job(
    job_id: str,
    product_id: str,
    entity_type: str,
    entity_id: str,
    model: Optional[str],
    language: str,
) -> None:
    """Worker-thread entry point: runs the async job in a brand-new event loop
    so the heavy sync work (git clone, file read, RLM) never touches the main
    loop. Cognee indexing is normally handed off to the MAIN event loop via
    ``_index_in_background``; any leftover tasks on the worker loop are drained
    best-effort and NON-FATAL — a drain timeout never marks the job as failed
    because the docs are already committed."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _run_docgen_job_async(job_id, product_id, entity_type, entity_id, model, language)
        )

        async def _drain() -> None:
            pending = [
                t for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
            ]
            if pending:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=_resolve_indexing_drain_seconds(),
                )

        try:
            loop.run_until_complete(_drain())
        except asyncio.TimeoutError:
            logger.warning(
                "Docgen background drain timed out for job %s; non-fatal (docs already committed).",
                job_id,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Docgen background drain error for job %s: %s", job_id, e)
    finally:
        try:
            loop.close()
        except Exception:
            pass
