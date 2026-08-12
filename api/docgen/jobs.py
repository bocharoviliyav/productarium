"""Async artifact documentation generation (202 + poll) job registry.

Long-running artifact doc generation (git clone, file read, RLM bootstrap)
is offloaded to a dedicated ThreadPoolExecutor. The POST returns 202 + job_id
immediately so the Next.js proxy never holds a long connection (which caused
ECONNRESET). Each worker thread runs its OWN event loop (the docgen pipeline
is async) with its OWN SQLAlchemy session, so the main FastAPI event loop is
never blocked and request-scoped sessions are not shared across threads.

Extracted from the former ``api/api.py`` monolith; the job registry is a
module-level singleton so the router can submit/poll and the worker threads
can update job state.
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
from api.models import ArtifactORM, ProductORM

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


def create_job(product_id: str, artifact_id: str) -> str:
    """Register a new queued job and return its id."""
    _docgen_prune_old_jobs()
    job_id = uuid.uuid4().hex
    _docgen_jobs[job_id] = {
        "job_id": job_id,
        "product_id": product_id,
        "artifact_id": artifact_id,
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
    artifact_id: str,
    provider: str,
    model: Optional[str],
    language: str,
) -> None:
    """Submit the job to the worker thread pool."""
    _docgen_executor.submit(
        _run_docgen_job,
        job_id,
        product_id,
        artifact_id,
        provider,
        model,
        language,
    )
    logger.info("Submitted docgen job %s for artifact %s", job_id, artifact_id)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Look up a job by id (or None if unknown)."""
    return _docgen_jobs.get(job_id)


# --- Worker-thread docgen pipeline -------------------------------------------

async def _run_docgen_job_async(
    job_id: str,
    product_id: str,
    artifact_id: str,
    provider: str,
    model: Optional[str],
    language: str,
) -> None:
    """Async body of a docgen job: loads the artifact in a FRESH DB session
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
            .options(selectinload(ProductORM.artifacts))
            .filter(ProductORM.id == product_id)
            .first()
        )
        if p_orm is None:
            raise ValueError("Product not found")
        artifact = next((a for a in p_orm.artifacts if a.id == artifact_id), None)
        if artifact is None:
            raise ValueError("Artifact not found")
        from api.docgen import generate_artifact_documentation
        # generate_artifact_documentation writes generated_docs/pages onto the
        # artifact ORM in-place; persist them in the same transaction.
        # Pass provider/model straight through (None when unset) so
        # _resolve_docgen_model resolves the admin-configured alias from the
        # DB settings store instead of an env-default shadowing it.
        docs = await generate_artifact_documentation(
            artifact,
            p_orm,
            provider=provider,
            model=model,
            language=language or "ru",
        )
        db.commit()
        job["status"] = "succeeded"
        # Display is decoupled from the knowledge graph: docs are already
        # committed, so the job is a success regardless of how long cognee
        # cognify takes (it can run 20-30 min and is handed off to the main
        # event loop, NOT gated on the worker thread). Indexing continues in
        # the background; its own outcome surfaces via cognee logs / admin
        # reindex, never as a docgen job failure.
        job["indexing_status"] = "succeeded"
        job["indexing_message"] = "Документы сгенерированы. Граф знаний обновляется в фоне."
        job["finished_at"] = time.time()
        job["docs_chars"] = len(docs or "")
        logger.info("Docgen job %s succeeded for artifact %s; cognee indexing handed off to background.", job_id, artifact_id)
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
    """Best-effort ceiling for the worker-loop indexing drain.

    Cognee indexing is normally handed off to the MAIN event loop (see
    ``_index_in_background``) so it survives the worker loop teardown. This drain
    only catches tasks that were NOT handed off (no main loop at startup, or the
    handoff failed). It is NON-FATAL: a timeout here never marks the job as
    failed — docs are already committed.

    Resolved through the central timeout config (admin > env > default). The
    default DERIVES from the cognee cognify timeout so a leftover cognify task
    (one that wasn't handed off to the main loop) gets the FULL cognify budget
    instead of being cancelled at a fixed 30s -- which previously dropped the
    connection mid-graph-build (the user-flagged api.py-vs-cognify conflict).
    Floor 5s so a typo can't make the drain instantaneous.
    """
    from api.timeout_config import resolve_docgen_indexing_drain_seconds
    return resolve_docgen_indexing_drain_seconds()


def _run_docgen_job(
    job_id: str,
    product_id: str,
    artifact_id: str,
    provider: str,
    model: Optional[str],
    language: str,
) -> None:
    """Worker-thread entry point: runs the async job in a brand-new event loop
    so the heavy sync work (git clone, file read, RLM) never touches the main
    loop. Cognee indexing is normally handed off to the MAIN event loop via
    ``_index_in_background`` (so a 20-30 min cognify survives this loop closing);
    any leftover tasks on the worker loop are drained best-effort and NON-FATAL
    — a drain timeout never marks the job as failed because the docs are already
    committed and display is decoupled from the knowledge graph."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    job = _docgen_jobs.get(job_id)
    try:
        loop.run_until_complete(
            _run_docgen_job_async(job_id, product_id, artifact_id, provider, model, language)
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
            # Non-fatal: docs are already committed. Indexing either was handed
            # off to the main loop (the common path) or this drain caught a
            # leftover task that couldn't finish in time — either way the job's
            # success / indexing_status (set in _run_docgen_job_async) stands.
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
