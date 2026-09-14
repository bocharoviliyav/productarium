"""Async documentation generation (202 + poll) job registry.

Long-running doc generation (git clone, file read, LLM calls) is offloaded to a
dedicated ThreadPoolExecutor. The POST returns 202 + job_id immediately so the
Next.js proxy never holds a long connection (which caused ECONNRESET). Each
worker thread runs its OWN event loop (the docgen pipeline is async) with its
OWN SQLAlchemy session, so the main FastAPI event loop is never blocked and
request-scoped sessions are not shared across threads.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from sqlalchemy.orm import selectinload

from api.db import SessionLocal
from api.docgen._common import JobCancelledError
from api.models import ProductORM

logger = logging.getLogger(__name__)

# --- Job registry + executor (module-level singletons) -----------------------
_docgen_jobs: Dict[str, Dict[str, Any]] = {}
_DocgenMaxWorkers = int(os.environ.get("DOCGEN_MAX_WORKERS", "2"))
_docgen_executor = ThreadPoolExecutor(
    max_workers=_DocgenMaxWorkers, thread_name_prefix="docgen"
)

# --- Dedup + serialization (port of the fork's wiki_generation H3/H4) --------
# The whole check-then-act (prune + scan-for-active + insert) runs under one
# lock, so two concurrent identical POSTs cannot both create a job for the
# same entity. A per-entity lock additionally serializes the generation
# itself: the entity row, the state-dir clone and (from 2.3 on) the
# introspection disk cache are shared mutable state for the same entity even
# across different models/languages.
_JOBS_LOCK = threading.Lock()


def job_key(product_id: str, entity_type: str, entity_id: str) -> Tuple[str, str, str]:
    """Canonical dedup/serialization key: one active job per product entity."""
    return (product_id, entity_type, entity_id)


def request_cancel(job_id: str) -> bool:
    """Flag a queued/running job for cooperative cancellation.

    The worker's pipelines poll the flag at safe checkpoints (between clone /
    planning / per-unit LLM calls) and raise ``JobCancelledError``, which rolls
    the artifact back to its pre-run version WITHOUT appending a new one.
    Returns False for unknown/finished jobs.
    """
    with _JOBS_LOCK:
        job = _docgen_jobs.get(job_id)
        if job is None or job.get("status") not in ("queued", "running"):
            return False
        job["cancel_requested"] = True
        logger.info("Docgen job %s: cancellation requested", job_id)
        return True


# --- Progress model -----------------------------------------------------------
# Phase lifecycle: queued -> cloning -> planning -> sections -> verifying ->
# indexing -> done. spec/database flows map onto the same vocabulary (planning
# for parse/resolve, sections for the single enrichment/introspection unit).
DOCGEN_PHASES = (
    "queued", "cloning", "planning", "sections", "verifying", "indexing", "done",
)


def _new_progress() -> Dict[str, Any]:
    return {
        "phase": "queued",
        "sections_total": None,
        "sections_done": 0,
        "current_section": None,
        "section_durations": {},
    }


def _progress_snapshot(job: Dict[str, Any]) -> Dict[str, Any]:
    """Reader-side copy of the progress block (poll endpoints / UI restore)."""
    prog = dict(job.get("progress") or _new_progress())
    prog["section_durations"] = dict(prog.get("section_durations") or {})
    return prog


def _fmt_duration(seconds: Optional[float]) -> str:
    """tqdm-readable duration: ``42.1s`` under a minute, else ``2m10s``."""
    s = max(0.0, float(seconds or 0.0))
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s"


def _tqdm_bar(job: Dict[str, Any]) -> Optional[Any]:
    """Lazily create the worker-thread progress bar (TTY-only via disable=None)."""
    bar = job.get("_tqdm")
    if bar is not None:
        return bar
    try:
        from tqdm.auto import tqdm  # optional dep guard
    except Exception:
        return None
    bar = tqdm(total=None, dynamic_ncols=True, disable=None, unit="section", desc="docgen")
    job["_tqdm"] = bar
    return bar


def _tqdm_sync(job: Dict[str, Any], prog: Dict[str, Any]) -> None:
    """Reflect the progress state onto the tqdm bar (never raises)."""
    try:
        bar = _tqdm_bar(job)
        if bar is None:
            return
        total = prog.get("sections_total")
        if total and bar.total != total:
            bar.reset(total=total)
        target = prog.get("sections_done") or 0
        if target > bar.n:
            bar.update(target - bar.n)
        else:
            bar.refresh()
        label = prog.get("current_section") or prog.get("phase") or "docgen"
        bar.set_description(f"docgen: {label}")
    except Exception:  # pragma: no cover - progress UI must never break a job
        pass


def _close_progress_bar(job_id: str) -> None:
    """Close the job's tqdm bar (worker-thread teardown)."""
    job = _docgen_jobs.get(job_id)
    if not job:
        return
    bar = job.pop("_tqdm", None)
    if bar is not None:
        try:
            bar.close()
        except Exception:  # pragma: no cover
            pass


def report_progress(job_id: str, **fields: Any) -> None:
    """Merge a progress update into the job (called from the worker thread).

    Accepted fields: ``phase``, ``sections_total``, ``sections_done``,
    ``current_section``, plus the section-completion pair ``section_done`` /
    ``section_seconds``. Emits the tqdm-style INFO lines (phase transitions and
    ``docgen progress [3/7] architecture done in 42.1s (total 2m10s)``) and
    drives the TTY bar. Never raises — broken progress plumbing must not fail
    a generation run.
    """
    job = _docgen_jobs.get(job_id)
    if job is None:
        return
    try:
        prog = job.setdefault("progress", _new_progress())
        old_phase = prog.get("phase")

        done_sid = fields.pop("section_done", None)
        done_seconds = fields.pop("section_seconds", None)

        if done_sid:
            durations = prog.setdefault("section_durations", {})
            try:
                durations[done_sid] = float(done_seconds) if done_seconds is not None else 0.0
            except (TypeError, ValueError):
                durations[done_sid] = 0.0
            if "sections_done" not in fields:
                fields["sections_done"] = len(durations)
            if prog.get("current_section") == done_sid:
                fields["current_section"] = None

        prog.update(fields)

        if done_sid:
            total = prog.get("sections_total")
            started = job.get("started_at")
            elapsed = (time.time() - started) if started else 0.0
            logger.info(
                "docgen progress [%s/%s] %s done in %s (total %s)",
                prog.get("sections_done", "?"),
                total if total else "?",
                done_sid,
                _fmt_duration(done_seconds),
                _fmt_duration(elapsed),
            )

        new_phase = fields.get("phase")
        if new_phase and new_phase != old_phase:
            logger.info("docgen job %s: phase %s -> %s", job_id, old_phase, new_phase)

        _tqdm_sync(job, prog)
    except Exception:  # pragma: no cover - defensive
        logger.debug("docgen progress update failed for %s", job_id, exc_info=True)


def _progress_reporter(job_id: str):
    """Closure passed to generate_* as the optional ``progress`` callback."""
    def report(**fields: Any) -> None:
        report_progress(job_id, **fields)
    return report


# --- Per-entity locks (P1-19/P1-22) -------------------------------------------
class EntityBusyError(RuntimeError):
    """The entity is currently locked by a running docgen job (HTTP 409)."""


# Refcounted registry: an entry lives while at least one holder/waiter is in
# the ``lock_for_entity`` context. The LAST release removes the entry from the
# dict (P1-22) — no periodic cleanup needed because the refcount drops to zero
# exactly when the holder and every waiter have left the context.
_ENTITY_LOCKS: Dict[str, threading.RLock] = {}
_ENTITY_REFCOUNTS: Dict[str, int] = {}
_entity_locks_guard = threading.Lock()

# Short by design: API writes never block a whole docgen job's duration —
# they fail fast with 409 Conflict instead (jobs themselves wait a bit to
# absorb quick back-to-back generations for the same entity).
_ENTITY_LOCK_TIMEOUT = float(os.environ.get("ENTITY_LOCK_TIMEOUT_SECONDS", "5"))


def _entity_key(entity_type: str, entity_id: str) -> str:
    return f"{entity_type}:{entity_id}"


@contextmanager
def lock_for_entity(
    entity_type: str,
    entity_id: str,
    *,
    timeout: Optional[float] = None,
) -> Iterator[None]:
    """Serialize API writes with running docgen jobs on the same entity.

    Raises :class:`EntityBusyError` when the lock cannot be taken within
    ``timeout`` seconds (default ``ENTITY_LOCK_TIMEOUT_SECONDS`` / 5s). The
    registry entry is dropped once the last holder/waiter leaves the context
    (refcount reaches zero), so the dict cannot grow unboundedly.
    """
    if timeout is None:
        timeout = _ENTITY_LOCK_TIMEOUT
    key = _entity_key(entity_type, entity_id)
    with _entity_locks_guard:
        lock = _ENTITY_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ENTITY_LOCKS[key] = lock
        _ENTITY_REFCOUNTS[key] = _ENTITY_REFCOUNTS.get(key, 0) + 1
    try:
        if not lock.acquire(timeout=max(0.0, timeout)):
            raise EntityBusyError(
                f"{entity_type} {entity_id!r} is busy: a docgen job is "
                "already running for this entity. Retry when it completes."
            )
        try:
            yield
        finally:
            lock.release()
    finally:
        with _entity_locks_guard:
            remaining = _ENTITY_REFCOUNTS.get(key, 0) - 1
            if remaining <= 0:
                # Last holder AND all waiters are out — safe to drop the entry.
                _ENTITY_REFCOUNTS.pop(key, None)
                _ENTITY_LOCKS.pop(key, None)
            else:
                _ENTITY_REFCOUNTS[key] = remaining


def _fail_job_with_busy(job_id: str, entity_type: str, entity_id: str) -> None:
    job = _docgen_jobs.get(job_id)
    if job is None or job.get("finished_at"):
        return
    msg = (
        f"{entity_type} {entity_id!r} is busy: another docgen job is already "
        "running for this entity. Retry when it completes."
    )
    job["status"] = "failed"
    job["indexing_status"] = "failed"
    job["indexing_message"] = f"Ошибка: {msg}"
    job["error"] = msg
    job["finished_at"] = time.time()
    logger.warning("Docgen job %s not started: %s", job_id, msg)


def _docgen_prune_old_jobs(max_age_seconds: int = 3600) -> None:
    """Drop finished jobs older than ``max_age_seconds`` to bound memory.

    Precondition: the caller holds ``_JOBS_LOCK`` (``threading.Lock`` is
    non-reentrant, so this helper never acquires it itself — it iterates a
    snapshot and mutates ``_docgen_jobs`` directly).
    """
    cutoff = time.time() - max_age_seconds
    stale = [
        jid for jid, j in _docgen_jobs.items()
        if j.get("finished_at") and j["finished_at"] < cutoff
    ]
    for jid in stale:
        _docgen_jobs.pop(jid, None)


def _register_job(key: Tuple[str, str, str]) -> str:
    """Insert a fresh queued job. Precondition: caller holds ``_JOBS_LOCK``."""
    job_id = uuid.uuid4().hex
    _docgen_jobs[job_id] = {
        "job_id": job_id,
        "product_id": key[0],
        "entity_type": key[1],
        "entity_id": key[2],
        "key": key,
        "status": "queued",
        "progress": _new_progress(),
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "docs_chars": None,
        "cancel_requested": False,
    }
    return job_id


def create_job(product_id: str, entity_type: str, entity_id: str) -> str:
    """Register a NEW queued job unconditionally (no dedup) and return its id.

    Request paths must prefer ``create_or_get_job`` — this entry point exists
    for tests and internal callers that explicitly want a fresh job.
    """
    with _JOBS_LOCK:
        _docgen_prune_old_jobs()
        return _register_job(job_key(product_id, entity_type, entity_id))


def create_or_get_job(
    product_id: str, entity_type: str, entity_id: str
) -> Tuple[str, bool]:
    """Atomic check-then-act job creation with in-flight dedup (fork H3/H4).

    Returns ``(job_id, is_new)``. While a job for the same (product, entity)
    is queued or running, the SAME job id is returned with ``is_new=False``
    and the caller MUST NOT dispatch a second generation run for it — a
    repeated POST just re-attaches to the in-flight job.
    """
    key = job_key(product_id, entity_type, entity_id)
    with _JOBS_LOCK:
        _docgen_prune_old_jobs()
        for job in _docgen_jobs.values():
            if job.get("key") == key and job.get("status") in ("queued", "running"):
                logger.info(
                    "Reusing in-flight docgen job %s for %s %s (duplicate POST)",
                    job["job_id"], entity_type, entity_id,
                )
                return job["job_id"], False
        return _register_job(key), True


def submit_job(
    job_id: str,
    product_id: str,
    entity_type: str,
    entity_id: str,
    model: Optional[str],
    language: Optional[str],
    force_pages: Optional[List[str]] = None,
) -> None:
    """Submit the job to the worker thread pool.

    ``language`` may be None/invalid (the deprecated request field): the
    worker resolves the effective language from the admin
    ``generation.language`` setting when the job STARTS, so a switch in the
    admin panel applies to jobs that were still queued. ``force_pages`` (per-
    page regeneration) limits the run to those pages: codebase units are
    forced past diff-reuse, database pages are merged at persist.
    """
    _docgen_executor.submit(
        _run_docgen_job,
        job_id, product_id, entity_type, entity_id, model, language, force_pages,
    )
    logger.info("Submitted docgen job %s for %s %s", job_id, entity_type, entity_id)


def request_cancel_for_entity(
    product_id: str, entity_type: str, entity_id: str
) -> Optional[str]:
    """Flag the entity's ACTIVE job for cancellation; returns its job_id.

    None when no queued/running job exists for the entity (idempotent cancel).
    """
    key = job_key(product_id, entity_type, entity_id)
    with _JOBS_LOCK:
        for job in _docgen_jobs.values():
            if job.get("key") == key and job.get("status") in ("queued", "running"):
                job["cancel_requested"] = True
                logger.info("Docgen job %s: cancellation requested", job["job_id"])
                return job["job_id"]
    return None


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Look up a job by id (or None if unknown)."""
    return _docgen_jobs.get(job_id)


def active_jobs_for_product(product_id: str) -> list:
    """Snapshot of the product's QUEUED/RUNNING jobs for the UI restore flow.

    Single-process in-memory registry: a backend restart kills the jobs
    anyway, so there is nothing durable to restore FROM — this answers "what
    is running right now" for ``GET /api/products/{id}/docgen/active`` after a
    page reload/navigation.
    """
    out = []
    with _JOBS_LOCK:
        for job in _docgen_jobs.values():
            if job.get("product_id") != product_id:
                continue
            if job.get("status") not in ("queued", "running"):
                continue
            out.append({
                "job_id": job["job_id"],
                "entity_type": job["entity_type"],
                "entity_id": job["entity_id"],
                "status": job["status"],
                "progress": _progress_snapshot(job),
            })
    return out


# --- Worker-thread docgen pipeline -------------------------------------------

async def _run_docgen_job_async(
    job_id: str,
    product_id: str,
    entity_type: str,
    entity_id: str,
    model: Optional[str],
    language: str,
    force_pages: Optional[List[str]] = None,
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
    # The generation language is an admin-managed setting: resolve it HERE
    # (job start), not at request time, so queued jobs honor the latest
    # ``generation.language``. The deprecated request field is ignored when
    # it does not name a supported language.
    from api.prompts import PROMPT_LANGUAGES, get_generation_language

    if language not in PROMPT_LANGUAGES:
        language = get_generation_language()
    # First phase transition is emitted by the generate_* pipeline itself
    # (cloning for codebase, planning for spec/database).
    progress_cb = _progress_reporter(job_id)
    db = SessionLocal()
    try:
        # 2.2 embedder preflight: one short probe BEFORE the (potentially
        # hours-long) pipeline — a run whose output could never be indexed
        # (embedder down / model missing / dimension changed) must fail in
        # seconds with a readable EMBEDDER_ERROR reason, not lose recall
        # silently at the very end. Skipped automatically on non-pgvector
        # installs (SQLite fallback, hermetic tests); internal preflight
        # errors never kill the job — only the classified diagnosis does.
        try:
            from api.memory.preflight import (
                EMBEDDER_ERROR_PREFIX,
                EmbedderUnavailable,
                preflight_embedder,
            )

            await preflight_embedder(product_id)
        except EmbedderUnavailable as e:
            raise ValueError(f"{EMBEDDER_ERROR_PREFIX}{e}") from None
        except Exception as e:  # pragma: no cover - preflight plumbing
            logger.warning("Embedder preflight skipped (unexpected error): %s", e)
        # Fork H4: the worker entry point (_run_docgen_job) already holds the
        # per-entity lock (refcounted lock_for_entity context manager) for the
        # whole job, so this async body runs serialized — no inner lock here.
        p_orm = (
            db.query(ProductORM)
            .options(
                selectinload(ProductORM.codebases),
                selectinload(ProductORM.specs),
                selectinload(ProductORM.databases),
            )
            .filter(ProductORM.id == product_id)
            .first()
        )
        if p_orm is None:
            raise ValueError("Product not found")

        # Cooperative cancellation: pipelines poll this closure at safe
        # checkpoints; a queued job that was cancelled before start raises at
        # its first checkpoint (before any expensive work).
        def should_cancel() -> bool:
            return bool(job.get("cancel_requested"))

        collections = {
            "codebase": p_orm.codebases,
            "spec": p_orm.specs,
            "database": p_orm.databases,
        }
        if entity_type not in collections:
            raise ValueError(f"Unsupported docgen entity_type: {entity_type}")
        entity = next(
            (e for e in collections[entity_type] if e.id == entity_id), None
        )
        if entity is None:
            raise ValueError(f"{entity_type.capitalize()} not found")

        # Vault-style versioning: bootstrap a v1 snapshot of legacy docs BEFORE
        # the run (committed eagerly — survives a failed run), then append the
        # generated result as a NEW immutable version in the final commit.
        from api.repositories import doc_version_repo

        doc_version_repo.ensure_baseline_version(db, entity_type, entity)
        db.commit()

        if entity_type == "codebase":
            from api.docgen.codebase import generate_codebase_docs
            docs = await generate_codebase_docs(
                entity, p_orm, model=model,
                language=language or "ru",
                progress=progress_cb,
                should_cancel=should_cancel,
                force_units=force_pages,
            )
        elif entity_type == "spec":
            # SpecORM.kind is the real column ("openapi" | "asyncapi").
            spec_kind = (getattr(entity, "kind", None) or "openapi").lower()
            from api.docgen.spec import generate_openapi_docs, generate_asyncapi_docs
            if spec_kind == "asyncapi":
                docs = await generate_asyncapi_docs(
                    entity, p_orm, model=model,
                    language=language or "ru",
                    progress=progress_cb,
                    should_cancel=should_cancel,
                )
            else:
                docs = await generate_openapi_docs(
                    entity, p_orm, model=model,
                    language=language or "ru",
                    progress=progress_cb,
                    should_cancel=should_cancel,
                )
        else:
            from api.docgen.database import generate_database_docs
            docs = await generate_database_docs(
                entity, p_orm, model=model,
                language=language or "ru",
                progress=progress_cb,
                should_cancel=should_cancel,
                force_pages=force_pages,
            )

        doc_version_repo.append_version(
            db, entity_type, entity,
            source="generate", model=model, job_id=job_id,
        )
        db.commit()
        job["status"] = "succeeded"
        # Display is decoupled from memory indexing: docs are already committed,
        # so the job is a success regardless of how long background indexing
        # takes (it is handed off to the main event loop, NOT gated on the
        # worker thread).
        job["indexing_status"] = "succeeded"
        job["indexing_message"] = "Документы сгенерированы. Индексация обновляется в фоне."
        job["finished_at"] = time.time()
        job["docs_chars"] = len(docs or "")
        report_progress(job_id, phase="done")
        logger.info("Docgen job %s succeeded for %s %s", job_id, entity_type, entity_id)
    except JobCancelledError:
        # Cancellation: no version is appended; the artifact is restored to its
        # pre-run (current) version — mid-run checkpoints may have committed
        # partial docs onto the row (see _checkpoint_partial_docs).
        try:
            db.rollback()
            p2 = (
                db.query(ProductORM)
                .options(
                    selectinload(ProductORM.codebases),
                    selectinload(ProductORM.specs),
                    selectinload(ProductORM.databases),
                )
                .filter(ProductORM.id == product_id)
                .first()
            )
            coll = ({
                "codebase": p2.codebases,
                "spec": p2.specs,
                "database": p2.databases,
            }.get(entity_type) or []) if p2 is not None else []
            ent = next((e for e in coll if e.id == entity_id), None)
            if ent is not None:
                doc_version_repo.restore_entity_from_current_version(
                    db, entity_type, ent
                )
                db.commit()
        except Exception:  # pragma: no cover - restore is best-effort
            try:
                db.rollback()
            except Exception:
                pass
            logger.warning(
                "Post-cancel restore failed for %s %s; partial docs may remain "
                "until the next generation",
                entity_type, entity_id, exc_info=True,
            )
        job["status"] = "cancelled"
        job["indexing_status"] = "cancelled"
        job["indexing_message"] = "Генерация отменена; предыдущая версия восстановлена."
        job["error"] = None
        job["finished_at"] = time.time()
        logger.info("Docgen job %s cancelled for %s %s", job_id, entity_type, entity_id)
    except ValueError as e:
        try:
            db.rollback()
        except Exception:
            pass
        job["status"] = "failed"
        job["indexing_status"] = "failed"
        # Controlled validation messages from our own docgen layer (entity
        # not found, unusable MCP surface, LLM fully unavailable) — safe to
        # surface to the polling client.
        job["indexing_message"] = f"Ошибка генерации документации: {e}"
        job["error"] = str(e)
        job["finished_at"] = time.time()
        logger.warning("Docgen job %s failed: %s", job_id, e)
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        job["status"] = "failed"
        job["indexing_status"] = "failed"
        # Unexpected exceptions may embed local FS paths / clone stderr /
        # upstream details — generic client message, full context in the
        # server log only (review #5).
        job["indexing_message"] = (
            "Ошибка генерации документации (подробности в логах сервера)"
        )
        job["error"] = f"Generation failed ({type(e).__name__}); see server logs"
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
    force_pages: Optional[List[str]] = None,
) -> None:
    """Worker-thread entry point: runs the async job in a brand-new event loop
    so the heavy sync work (git clone, file read, LLM calls) never touches the
    main loop. Memory indexing is normally handed off to the MAIN event loop
    via ``_index_in_background``; any leftover tasks on the worker loop are
    drained best-effort and NON-FATAL — a drain timeout never marks the job as
    failed because the docs are already committed."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # P1-19: hold the entity lock for the whole job so API writes to the
        # same entity fail fast (409) instead of racing the final commit.
        try:
            with lock_for_entity(entity_type, entity_id):
                loop.run_until_complete(
                    _run_docgen_job_async(
                        job_id, product_id, entity_type, entity_id, model, language,
                        force_pages,
                    )
                )
        except EntityBusyError:
            _fail_job_with_busy(job_id, entity_type, entity_id)

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
        _close_progress_bar(job_id)
        try:
            loop.close()
        except Exception:
            pass
