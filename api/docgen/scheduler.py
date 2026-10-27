"""SPEC-009 scheduled docgen — cron-driven automatic documentation runs.

One APScheduler ``AsyncIOScheduler`` runs a single minute-interval tick
(started in the FastAPI lifespan). The tick loads enabled ``docgen_schedules``
rows and fires the ones whose cron expression is due, evaluated against the
schedule's last run — a missed window fires once on the next tick and is
never queued or accumulated (spec non-goal).

Per due schedule:
- entity busy (any active job in the docgen registry, incl. a manual run) →
  ``skipped_busy`` recorded on the schedule; the running job is untouched;
- free → an ordinary docgen job via the existing machinery (dedup, per-entity
  lock, budgets unchanged), labelled ``source="schedule"``; incremental mode
  when a SPEC-010 baseline exists, else a full generation.

Job outcomes are reconciled into ``last_result`` (ok|error|cancelled) by the
next tick that sees the job finished.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from api.db import SessionLocal
from api.docgen import jobs
from api.models import DocgenScheduleORM, ProductORM
from api.repositories import product_repo

logger = logging.getLogger(__name__)

TICK_SECONDS = 60
SCHEDULE_ENTITIES = ("codebase", "database")

_scheduler: Optional[AsyncIOScheduler] = None
# schedule.id -> job_id of runs launched by the scheduler whose outcome has
# not yet been reconciled into last_result by a later tick.
_pending: Dict[str, str] = {}

# Terminal job statuses (registry) -> schedule last_result values.
_JOB_RESULTS = {"succeeded": "ok", "failed": "error", "cancelled": "cancelled"}


def parse_cron(expr: str) -> CronTrigger:
    """Validate and compile a standard 5-field cron expression (UTC).

    Raises ``ValueError`` on invalid input (APScheduler's CronTrigger).
    """
    return CronTrigger.from_crontab((expr or "").strip(), timezone="UTC")


def next_run_at(expr: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """Next fire time of ``expr`` strictly after ``now`` (UTC-aware)."""
    now = now or datetime.now(timezone.utc)
    return parse_cron(expr).get_next_fire_time(None, now)


def _naive(now: datetime) -> datetime:
    """Naive UTC timestamp for the DB DateTime columns (codebase convention)."""
    return now.astimezone(timezone.utc).replace(tzinfo=None)


def _is_due(schedule: DocgenScheduleORM, trigger: CronTrigger, now: datetime) -> bool:
    """Due when the cron's next fire after the last run has already passed."""
    anchor = schedule.last_run_at or schedule.created_at
    if anchor is None:  # fresh row not yet flushed
        return True
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    fire = trigger.get_next_fire_time(anchor, now)
    return fire is not None and fire <= now


def _load_entity(db: Any, schedule: DocgenScheduleORM) -> Optional[Any]:
    p_orm = db.get(ProductORM, schedule.product_id)
    if p_orm is None:
        return None
    try:
        return product_repo.find_entity(p_orm, schedule.entity_type, schedule.entity_id)
    except KeyError:  # corrupted entity_type outside SCHEDULE_ENTITIES
        return None


def _fire(db: Any, schedule: DocgenScheduleORM, now: datetime) -> None:
    """Submit one due schedule (or record skipped_busy / error) and commit."""
    entity = _load_entity(db, schedule)
    if entity is None:
        schedule.last_run_at = _naive(now)
        schedule.last_result = "error"
        db.commit()
        logger.warning(
            "Docgen schedule %s: %s %s not found; recorded error.",
            schedule.id, schedule.entity_type, schedule.entity_id,
        )
        return
    if jobs.entity_busy(schedule.entity_type, schedule.entity_id):
        # Any active job holding the entity — including a detect job
        # (registered as detect:{type} under the same entity lock).
        schedule.last_run_at = _naive(now)
        schedule.last_result = "skipped_busy"
        db.commit()
        logger.info(
            "Docgen schedule %s fired for %s %s: skipped_busy",
            schedule.id, schedule.entity_type, schedule.entity_id,
        )
        return
    has_baseline = bool(
        (getattr(entity, "source_state", None) or {}).get("baseline")
    )
    if has_baseline:
        job_id, is_new = jobs.submit_incremental_job(
            schedule.product_id, schedule.entity_type, schedule.entity_id,
            source="schedule",
        )
    else:
        job_id, is_new = jobs.create_or_get_job(
            schedule.product_id, schedule.entity_type, schedule.entity_id,
            source="schedule",
        )
        if is_new:
            jobs.submit_job(
                job_id, schedule.product_id, schedule.entity_type,
                schedule.entity_id, None, None,
            )
    schedule.last_run_at = _naive(now)
    if is_new:
        schedule.last_result = "submitted"
        _pending[schedule.id] = job_id
    else:
        schedule.last_result = "skipped_busy"
    db.commit()
    logger.info(
        "Docgen schedule %s fired for %s %s: %s",
        schedule.id, schedule.entity_type, schedule.entity_id, schedule.last_result,
    )


def _reconcile(db: Any) -> None:
    """Finalize last_result for scheduler-launched jobs that have finished."""
    if not _pending:
        return
    for schedule_id, job_id in list(_pending.items()):
        job = jobs.get_job(job_id)
        if job is None:
            # Registry lost (restart / pruned): leave the recorded result.
            _pending.pop(schedule_id, None)
            continue
        if job.get("status") in ("queued", "running"):
            continue
        _pending.pop(schedule_id, None)
        schedule = db.get(DocgenScheduleORM, schedule_id)
        if schedule is None:  # deleted while its job ran
            continue
        schedule.last_result = _JOB_RESULTS.get(job["status"], "error")
    db.commit()


def _tick_sync() -> None:
    """One scheduler pass: reconcile finished runs, fire due schedules."""
    now = datetime.now(timezone.utc)
    db = SessionLocal()
    try:
        _reconcile(db)
        schedules = (
            db.query(DocgenScheduleORM)
            .filter(DocgenScheduleORM.enabled.is_(True))
            .all()
        )
        for schedule in schedules:
            try:
                trigger = parse_cron(schedule.cron_expr)
            except ValueError:
                logger.warning(
                    "Docgen schedule %s has an invalid cron %r; skipping.",
                    schedule.id, schedule.cron_expr,
                )
                continue
            if _is_due(schedule, trigger, now):
                _fire(db, schedule, now)
    except Exception:
        db.rollback()
        logger.error("Docgen scheduler tick failed", exc_info=True)
    finally:
        try:
            db.close()
        except Exception:  # pragma: no cover - defensive
            pass


async def tick() -> None:
    """Async entry (APScheduler): the pass is sync SQLAlchemy, so it runs in
    a worker thread and never blocks the main event loop."""
    await asyncio.to_thread(_tick_sync)


def start_scheduler() -> None:
    """Start the in-process scheduler (idempotent, called from the lifespan)."""
    global _scheduler
    if _scheduler is not None:
        return
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        tick,
        "interval",
        seconds=TICK_SECONDS,
        id="docgen_schedules_tick",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    _scheduler = scheduler
    logger.info("Docgen scheduler started (tick every %ss)", TICK_SECONDS)


def shutdown_scheduler() -> None:
    """Stop the scheduler (idempotent, called from the lifespan shutdown)."""
    global _scheduler
    if _scheduler is None:
        return
    try:
        _scheduler.shutdown(wait=False)
    except Exception:  # pragma: no cover - already stopped
        logger.warning("Docgen scheduler shutdown issue", exc_info=True)
    _scheduler = None
    logger.info("Docgen scheduler stopped")
