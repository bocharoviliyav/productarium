"""Docgen schedules router (SPEC-009) — cron-driven automatic doc generation.

Endpoints (prefix ``/api/products``, tags ``schedules``), configure right
(SPEC-023: owner / manager / admin — NOT via grants):
- ``GET    /api/products/{product_id}/schedules``
- ``PUT    /api/products/{product_id}/schedules``  (upsert one, body-param)
- ``DELETE /api/products/{product_id}/schedules``  (?entity_type=&entity_id=)

Cron expressions are standard 5-field, evaluated in server time (UTC);
responses carry ``next_run_at`` computed from the schedule.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.auth.deps import get_current_user, require_product_permission
from api.db import get_db
from api.docgen.scheduler import SCHEDULE_ENTITIES, next_run_at, parse_cron
from api.models import DocgenScheduleORM, ProductORM, UserORM
from api.repositories import product_repo

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/products",
    tags=["schedules"],
    dependencies=[Depends(get_current_user)],
)


class ScheduleUpdate(BaseModel):
    entity_type: str  # codebase | database
    entity_id: str
    cron_expr: str
    enabled: bool = True


def _schedule_out(schedule: DocgenScheduleORM) -> Dict[str, Any]:
    next_run: Optional[str] = None
    if schedule.enabled:
        try:
            fire = next_run_at(schedule.cron_expr)
            next_run = fire.isoformat() if fire is not None else None
        except ValueError:
            next_run = None
    return {
        "id": schedule.id,
        "product_id": schedule.product_id,
        "entity_type": schedule.entity_type,
        "entity_id": schedule.entity_id,
        "cron_expr": schedule.cron_expr,
        "enabled": schedule.enabled,
        "last_run_at": schedule.last_run_at.isoformat() if schedule.last_run_at else None,
        "last_result": schedule.last_result,
        "created_by": schedule.created_by,
        "updated_at": schedule.updated_at.isoformat() if schedule.updated_at else None,
        "next_run_at": next_run,
    }


def _require_entity(product: ProductORM, entity_type: str, entity_id: str) -> None:
    if product_repo.find_entity(product, entity_type, entity_id) is None:
        raise HTTPException(
            status_code=404, detail=f"{entity_type.capitalize()} not found"
        )


@router.get("/{product_id}/schedules")
def list_schedules(
    product_id: str,
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_permission("configure")),
):
    rows = (
        db.query(DocgenScheduleORM)
        .filter(DocgenScheduleORM.product_id == product_id)
        .order_by(DocgenScheduleORM.entity_type, DocgenScheduleORM.entity_id)
        .all()
    )
    return [_schedule_out(r) for r in rows]


@router.put("/{product_id}/schedules")
def upsert_schedule(
    product_id: str,
    body: ScheduleUpdate,
    db: Session = Depends(get_db),
    product: ProductORM = Depends(require_product_permission("configure")),
    user: UserORM = Depends(get_current_user),
):
    """Create or update the schedule of ONE product entity (upsert on the
    unique (product_id, entity_id))."""
    if body.entity_type not in SCHEDULE_ENTITIES:
        raise HTTPException(
            status_code=400,
            detail="entity_type must be 'codebase' or 'database'",
        )
    try:
        parse_cron(body.cron_expr)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid cron expression (expected 5 fields, UTC)"
        )
    _require_entity(product, body.entity_type, body.entity_id)

    schedule = (
        db.query(DocgenScheduleORM)
        .filter(
            DocgenScheduleORM.product_id == product_id,
            DocgenScheduleORM.entity_id == body.entity_id,
        )
        .one_or_none()
    )
    if schedule is None:
        schedule = DocgenScheduleORM(
            id=f"sch_{secrets.token_hex(12)}",
            product_id=product_id,
            entity_type=body.entity_type,
            entity_id=body.entity_id,
            cron_expr=body.cron_expr.strip(),
            enabled=body.enabled,
            created_by=user.id,
        )
        db.add(schedule)
    else:
        schedule.entity_type = body.entity_type
        schedule.cron_expr = body.cron_expr.strip()
        schedule.enabled = body.enabled
    try:
        db.commit()
    except IntegrityError:
        # Lost a concurrent first-PUT race on the same (product, entity):
        # adopt the winner's row and update it (upsert), never a 500.
        db.rollback()
        schedule = (
            db.query(DocgenScheduleORM)
            .filter(
                DocgenScheduleORM.product_id == product_id,
                DocgenScheduleORM.entity_id == body.entity_id,
            )
            .one_or_none()
        )
        if schedule is None:
            raise
        schedule.entity_type = body.entity_type
        schedule.cron_expr = body.cron_expr.strip()
        schedule.enabled = body.enabled
        db.commit()
    logger.info(
        "Schedule %s for %s %s: cron=%r enabled=%s",
        schedule.id, body.entity_type, body.entity_id,
        schedule.cron_expr, schedule.enabled,
    )
    return _schedule_out(schedule)


@router.delete("/{product_id}/schedules")
def delete_schedule(
    product_id: str,
    entity_type: str = Query(...),
    entity_id: str = Query(...),
    db: Session = Depends(get_db),
    _product: ProductORM = Depends(require_product_permission("configure")),
):
    """Remove one entity's schedule (idempotent)."""
    schedule = (
        db.query(DocgenScheduleORM)
        .filter(
            DocgenScheduleORM.product_id == product_id,
            DocgenScheduleORM.entity_type == entity_type,
            DocgenScheduleORM.entity_id == entity_id,
        )
        .one_or_none()
    )
    if schedule is not None:
        db.delete(schedule)
        db.commit()
    return {"message": "Schedule removed"}


__all__ = ["router"]
