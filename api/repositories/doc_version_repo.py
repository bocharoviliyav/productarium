"""Append-only documentation version snapshots (Vault KV-v2 semantics).

Every successful generation, manual edit and rollback APPENDS an immutable
snapshot row; the artifact row's ``current_version`` points at the active
one. Restores append a NEW version with the old payload — history is never
rewritten. The vector memory stays keyed to the artifact (delete-then-insert
by source_id), so only the CURRENT version is searchable.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from api.models import DocVersionORM

logger = logging.getLogger(__name__)


def _payload(entity_type: str, entity: Any) -> Dict[str, Any]:
    """Snapshot fields by entity type: codebase/database → docs+pages, spec → content."""
    if entity_type == "spec":
        return {"content": getattr(entity, "content", None)}
    return {
        "generated_docs": getattr(entity, "generated_docs", None),
        "pages": getattr(entity, "pages", None),
    }


def _has_payload(entity_type: str, entity: Any) -> bool:
    return any(
        v not in (None, {}, "")
        for v in _payload(entity_type, entity).values()
    )


def _versions_query(db: Session, entity_type: str, entity_id: str):
    return db.query(DocVersionORM).filter(
        DocVersionORM.entity_type == entity_type,
        DocVersionORM.entity_id == entity_id,
    )


def ensure_baseline_version(
    db: Session, entity_type: str, entity: Any
) -> Optional[int]:
    """Snapshot the CURRENT stored state as version 1 (legacy bootstrap).

    No-op when any version already exists or the artifact has nothing to
    snapshot. Called before the first mutation (job start / edit) so legacy
    artifacts get a restore point.
    """
    entity_id = getattr(entity, "id", None)
    if not entity_id or _versions_query(db, entity_type, entity_id).first() is not None:
        return None
    if not _has_payload(entity_type, entity):
        return None
    return append_version(db, entity_type, entity, source="baseline")


def append_version(
    db: Session,
    entity_type: str,
    entity: Any,
    *,
    source: str,
    model: Optional[str] = None,
    job_id: Optional[str] = None,
) -> int:
    """Snapshot the entity's CURRENT doc state as max(version)+1 (append-only)."""
    entity_id = getattr(entity, "id", None)
    top = (
        _versions_query(db, entity_type, entity_id)
        .with_entities(func.max(DocVersionORM.version))
        .scalar()
    )
    version = int(top) + 1 if top else 1
    db.add(DocVersionORM(
        id=f"docv_{secrets.token_hex(16)}",
        product_id=getattr(entity, "product_id", None) or "",
        entity_type=entity_type,
        entity_id=entity_id,
        version=version,
        source=source,
        model=model,
        job_id=job_id,
        created_at=datetime.utcnow(),
        **_payload(entity_type, entity),
    ))
    # Sessions run with autoflush=False: flush so a second append in the SAME
    # transaction (baseline + edit) already sees this row in max(version).
    db.flush()
    try:
        entity.current_version = version
    except Exception:  # pragma: no cover - test doubles without the column
        pass
    logger.info(
        "Doc version appended: %s %s v%d (%s)", entity_type, entity_id, version, source
    )
    return version


def list_versions(
    db: Session, entity_type: str, entity_id: str
) -> List[DocVersionORM]:
    """All versions of one artifact, newest first."""
    return (
        _versions_query(db, entity_type, entity_id)
        .order_by(DocVersionORM.version.desc())
        .all()
    )


def get_version(
    db: Session, entity_type: str, entity_id: str, version: int
) -> Optional[DocVersionORM]:
    return _versions_query(db, entity_type, entity_id).filter(
        DocVersionORM.version == version
    ).first()


def restore_version(
    db: Session, entity_type: str, entity: Any, version: int
) -> Optional[str]:
    """Write an old snapshot back onto the artifact + append a rollback version.

    Returns the text to re-index (None when the version does not exist).
    History is preserved: the rollback itself becomes the new current version.
    """
    entity_id = getattr(entity, "id", None)
    row = get_version(db, entity_type, entity_id, version)
    if row is None:
        return None
    if entity_type == "spec":
        entity.content = row.content
        indexed = row.content
    else:
        entity.generated_docs = row.generated_docs
        entity.pages = row.pages
        indexed = row.generated_docs
    append_version(db, entity_type, entity, source="rollback", model=row.model)
    return indexed


def clear_artifact_docs(entity_type: str, entity: Any) -> None:
    """Reset the artifact's doc columns (cancel of a first-ever run: nothing to restore)."""
    if entity_type == "spec":
        return  # spec generation rewrites `content` only at the final node
    entity.generated_docs = None
    entity.pages = None


def restore_entity_from_current_version(
    db: Session, entity_type: str, entity: Any
) -> bool:
    """Restore the artifact from its current version row (cancel rollback).

    Mid-run checkpoints may have committed partial docs onto the row; a
    cancelled job must leave the artifact at its pre-run (current) version.
    When no version exists the doc columns are simply cleared. The caller
    commits; returns True when a restore happened.
    """
    version = getattr(entity, "current_version", None)
    row = (
        get_version(db, entity_type, getattr(entity, "id", None), int(version))
        if version
        else None
    )
    if row is None:
        clear_artifact_docs(entity_type, entity)
        return bool(version)
    if entity_type == "spec":
        entity.content = row.content
    else:
        entity.generated_docs = row.generated_docs
        entity.pages = row.pages
    return True
