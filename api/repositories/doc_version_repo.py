"""Append-only documentation version snapshots (Vault KV-v2 semantics).

Every successful generation, manual edit and rollback APPENDS an immutable
snapshot row; the artifact row's ``current_version`` points at the active
one. Restores append a NEW version with the old payload — history is never
rewritten. The vector memory stays keyed to the artifact (delete-then-insert
by source_id), so only the CURRENT version is searchable.

SPEC-012: page-scoped changes append DELTA rows — ``base_version`` names the
version they build on and ``pages`` holds only the overridden pages
(``generated_docs`` is None, the blob resolves through the chain). Reads
materialize the chain (base + overrides) transparently; chains that would
grow deeper than ``_MAX_DELTA_CHAIN`` flatten into a full snapshot at append
time. Rows without ``base_version`` — including every pre-SPEC-012 row —
are full snapshots.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import set_committed_value

from api.models import DocVersionORM

logger = logging.getLogger(__name__)

#: Delta chains at/above this depth flatten into a full snapshot on append.
_MAX_DELTA_CHAIN = 50
#: Entity types documented as a page tree (spec stores one content blob).
_PAGE_TYPES = ("codebase", "database", "hld")


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


def _raw_version(
    db: Session, entity_type: str, entity_id: str, version: int
) -> Optional[DocVersionORM]:
    """Stored row as-is (delta rows keep their override-only payload)."""
    return _versions_query(db, entity_type, entity_id).filter(
        DocVersionORM.version == version
    ).first()


def _base_chain(
    db: Session, entity_type: str, entity_id: str, row: DocVersionORM
) -> List[DocVersionORM]:
    """Materialization path ``[root, …, row]``; the root is the nearest full
    snapshot (full rows end the walk — newer history below them is not
    needed). Broken/cyclic base references stop at what exists."""
    chain = [row]
    seen = {row.version}
    while chain[-1].base_version is not None:
        base = _raw_version(
            db, entity_type, entity_id, int(chain[-1].base_version)
        )
        if base is None or base.version in seen:
            break
        chain.append(base)
        seen.add(base.version)
    chain.reverse()
    return chain


def _overlay(chain: List[DocVersionORM]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Materialized ``(pages, generated_docs)`` of a base chain: the root's
    payload updated with every delta's page overrides."""
    root = chain[0]
    pages: Dict[str, Any] = dict(root.pages or {})
    for row in chain[1:]:
        pages.update(row.pages or {})
    return pages, root.generated_docs


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


def _auto_delta(
    db: Session,
    entity_type: str,
    entity_id: str,
    entity: Any,
    source: str,
    force_pages: Optional[List[str]],
) -> Optional[Tuple[int, Dict[str, Any]]]:
    """``(base_version, override_pages)`` for a page-scoped change, else None.

    Page-scoped = a page edit, or a generate/schedule run limited to
    ``force_pages`` (per-page/branch regen, SPEC-010 incremental impact).
    The overrides come from DIFFING the entity against the materialized
    current version — unchanged pages stay out of the snapshot. A full
    snapshot is required when a delta cannot express the change: page
    deletions, an empty diff (blob-only edits) or the chain depth cap.
    """
    if entity_type not in _PAGE_TYPES:
        return None
    page_scoped = source == "edit" or (
        source in ("generate", "schedule") and force_pages
    )
    if not page_scoped:
        return None
    current = getattr(entity, "current_version", None)
    if not current:
        return None
    base = _raw_version(db, entity_type, entity_id, int(current))
    if base is None:
        return None
    chain = _base_chain(db, entity_type, entity_id, base)
    if len(chain) > _MAX_DELTA_CHAIN:
        return None  # at the cap: the next version flattens to a full snapshot
    base_pages, _ = _overlay(chain)
    pages = entity.pages if isinstance(entity.pages, dict) else {}
    if any(pid not in pages for pid in base_pages):
        return None  # deletions are not expressible as override subsets
    overrides = {pid: p for pid, p in pages.items() if base_pages.get(pid) != p}
    return (int(current), overrides) if overrides else None


def append_version(
    db: Session,
    entity_type: str,
    entity: Any,
    *,
    source: str,
    model: Optional[str] = None,
    job_id: Optional[str] = None,
    force_pages: Optional[List[str]] = None,
    delta: Optional[Tuple[int, Dict[str, Any]]] = None,
) -> int:
    """Snapshot the entity's CURRENT doc state as max(version)+1 (append-only).

    Page-scoped runs append a DELTA (SPEC-012): derived here from the diff
    against the current version, or taken from the explicit ``delta``
    directive (the rollback path in :func:`restore_version`). Everything a
    page subset cannot express stays a full snapshot — see
    :func:`_auto_delta`.
    """
    entity_id = getattr(entity, "id", None)
    top = (
        _versions_query(db, entity_type, entity_id)
        .with_entities(func.max(DocVersionORM.version))
        .scalar()
    )
    version = int(top) + 1 if top else 1
    if delta is None:
        delta = _auto_delta(db, entity_type, entity_id, entity, source, force_pages)
    if delta is None:
        base_version = None
        payload = _payload(entity_type, entity)
    else:
        base_version, overrides = delta
        payload = {"generated_docs": None, "pages": overrides}
    db.add(DocVersionORM(
        id=f"docv_{secrets.token_hex(16)}",
        product_id=getattr(entity, "product_id", None) or "",
        entity_type=entity_type,
        entity_id=entity_id,
        version=version,
        base_version=base_version,
        source=source,
        model=model,
        job_id=job_id,
        created_at=datetime.utcnow(),
        **payload,
    ))
    # Sessions run with autoflush=False: flush so a second append in the SAME
    # transaction (baseline + edit) already sees this row in max(version).
    db.flush()
    try:
        entity.current_version = version
    except Exception:  # pragma: no cover - test doubles without the column
        pass
    logger.info(
        "Doc version appended: %s %s v%d (%s, %s)",
        entity_type, entity_id, version, source,
        f"delta on v{base_version}" if base_version else "full",
    )
    return version


def list_versions(
    db: Session, entity_type: str, entity_id: str
) -> List[DocVersionORM]:
    """All versions of one artifact, newest first (raw rows: deltas keep
    their override-only ``pages``; use :func:`get_version` for the full view)."""
    return (
        _versions_query(db, entity_type, entity_id)
        .order_by(DocVersionORM.version.desc())
        .all()
    )


def get_version(
    db: Session, entity_type: str, entity_id: str, version: int
) -> Optional[DocVersionORM]:
    """One version row with its payload MATERIALIZED (SPEC-012): delta rows
    resolve through their base chain, so every consumer (detail endpoint,
    archived page reads, restore, cancel rollback) sees the full document.

    ``set_committed_value`` keeps the session clean — the materialized view
    is a read-only projection, never flushed back over the stored overrides.
    """
    row = _raw_version(db, entity_type, entity_id, version)
    if row is not None and row.base_version is not None:
        pages, generated_docs = _overlay(
            _base_chain(db, entity_type, entity_id, row)
        )
        set_committed_value(row, "pages", pages)
        set_committed_value(row, "generated_docs", generated_docs)
    return row


def restore_version(
    db: Session, entity_type: str, entity: Any, version: int
) -> Optional[str]:
    """Write an old snapshot back onto the artifact + append a rollback version.

    Returns the text to re-index (None when the version does not exist).
    History is preserved: the rollback itself becomes the new current version.
    SPEC-012: the rollback appends as a delta BASED ON THE RESTORED VERSION —
    its overrides are exactly the pages the rollback changes vs the outgoing
    current state (page deletions resolve naturally through the base).
    """
    entity_id = getattr(entity, "id", None)
    row = get_version(db, entity_type, entity_id, version)
    if row is None:
        return None
    delta = None
    if entity_type in _PAGE_TYPES:
        current = getattr(entity, "current_version", None)
        prev = get_version(db, entity_type, entity_id, int(current)) if current else None
        if prev is not None:
            prev_pages = prev.pages if isinstance(prev.pages, dict) else {}
            target_pages = row.pages if isinstance(row.pages, dict) else {}
            overrides = {
                pid: p for pid, p in target_pages.items()
                if prev_pages.get(pid) != p
            }
            target = _raw_version(db, entity_type, entity_id, version)
            if (
                overrides
                and len(_base_chain(db, entity_type, entity_id, target))
                <= _MAX_DELTA_CHAIN
            ):
                delta = (version, overrides)
    if entity_type == "spec":
        entity.content = row.content
        indexed = row.content
    else:
        entity.generated_docs = row.generated_docs
        entity.pages = row.pages
        indexed = row.generated_docs
    append_version(db, entity_type, entity, source="rollback", model=row.model, delta=delta)
    return indexed


def delete_version(
    db: Session, entity_type: str, entity: Any, version: int
) -> Optional[str]:
    """Delete ONE snapshot row (SPEC-013). Returns None on success, else a
    refusal reason: ``not_found`` (404) / ``current`` / ``single`` / ``base``
    (409). Only the snapshot row goes — the artifact row, its docs, the
    memory chunks (keyed to the artifact, not the version) and the
    ``current_version`` pointer are never touched.
    """
    entity_id = getattr(entity, "id", None)
    row = _raw_version(db, entity_type, entity_id, version)
    if row is None:
        return "not_found"
    if version == getattr(entity, "current_version", None):
        return "current"
    if _versions_query(db, entity_type, entity_id).count() <= 1:
        return "single"
    if (
        _versions_query(db, entity_type, entity_id)
        .filter(DocVersionORM.base_version == version)
        .first()
        is not None
    ):
        return "base"
    db.delete(row)
    return None


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
