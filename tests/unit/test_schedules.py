"""Unit + hermetic integration tests for SPEC-009 scheduled docgen.

Covers:
- cron parsing (valid/invalid) and next_run math (UTC);
- schedule uniqueness per (product, entity);
- the scheduler tick: due + free entity → job registered with
  source="schedule" (incremental when a SPEC-010 baseline exists, else full
  generate); busy entity → skipped_busy recorded, the running job untouched;
  missing entity → error; job outcome reconciliation (ok/error);
  not-due schedules never fire;
- the REST surface (configure right, SPEC-023): upsert/get/delete, cron +
  entity validation, next_run_at in the payload, permission matrix.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import api.docgen.jobs as jobs_mod
import api.docgen.scheduler as sched_mod
from tests.conftest import build_test_client

PROD = "prod_sch"
CB = "cb_sch"
DB = "db_sch"


# --- Shared helpers -----------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_registries():
    jobs_mod._docgen_jobs.clear()
    sched_mod._pending.clear()
    yield
    jobs_mod._docgen_jobs.clear()
    sched_mod._pending.clear()


def _seed(db_mod, *, baseline: bool = False):
    """Product + codebase + database; codebase optionally carries a SPEC-010
    baseline (source_state.baseline) so the tick takes the incremental path."""
    from api.models import CodebaseORM, DatabaseORM, ProductORM

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id=PROD, name="SchedProduct"))
        db.add(CodebaseORM(
            id=CB, product_id=PROD, name="repo",
            repo_url="https://github.com/o/repo",
            source_state={"kind": "codebase", "baseline": "a" * 40} if baseline else None,
        ))
        db.add(DatabaseORM(id=DB, product_id=PROD, name="oltp"))
        db.commit()


def _add_schedule(db_mod, *, entity_type="codebase", entity_id=CB,
                  cron="*/1 * * * *", enabled=True, last_run_at=None,
                  last_result=None):
    from api.models import DocgenScheduleORM

    with db_mod.SessionLocal() as db:
        schedule = DocgenScheduleORM(
            id=f"sch_{entity_id}",
            product_id=PROD,
            entity_type=entity_type,
            entity_id=entity_id,
            cron_expr=cron,
            enabled=enabled,
            last_run_at=last_run_at,
            last_result=last_result,
        )
        db.add(schedule)
        db.commit()
        return schedule.id


def _read_schedule(db_mod, schedule_id):
    from api.models import DocgenScheduleORM

    with db_mod.SessionLocal() as db:
        row = db.get(DocgenScheduleORM, schedule_id)
        assert row is not None
        return row.last_run_at, row.last_result


def _patch_submits(monkeypatch):
    """Keep the tick hermetic: register jobs in the real registry but never
    dispatch the real generation pipeline to the executor."""
    submitted = []
    monkeypatch.setattr(
        jobs_mod, "submit_job",
        lambda job_id, *a, **kw: submitted.append(job_id),
    )
    return submitted


def _run_tick(db_mod, monkeypatch):
    monkeypatch.setattr(sched_mod, "SessionLocal", db_mod.SessionLocal)
    asyncio.run(sched_mod.tick())


# --- Cron parsing / next_run --------------------------------------------------
class TestCron:
    def test_parse_valid_and_next_run_utc(self):
        fire = sched_mod.next_run_at(
            "0 2 * * *", datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
        )
        assert fire == datetime(2026, 10, 27, 2, 0, tzinfo=timezone.utc)
        assert fire.tzinfo is not None

    def test_parse_every_minute(self):
        fire = sched_mod.next_run_at(
            "*/1 * * * *", datetime(2026, 10, 26, 12, 0, 30, tzinfo=timezone.utc)
        )
        assert fire == datetime(2026, 10, 26, 12, 1, tzinfo=timezone.utc)

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError):
            sched_mod.parse_cron("99 * * * *")
        with pytest.raises(ValueError):
            sched_mod.parse_cron("* * *")  # not 5 fields

    def test_next_run_at_boundary_is_inclusive(self):
        # APScheduler semantics: a fire time equal to ``now`` is returned as
        # the next run (>= now, not strictly after).
        now = datetime(2026, 10, 26, 2, 0, tzinfo=timezone.utc)
        assert sched_mod.next_run_at("0 2 * * *", now) == now
        just_before = datetime(2026, 10, 26, 1, 59, 59, tzinfo=timezone.utc)
        assert sched_mod.next_run_at("0 2 * * *", just_before) == now


# --- Uniqueness ---------------------------------------------------------------
class TestScheduleUniqueness:
    def test_duplicate_entity_rejected(self, isolated_db):
        _seed(isolated_db)
        _add_schedule(isolated_db)
        from api.models import DocgenScheduleORM

        with isolated_db.SessionLocal() as db:
            db.add(DocgenScheduleORM(
                id="sch_dup", product_id=PROD, entity_type="codebase",
                entity_id=CB, cron_expr="0 2 * * *",
            ))
            with pytest.raises(IntegrityError):
                db.commit()
            db.rollback()


# --- Tick behaviour -----------------------------------------------------------
class TestTick:
    def test_due_free_entity_full_generate(self, isolated_db, monkeypatch):
        _seed(isolated_db, baseline=False)  # no baseline -> full generate
        submitted = _patch_submits(monkeypatch)
        sid = _add_schedule(
            isolated_db, last_run_at=datetime.utcnow() - timedelta(minutes=2)
        )
        _run_tick(isolated_db, monkeypatch)

        last_run, last_result = _read_schedule(isolated_db, sid)
        assert last_result == "submitted"
        assert last_run is not None
        assert len(submitted) == 1
        job = jobs_mod.get_job(submitted[0])
        assert job["source"] == "schedule"
        assert sched_mod._pending[sid] == submitted[0]

    def test_due_free_entity_with_baseline_incremental(self, isolated_db, monkeypatch):
        _seed(isolated_db, baseline=True)
        submitted = _patch_submits(monkeypatch)
        sid = _add_schedule(
            isolated_db, last_run_at=datetime.utcnow() - timedelta(minutes=2)
        )
        _run_tick(isolated_db, monkeypatch)

        _, last_result = _read_schedule(isolated_db, sid)
        assert last_result == "submitted"
        assert len(submitted) == 1
        assert jobs_mod.get_job(submitted[0])["source"] == "schedule"

    def test_busy_entity_records_skipped_busy(self, isolated_db, monkeypatch):
        """Acceptance: a schedule event while a job is active → skipped_busy
        in the history, the running job is never touched."""
        _seed(isolated_db, baseline=True)
        submitted = _patch_submits(monkeypatch)
        active_id, is_new = jobs_mod.create_or_get_job(PROD, "codebase", CB)
        assert is_new
        job = jobs_mod.get_job(active_id)
        job["status"] = "running"
        job["source"] = "generate"  # a manual run must not be relabelled

        sid = _add_schedule(
            isolated_db, last_run_at=datetime.utcnow() - timedelta(minutes=2)
        )
        _run_tick(isolated_db, monkeypatch)

        _, last_result = _read_schedule(isolated_db, sid)
        assert last_result == "skipped_busy"
        assert submitted == []  # nothing dispatched
        assert jobs_mod.get_job(active_id)["source"] == "generate"
        assert sid not in sched_mod._pending

    def test_busy_detect_job_records_skipped_busy(self, isolated_db, monkeypatch):
        """Q-2: a tick while a DETECT job (key detect:codebase) holds the
        entity → skipped_busy, the detect job untouched (not error)."""
        _seed(isolated_db, baseline=True)
        submitted = _patch_submits(monkeypatch)
        detect_id, is_new = jobs_mod.create_or_get_job(
            PROD, "detect:codebase", CB
        )
        assert is_new
        jobs_mod.get_job(detect_id)["status"] = "running"

        sid = _add_schedule(
            isolated_db, last_run_at=datetime.utcnow() - timedelta(minutes=2)
        )
        _run_tick(isolated_db, monkeypatch)

        _, last_result = _read_schedule(isolated_db, sid)
        assert last_result == "skipped_busy"
        assert submitted == []  # nothing dispatched, no failed job
        assert jobs_mod.get_job(detect_id)["status"] == "running"
        assert sid not in sched_mod._pending

    def test_missing_entity_records_error(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        _patch_submits(monkeypatch)
        sid = _add_schedule(
            isolated_db, entity_id="cb_gone",
            last_run_at=datetime.utcnow() - timedelta(minutes=2),
        )
        _run_tick(isolated_db, monkeypatch)
        _, last_result = _read_schedule(isolated_db, sid)
        assert last_result == "error"

    def test_not_due_does_not_fire(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        submitted = _patch_submits(monkeypatch)
        # Daily 02:00 last fired a minute ago: next fire is tomorrow 02:00.
        sid = _add_schedule(
            isolated_db, cron="0 2 * * *",
            last_run_at=datetime.utcnow() - timedelta(minutes=1),
        )
        _run_tick(isolated_db, monkeypatch)
        _, last_result = _read_schedule(isolated_db, sid)
        assert last_result is None
        assert submitted == []

    def test_disabled_schedule_never_fires(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        submitted = _patch_submits(monkeypatch)
        _add_schedule(
            isolated_db, enabled=False,
            last_run_at=datetime.utcnow() - timedelta(hours=2),
        )
        _run_tick(isolated_db, monkeypatch)
        assert submitted == []

    def test_invalid_cron_row_skipped_without_crash(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        submitted = _patch_submits(monkeypatch)
        sid = _add_schedule(
            isolated_db, cron="not a cron",
            last_run_at=datetime.utcnow() - timedelta(minutes=2),
        )
        _run_tick(isolated_db, monkeypatch)
        assert submitted == []
        _, last_result = _read_schedule(isolated_db, sid)
        assert last_result is None  # untouched, only logged

    def test_reconcile_finalizes_job_outcome(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        submitted = _patch_submits(monkeypatch)
        # Just fired (anchor = now): not due again, so the tick only
        # reconciles the finished job into last_result.
        sid = _add_schedule(
            isolated_db, last_run_at=datetime.utcnow(),
            last_result="submitted",
        )
        job_id = jobs_mod.create_job(PROD, "codebase", CB)
        sched_mod._pending[sid] = job_id
        job = jobs_mod.get_job(job_id)
        job["status"] = "succeeded"
        job["finished_at"] = 1.0

        _run_tick(isolated_db, monkeypatch)
        _, last_result = _read_schedule(isolated_db, sid)
        assert last_result == "ok"
        assert sid not in sched_mod._pending

    def test_reconcile_failed_job(self, isolated_db, monkeypatch):
        _seed(isolated_db)
        _patch_submits(monkeypatch)
        sid = _add_schedule(isolated_db, last_result="submitted")
        job_id = jobs_mod.create_job(PROD, "codebase", CB)
        sched_mod._pending[sid] = job_id
        job = jobs_mod.get_job(job_id)
        job["status"] = "failed"
        job["finished_at"] = 1.0
        _run_tick(isolated_db, monkeypatch)
        _, last_result = _read_schedule(isolated_db, sid)
        assert last_result == "error"


# --- REST surface -------------------------------------------------------------
class TestSchedulesRest:
    def _client(self, isolated_db, role="admin"):
        import api.routers.schedules as schedules_mod

        _seed(isolated_db)
        app, client = build_test_client(isolated_db, [schedules_mod])
        if role != "admin":
            from datetime import datetime as _dt

            from api.auth import deps as auth_deps
            from api.models import UserORM

            plain = UserORM(
                id="user_1", username="u", role=role, provider="local",
                created_at=_dt.utcnow(),
            )
            app.dependency_overrides[auth_deps.get_current_user] = lambda: plain
        return client

    def test_put_then_get_roundtrip(self, isolated_db):
        client = self._client(isolated_db)
        resp = client.put(f"/api/products/{PROD}/schedules", json={
            "entity_type": "codebase", "entity_id": CB,
            "cron_expr": "0 2 * * *", "enabled": True,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["cron_expr"] == "0 2 * * *"
        assert body["enabled"] is True
        assert body["next_run_at"] is not None
        assert body["next_run_at"].endswith("+00:00")

        listed = client.get(f"/api/products/{PROD}/schedules").json()
        assert [r["entity_id"] for r in listed] == [CB]

    def test_put_upsert_single_row(self, isolated_db):
        client = self._client(isolated_db)
        for cron in ("0 2 * * *", "0 3 * * 0"):
            resp = client.put(f"/api/products/{PROD}/schedules", json={
                "entity_type": "codebase", "entity_id": CB,
                "cron_expr": cron, "enabled": True,
            })
            assert resp.status_code == 200
        listed = client.get(f"/api/products/{PROD}/schedules").json()
        assert len(listed) == 1
        assert listed[0]["cron_expr"] == "0 3 * * 0"

    def test_put_race_adopts_winner_row(self, isolated_db):
        """Q-16: a PUT that races a concurrent first-PUT on the same (product,
        entity) hits IntegrityError, adopts the winner's row and updates —
        an upsert, never a 500."""
        import api.routers.schedules as schedules_mod
        from api.models import DocgenScheduleORM

        _seed(isolated_db)
        # The "winner" already committed its row for the same entity.
        _add_schedule(isolated_db, cron="0 2 * * *")
        app, client = build_test_client(isolated_db, [schedules_mod])

        class _MissingRowQuery:
            """The router's lookup sees nothing (the race window) even though
            the row is committed."""

            def __init__(self, q):
                self._q = q

            def one_or_none(self):
                return None

            def __getattr__(self, name):
                attr = getattr(self._q, name)
                if name in ("filter", "filter_by", "order_by", "options"):
                    return lambda *a, **kw: _MissingRowQuery(attr(*a, **kw))
                return attr

        class _RaceSession:
            def __init__(self, real):
                self._real = real
                self._spoofed = False

            def __getattr__(self, name):
                return getattr(self._real, name)

            def query(self, *entities, **kw):
                q = self._real.query(*entities, **kw)
                if not self._spoofed and entities and entities[0] is DocgenScheduleORM:
                    self._spoofed = True  # only the FIRST lookup misses
                    return _MissingRowQuery(q)
                return q

        def _get_race_db():
            s = _RaceSession(isolated_db.SessionLocal())
            try:
                yield s
            finally:
                s._real.close()

        app.dependency_overrides[schedules_mod.get_db] = _get_race_db

        resp = client.put(f"/api/products/{PROD}/schedules", json={
            "entity_type": "codebase", "entity_id": CB,
            "cron_expr": "0 3 * * 0", "enabled": False,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["cron_expr"] == "0 3 * * 0"  # updated on the winner's row
        assert body["enabled"] is False

        listed = client.get(f"/api/products/{PROD}/schedules").json()
        assert len(listed) == 1
        assert listed[0]["cron_expr"] == "0 3 * * 0"

    def test_put_validates_cron_and_entity(self, isolated_db):
        client = self._client(isolated_db)
        bad_cron = client.put(f"/api/products/{PROD}/schedules", json={
            "entity_type": "codebase", "entity_id": CB, "cron_expr": "oops",
        })
        assert bad_cron.status_code == 400
        unknown = client.put(f"/api/products/{PROD}/schedules", json={
            "entity_type": "codebase", "entity_id": "cb_nope",
            "cron_expr": "0 2 * * *",
        })
        assert unknown.status_code == 404
        wrong_type = client.put(f"/api/products/{PROD}/schedules", json={
            "entity_type": "spec", "entity_id": "s1", "cron_expr": "0 2 * * *",
        })
        assert wrong_type.status_code == 400

    def test_put_unknown_product_404(self, isolated_db):
        client = self._client(isolated_db)
        resp = client.put("/api/products/prod_none/schedules", json={
            "entity_type": "codebase", "entity_id": CB, "cron_expr": "0 2 * * *",
        })
        assert resp.status_code == 404

    def test_delete_idempotent(self, isolated_db):
        client = self._client(isolated_db)
        client.put(f"/api/products/{PROD}/schedules", json={
            "entity_type": "database", "entity_id": DB, "cron_expr": "0 2 * * *",
        })
        q = f"entity_type=database&entity_id={DB}"
        assert client.delete(f"/api/products/{PROD}/schedules?{q}").status_code == 200
        assert client.delete(f"/api/products/{PROD}/schedules?{q}").status_code == 200
        assert client.get(f"/api/products/{PROD}/schedules").json() == []

    def test_delete_filters_entity_type(self, isolated_db):
        """S-8: DELETE removes the schedule of THAT entity type only."""
        client = self._client(isolated_db)
        client.put(f"/api/products/{PROD}/schedules", json={
            "entity_type": "codebase", "entity_id": CB, "cron_expr": "0 2 * * *",
        })
        wrong_type = f"entity_type=database&entity_id={CB}"
        assert client.delete(
            f"/api/products/{PROD}/schedules?{wrong_type}"
        ).status_code == 200
        listed = client.get(f"/api/products/{PROD}/schedules").json()
        assert [r["entity_id"] for r in listed] == [CB]  # codebase row intact
        right_type = f"entity_type=codebase&entity_id={CB}"
        assert client.delete(
            f"/api/products/{PROD}/schedules?{right_type}"
        ).status_code == 200
        assert client.get(f"/api/products/{PROD}/schedules").json() == []

    def test_configure_permission_matrix(self, isolated_db):
        """SPEC-023: schedules need the configure right (owner / manager /
        admin) — no longer admin-only: manager 200 everywhere, rw-grant and
        viewer_global 403, invisible stranger 404."""
        import api.routers.schedules as schedules_mod
        from api.auth import deps as auth_deps
        from api.models import ProductGrantORM, UserORM

        _seed(isolated_db)
        with isolated_db.SessionLocal() as db:
            for uid, role in (
                ("u_mgr", "manager"), ("u_rw", "user"),
                ("u_str", "user"), ("u_view", "viewer_global"),
            ):
                db.add(UserORM(
                    id=uid, username=uid, role=role, provider="local",
                    created_at=datetime.utcnow(),
                ))
            db.add(ProductGrantORM(
                product_id=PROD, user_id="u_rw", level="rw", granted_by="seed",
            ))
            db.commit()

        def _client(uid: str, role: str):
            app, client = build_test_client(isolated_db, [schedules_mod])
            app.dependency_overrides[auth_deps.get_current_user] = lambda: UserORM(
                id=uid, username=uid, role=role, provider="local",
                created_at=datetime.utcnow(),
            )
            return client

        base = f"/api/products/{PROD}/schedules"
        body = {"entity_type": "codebase", "entity_id": CB,
                "cron_expr": "0 2 * * *", "enabled": True}
        q = f"entity_type=codebase&entity_id={CB}"

        c = _client("u_mgr", "manager")
        assert c.get(base).status_code == 200
        assert c.put(base, json=body).status_code == 200
        assert c.delete(f"{base}?{q}").status_code == 200

        c = _client("u_rw", "user")
        assert c.get(base).status_code == 403
        assert c.put(base, json=body).status_code == 403
        assert c.delete(f"{base}?{q}").status_code == 403

        assert _client("u_str", "user").get(base).status_code == 404
        assert _client("u_view", "viewer_global").get(base).status_code == 403

    def test_disabled_schedule_next_run_none(self, isolated_db):
        client = self._client(isolated_db)
        body = client.put(f"/api/products/{PROD}/schedules", json={
            "entity_type": "codebase", "entity_id": CB,
            "cron_expr": "0 2 * * *", "enabled": False,
        }).json()
        assert body["enabled"] is False
        assert body["next_run_at"] is None
