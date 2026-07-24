"""SQLAlchemy database setup for Product/Artifact persistence.

Reuses the SAME environment variables as ``api/cognee_manager.py`` so that
the products/artifacts tables live in the same Postgres database that cognee
uses:

    DB_PROVIDER  (default: "postgres")
    DB_HOST      (default: "localhost" for local runs)
    DB_PORT      (default: "5432")
    DB_NAME      (default: "cognee_db")
    DB_USERNAME  (default: "cognee")
    DB_PASSWORD  (default: "cognee")

If the database is unavailable at startup, ``init_db`` logs a warning and
returns ``False`` instead of crashing the FastAPI application. CRUD endpoints
will then fail at request time, which is acceptable per the graceful-fallback
requirement.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from api.models import Base

logger = logging.getLogger(__name__)

# --- Connection configuration (mirrors api/cognee_manager.py env vars) ---
DB_PROVIDER = os.environ.get("DB_PROVIDER", "postgres")
# IMPORTANT: default to "localhost" for local runs; never hardcode the
# redacted placeholder used in cognee_manager.py.
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "cognee_db")
DB_USERNAME = os.environ.get("DB_USERNAME", "cognee")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "cognee")


def _build_database_url() -> str:
    """Build a SQLAlchemy URL from the configured provider/credentials."""
    provider = (DB_PROVIDER or "postgres").lower()
    if provider in ("postgres", "postgresql"):
        # psycopg (v3) sync driver.
        return (
            f"postgresql+psycopg://{DB_USERNAME}:{DB_PASSWORD}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
    # Fallback to an in-memory SQLite DB so the module still imports/tests
    # without a real Postgres instance. This keeps unit tests runnable.
    logger.warning(
        "Unsupported DB_PROVIDER=%r; falling back to in-memory SQLite.",
        provider,
    )
    return "sqlite:///:memory:"


DATABASE_URL = _build_database_url()

# ``pool_pre_ping`` avoids stale-connection errors after DB restarts.
# ``future=True`` enables SQLAlchemy 2.0-style behavior.
engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Track whether the schema has been created so init_db is idempotent.
_db_ready: bool = False


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_one_shot_migration() -> None:
    """One-shot additive + drop migration for the Productarium schema changes.

    Runs AFTER ``create_all``. ``create_all`` only creates missing tables and
    never adds/drops columns on existing tables, so the Product/Artifact schema
    evolution is handled here with explicit ALTER statements guarded by
    column-existence checks. Everything is wrapped so a failure is non-fatal
    (logs a warning, app startup continues). Safe on Postgres and SQLite.
    """
    try:
        insp = inspect(engine)
        table_names = set(insp.get_table_names())
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Migration: could not inspect DB schema (non-fatal): %s", e)
        return

    # --- products: add summary + owner_id, drop legacy `type` ---
    try:
        if "products" in table_names:
            cols = {c["name"] for c in insp.get_columns("products")}
            with engine.begin() as conn:
                if "summary" not in cols:
                    conn.execute(text("ALTER TABLE products ADD COLUMN summary TEXT"))
                    logger.info("Migration: added products.summary")
                if "owner_id" not in cols:
                    conn.execute(
                        text("ALTER TABLE products ADD COLUMN owner_id VARCHAR(64)")
                    )
                    logger.info("Migration: added products.owner_id")
                    try:
                        conn.execute(
                            text(
                                "CREATE INDEX IF NOT EXISTS ix_products_owner_id "
                                "ON products (owner_id)"
                            )
                        )
                    except Exception as e:  # pragma: no cover - index best-effort
                        logger.debug("Migration: owner_id index skipped: %s", e)
                if "type" in cols:
                    conn.execute(text("ALTER TABLE products DROP COLUMN type"))
                    logger.info("Migration: dropped products.type")
    except Exception as e:
        logger.warning("Migration: products schema update failed (non-fatal): %s", e)

    # --- artifacts: add kind/verified*/source, map legacy types ---
    try:
        if "artifacts" in table_names:
            cols = {c["name"] for c in insp.get_columns("artifacts")}
            with engine.begin() as conn:
                if "kind" not in cols:
                    conn.execute(text("ALTER TABLE artifacts ADD COLUMN kind VARCHAR(64)"))
                    logger.info("Migration: added artifacts.kind")
                if "verified" not in cols:
                    conn.execute(
                        text("ALTER TABLE artifacts ADD COLUMN verified BOOLEAN DEFAULT FALSE")
                    )
                    logger.info("Migration: added artifacts.verified")
                if "verified_by" not in cols:
                    conn.execute(
                        text("ALTER TABLE artifacts ADD COLUMN verified_by VARCHAR(64)")
                    )
                    logger.info("Migration: added artifacts.verified_by")
                if "verified_at" not in cols:
                    conn.execute(
                        text("ALTER TABLE artifacts ADD COLUMN verified_at TIMESTAMP")
                    )
                    logger.info("Migration: added artifacts.verified_at")
                if "source" not in cols:
                    conn.execute(
                        text("ALTER TABLE artifacts ADD COLUMN source VARCHAR(32) DEFAULT 'manual'")
                    )
                    logger.info("Migration: added artifacts.source")
                # Map legacy types -> new (type, kind). COALESCE keeps an explicit
                # kind if one was already set.
                conn.execute(
                    text(
                        "UPDATE artifacts SET kind = COALESCE(kind, 'openapi'), "
                        "type = 'spec' WHERE type = 'openapi'"
                    )
                )
                conn.execute(
                    text(
                        "UPDATE artifacts SET kind = COALESCE(kind, 'asyncapi'), "
                        "type = 'spec' WHERE type = 'asyncapi'"
                    )
                )
                conn.execute(
                    text(
                        "UPDATE artifacts SET kind = COALESCE(kind, 'testcase'), "
                        "type = 'documentation' WHERE type = 'testcase'"
                    )
                )
                logger.info("Migration: mapped legacy artifact types -> spec/documentation")
    except Exception as e:
        logger.warning("Migration: artifacts schema update failed (non-fatal): %s", e)


def init_db() -> bool:
    """Create all Productarium tables if missing and run the one-shot migration.

    Idempotent: only attempts ``create_all`` + migration once per process. On
    failure (e.g. DB unreachable) logs a warning and returns ``False`` without
    raising, so app startup is never blocked. The one-shot migration is
    best-effort and ALWAYS runs (even if ``create_all`` partially fails) so
    pre-existing tables (products/artifacts) still get new columns.
    """
    global _db_ready
    if _db_ready:
        return True
    create_ok = False
    try:
        Base.metadata.create_all(bind=engine)
        create_ok = True
        logger.info("SQLAlchemy tables ready (url=%s).", _safe_url(DATABASE_URL))
    except Exception as e:
        # create_all can partially fail (e.g. a FK target table name collides
        # with cognee's tables in the shared DB). Don't abort: the one-shot
        # migration below still heals pre-existing products/artifacts tables,
        # and a clean rename of the colliding table fixes the next run.
        logger.warning(
            "create_all failed (non-fatal; will still run migration): %s", e
        )
    # Always run the one-shot migration so existing tables get new columns
    # regardless of whether create_all fully succeeded.
    _run_one_shot_migration()
    _db_ready = True
    return create_ok


def _safe_url(url: str) -> str:
    """Strip the password from a DB URL for safe logging."""
    try:
        if "://" in url and "@" in url:
            creds, rest = url.split("://", 1)[1].split("@", 1)
            if ":" in creds:
                user = creds.split(":", 1)[0]
                return f"{url.split('://', 1)[0]}://{user}:***@{rest}"
        return url
    except Exception:
        return "<unparseable db url>"
