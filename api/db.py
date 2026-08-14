"""SQLAlchemy database setup for Product persistence.

Reuses the same environment variables as the cognee setup so the products
and knowledge tables live in the same Postgres database that cognee uses.
If the database is unavailable at startup, ``init_db`` logs a warning and
returns ``False`` instead of crashing the FastAPI application.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from api.models import Base

logger = logging.getLogger(__name__)

# --- Connection configuration (mirrors api/cognee/ env vars) ---
DB_PROVIDER = os.environ.get("DB_PROVIDER", "postgres")
# IMPORTANT: default to "localhost" for local runs; never hardcode the
# redacted placeholder used in api/cognee/_runtime.py.
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


def init_db() -> bool:
    """Create all Productarium tables if missing.

    Idempotent: only attempts ``create_all`` once per process. On failure
    (e.g. DB unreachable) logs a warning and returns ``False`` without
    raising, so app startup is never blocked.
    """
    global _db_ready
    if _db_ready:
        return True
    try:
        Base.metadata.create_all(bind=engine)
        _db_ready = True
        logger.info("SQLAlchemy tables ready (url=%s).", _safe_url(DATABASE_URL))
        return True
    except Exception as e:
        logger.warning("create_all failed (non-fatal): %s", e)
        return False


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
