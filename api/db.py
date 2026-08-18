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

    On Postgres also ensures the ``vector`` extension exists (required by the
    pgvector-direct memory backend's ``knowledge_chunks.embedding`` column)
    and creates an HNSW index on it for fast cosine search. Both are best-
    effort and non-fatal; on SQLite (tests) they are skipped.
    """
    global _db_ready
    if _db_ready:
        return True
    try:
        _ensure_pgvector_extension()
        Base.metadata.create_all(bind=engine)
        _ensure_hnsw_index()
        _db_ready = True
        logger.info("SQLAlchemy tables ready (url=%s).", _safe_url(DATABASE_URL))
        return True
    except Exception as e:
        logger.warning("create_all failed (non-fatal): %s", e)
        return False


def _ensure_pgvector_extension() -> None:
    """CREATE EXTENSION IF NOT EXISTS vector on Postgres (non-fatal).

    Required by the ``knowledge_chunks.embedding`` pgvector column. Skipped on
    non-Postgres backends (e.g. SQLite in tests). Requires superuser or the
    ``pgvector`` extension to be pre-installed in the Postgres image; on a
    failure (privileges / extension absent) we log and continue — the table
    creation below will then raise a clearer error if the column type is
    actually needed, and tests on SQLite never reach this path.
    """
    provider = (DB_PROVIDER or "").lower()
    if provider not in ("postgres", "postgresql"):
        return
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
    except Exception as e:  # pragma: no cover - depends on live Postgres
        logger.warning("Could not create 'vector' extension (non-fatal): %s", e)


def _ensure_hnsw_index() -> None:
    """Create an HNSW index on knowledge_chunks.embedding for cosine search.

    Idempotent: uses ``IF NOT EXISTS``. Skipped on non-Postgres and when
    pgvector is unavailable. The index accelerates the cosine-distance
    ``ORDER BY embedding <=> :q`` query in the pgvector memory backend; without
    it Postgres falls back to a sequential scan (correct but slower). HNSW is
    chosen over IVFFlat for its build-as-you-go nature (no separate training
    step) and good recall at low data volumes. Non-fatal.
    """
    provider = (DB_PROVIDER or "").lower()
    if provider not in ("postgres", "postgresql"):
        return
    try:
        from api.models import _PGVECTOR_AVAILABLE
        if not _PGVECTOR_AVAILABLE:
            return
    except Exception:
        return
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding_hnsw "
                "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
            )
            conn.commit()
    except Exception as e:  # pragma: no cover - depends on live Postgres
        logger.warning("Could not create HNSW index on knowledge_chunks.embedding (non-fatal): %s", e)


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
