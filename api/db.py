"""SQLAlchemy database setup for Product persistence.

The products and knowledge tables live in the Postgres database configured
via the ``DB_*`` environment variables (shared with the pgvector memory
backend). ``DB_PROVIDER=sqlite`` selects a file-based SQLite database (the
degraded local/test mode; see ``_sqlite_url``). Any other unsupported
provider falls back to an in-memory SQLite DB so the module still imports
and tests run without a real Postgres instance. If the database is
unavailable at startup, ``init_db`` logs a warning and returns ``False``
instead of crashing the FastAPI application.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, Iterator, Optional, Tuple

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from api.models import Base

logger = logging.getLogger(__name__)

# --- Connection configuration (DB_* env vars) ---
DB_PROVIDER = os.environ.get("DB_PROVIDER", "postgres")
# IMPORTANT: default to "localhost" for local runs.
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "cognee_db")
DB_USERNAME = os.environ.get("DB_USERNAME", "cognee")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "cognee")

# Default directory for a SQLite DB file when no explicit path is configured
# (only used when DB_PROVIDER=sqlite and DB_NAME is not already a path).
# Mirrors the agent-checkpointer state dir (api/agents/runtime.py).
def _default_sqlite_dir() -> str:
    """The default directory for SQLite database files (env-overridable)."""
    return os.environ.get("PRODUCTARIUM_STATE_DIR") or os.path.expanduser(
        "~/.productarium"
    )


def _sqlite_url() -> str:
    """Resolve the SQLite database URL from the DB_* configuration.

    Resolution order for the database file path:
    1. ``DB_NAME`` set to ``:memory:`` → the shared in-memory database.
    2. ``DB_NAME`` containing a path separator (``/`` or ``os.sep``) → used
       as the path as-is (absolute or relative; parent dirs are created).
    3. ``DB_HOST`` pointing at an existing directory → ``<DB_HOST>/<DB_NAME>``
       (the convention used by the test suite: ``DB_HOST=<tmpdir>``).
    4. Otherwise ``<state_dir>/<DB_NAME>`` where ``state_dir`` is
       ``PRODUCTARIUM_STATE_DIR`` or ``~/.productarium``.

    A path-like ``DB_NAME`` containing ``..`` segments is rejected outright
    (raises ``ValueError``): the SQLite file location is operator config, not
    a traversal surface.

    SQLite requires no credentials; ``DB_USERNAME``/``DB_PASSWORD`` are
    ignored for this provider.
    """
    name = (DB_NAME or "").strip()
    if name == ":memory:":
        return "sqlite:///:memory:"
    if not name:
        name = "productarium.db"
    if "/" in name or os.sep in name:
        if ".." in name.replace("\\", "/").split("/"):
            raise ValueError(
                f"DB_NAME must not contain '..' path segments (got {name!r})"
            )
        path = name
    elif DB_HOST and os.path.isdir(DB_HOST):
        path = os.path.join(DB_HOST, name)
    else:
        path = os.path.join(_default_sqlite_dir(), name)
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
        try:
            # Owner-only permissions for a state directory WE created (never
            # chmod pre-existing/shared directories).
            os.chmod(directory, 0o700)
        except OSError:  # pragma: no cover - FS dependent
            pass
    return f"sqlite:///{path}"


def _build_database_url() -> str:
    """Build a SQLAlchemy URL from the configured provider/credentials."""
    provider = (DB_PROVIDER or "postgres").lower()
    if provider in ("postgres", "postgresql"):
        # psycopg (v3) sync driver.
        return (
            f"postgresql+psycopg://{DB_USERNAME}:{DB_PASSWORD}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
    if provider in ("sqlite", "sqlite3"):
        # File-based degraded mode (tests / local runs without Postgres).
        return _sqlite_url()
    # Fallback to an in-memory SQLite DB so the module still imports/tests
    # without a real Postgres instance. This keeps unit tests runnable.
    logger.warning(
        "Unsupported DB_PROVIDER=%r; falling back to in-memory SQLite.",
        provider,
    )
    return "sqlite:///:memory:"


def _engine_connect_args(url: str) -> Dict[str, Any]:
    """Per-provider ``create_engine`` connect args for ``url``.

    SQLite connections are allowed to cross threads (FastAPI serves sync
    dependencies on worker threads while async handlers open their own
    sessions) and get a busy timeout so concurrent writers wait briefly
    instead of failing immediately. Postgres needs no extra args.
    """
    if url.startswith("sqlite"):
        return {"check_same_thread": False, "timeout": 30}
    return {}


DATABASE_URL = _build_database_url()

# ``pool_pre_ping`` avoids stale-connection errors after DB restarts.
# ``future=True`` enables SQLAlchemy 2.0-style behavior.
_engine_kwargs: Dict[str, Any] = {
    "pool_pre_ping": True,
    "future": True,
    "connect_args": _engine_connect_args(DATABASE_URL),
}
if DATABASE_URL == "sqlite:///:memory:":
    # In-memory SQLite is PER-CONNECTION: the default SingletonThreadPool
    # hands every thread its own EMPTY database, so P1-13's off-loop
    # ``asyncio.to_thread`` DB reads would silently see no tables in the
    # no-Postgres fallback config. Share one connection across threads
    # instead (same StaticPool + check_same_thread=False setup as the test
    # suite's isolated_db fixture). Postgres connections are pooled and
    # thread-safe, so this branch never affects the real deployment path.
    from sqlalchemy.pool import StaticPool

    _engine_kwargs.update(
        poolclass=StaticPool,
        connect_args={
            **_engine_kwargs["connect_args"],
            "check_same_thread": False,
        },
    )
engine: Engine = create_engine(DATABASE_URL, **_engine_kwargs)


def _harden_sqlite_connection(dbapi_conn: Any) -> None:
    """Per-connection SQLite hardening (file-based databases only).

    - WAL journal + NORMAL sync: concurrent readers no longer block on the
      writer (the chat-transcript writes vs. session reads pattern).
    - Owner-only permissions (0600) on the database file. Best-effort.
    """
    try:
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()
    except Exception:  # pragma: no cover - defensive
        return
    try:
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA database_list")
            row = cursor.fetchone()
        finally:
            cursor.close()
        # row: (seq, name, file); skip in-memory databases (empty file field).
        if row and len(row) >= 3 and row[2]:
            os.chmod(row[2], 0o600)
    except Exception:  # pragma: no cover - defensive
        pass


if DATABASE_URL.startswith("sqlite") and ":memory:" not in DATABASE_URL:
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _sqlite_hardening(dbapi_conn, _record):  # pragma: no cover - via engine
        _harden_sqlite_connection(dbapi_conn)

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
    pgvector-direct memory backend's ``knowledge_chunks.embedding`` column).
    The HNSW index needs a TYPED ``vector(dim)`` column while this app
    deliberately creates a dimensionless one (any embedder dim, no migration
    on model change) — the pin + index therefore happen lazily on the first
    memory index run (``ensure_embedding_dimension_and_hnsw``). Everything is
    best-effort and non-fatal; on SQLite (tests) it is skipped.
    """
    global _db_ready
    if _db_ready:
        return True
    try:
        _ensure_pgvector_extension()
        Base.metadata.create_all(bind=engine)
        _ensure_schema_columns()
        _ensure_hnsw_index()
        _db_ready = True
        logger.info("SQLAlchemy tables ready (url=%s).", _safe_url(DATABASE_URL))
        return True
    except Exception as e:
        logger.warning("create_all failed (non-fatal): %s", e)
        return False


# Columns added to EXISTING tables after their initial create_all. There is
# no migration machinery in this project (create_all only creates missing
# tables, never alters existing ones), so nullable additions are shimmed
# here idempotently at startup ("already exists" = success, non-fatal).
_SCHEMA_COLUMN_SHIMS: tuple = (
    ("databases", "db_type", "VARCHAR(32)"),
    ("mcp_servers", "preset_key", "VARCHAR(32)"),
    # Documentation versioning pointer (productarium_doc_versions).
    ("codebases", "current_version", "INTEGER"),
    ("databases", "current_version", "INTEGER"),
    ("specs", "current_version", "INTEGER"),
)


def _ensure_schema_columns() -> None:
    """Best-effort ``ALTER TABLE … ADD COLUMN`` for ``_SCHEMA_COLUMN_SHIMS``."""
    # Postgres supports IF NOT EXISTS (avoids ERROR-noise in the server log
    # at every start); SQLite lacks the clause and relies on the catch below.
    if_not_exists = "IF NOT EXISTS" if _is_postgres() else ""
    for table, column, ddl_type in _SCHEMA_COLUMN_SHIMS:
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {if_not_exists} {column} {ddl_type}"
                )
            logger.debug("Schema shim ensured column %s.%s", table, column)
        except Exception as e:
            text = str(e).lower()
            if "already exists" in text or "duplicate column" in text:
                continue
            logger.warning(
                "Schema shim for %s.%s failed (non-fatal): %s", table, column, e
            )


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


def _is_postgres() -> bool:
    """True when the configured provider is Postgres."""
    return (DB_PROVIDER or "").lower() in ("postgres", "postgresql")


def _embedding_column_typmod(conn: Any) -> Optional[int]:
    """``pg_attribute.typmod`` of knowledge_chunks.embedding.

    ``-1`` = dimensionless (the freshly created column), ``None`` = table or
    column missing. The stored typmod convention differs across pgvector
    builds: most report ``dim + 4`` (varlena-style), current builds report
    the bare ``dim`` — callers accept either.
    """
    from sqlalchemy import text

    row = conn.execute(text(
        "SELECT a.atttypmod FROM pg_attribute a "
        "JOIN pg_class c ON a.attrelid = c.oid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE c.relname = 'knowledge_chunks' AND a.attname = 'embedding' "
        "AND n.nspname = current_schema()"
    )).scalar()
    return None if row is None else int(row)


# pgvector cannot index a dimensionless `vector` column, so the pin + index
# run ONCE, when the first real embedding batch reveals the embedder dim.
# Above the vector-index limit the index switches to the halfvec CAST
# expression (pgvector >= 0.7), which HNSW supports up to 4000 dims — covers
# 2560-dim embedders (e.g. Qwen3-Embedding-4B) that previously fell back to
# a sequential scan.
_HNSW_VECTOR_INDEX_MAX_DIM = 2000  # vector_cosine_ops HNSW dimension limit
_HNSW_MAX_DIM = 4000               # halfvec_cosine_ops HNSW dimension limit
_hnsw_lock = threading.Lock()
_hnsw_ready: bool = False
_hnsw_failed: Optional[str] = None
# Cached (cosine ORDER BY expression, query param cast) derived from the live
# index kind — invalidated on every (re)build and by reset_hnsw_state.
_hnsw_distance_expr: Optional[Tuple[str, str]] = None


def _hnsw_index_sql(dim: int) -> str:
    """CREATE INDEX statement for the embedding HNSW index at ``dim``.

    Queries must compare through the SAME expression the index uses — see
    :func:`embedding_distance_expr`.
    """
    target = (
        f"((embedding::halfvec({dim})) halfvec_cosine_ops)"
        if dim > _HNSW_VECTOR_INDEX_MAX_DIM
        else "(embedding vector_cosine_ops)"
    )
    return (
        "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding_hnsw "
        f"ON knowledge_chunks USING hnsw {target}"
    )


def embedding_distance_expr() -> Tuple[str, str]:
    """(cosine ORDER BY expression, param cast) matching the live HNSW index.

    ``("embedding <=> CAST(:q AS vector)", "vector")`` normally; the halfvec
    cast pair when the pinned column exceeds the 2000-dim vector-index limit
    and the existing index is the halfvec expression index. Derived from the
    actual index definition (cached) so the query always hits the index.
    """
    global _hnsw_distance_expr
    if _hnsw_distance_expr is not None:
        return _hnsw_distance_expr
    pair: Tuple[str, str] = ("embedding <=> CAST(:q AS vector)", "vector")
    try:
        if _is_postgres():
            from sqlalchemy import text

            with engine.connect() as conn:
                typmod = _embedding_column_typmod(conn)
                if typmod is not None and typmod > _HNSW_VECTOR_INDEX_MAX_DIM:
                    indexdef = conn.execute(text(
                        "SELECT indexdef FROM pg_indexes "
                        "WHERE indexname = 'ix_knowledge_chunks_embedding_hnsw'"
                    )).scalar()
                    if indexdef and "halfvec" in indexdef:
                        m = re.search(r"halfvec\((\d+)\)", indexdef)
                        dim = m.group(1) if m else str(typmod)
                        pair = (
                            f"(embedding::halfvec({dim})) <=> CAST(:q AS halfvec)",
                            "halfvec",
                        )
        _hnsw_distance_expr = pair
    except Exception as e:  # pragma: no cover - introspection is non-fatal
        logger.debug("embedding distance expression introspection failed: %s", e)
    return pair


def reset_hnsw_state() -> None:
    """Drop the cached HNSW pin/index state (tests)."""
    global _hnsw_ready, _hnsw_failed, _hnsw_distance_expr
    _hnsw_ready = False
    _hnsw_failed = None
    _hnsw_distance_expr = None


def ensure_embedding_dimension_and_hnsw(dim: int) -> bool:
    """Pin ``knowledge_chunks.embedding`` to ``vector(dim)`` + build HNSW.

    The column is created DIMENSIONLESS (any embedder dimension, no migration
    on model change), but pgvector cannot build an HNSW index over a
    dimensionless column — the startup-time ``CREATE INDEX`` always failed
    with ``column does not have dimensions``. The pin + index therefore run
    here, called by the pgvector memory backend once the first real
    embedding batch has revealed the dimension.

    ``ALTER COLUMN ... TYPE vector(dim)`` succeeds while existing rows are
    NULL or share the dimension; mixed dimensions (an embedder change after
    data was stored) abort the pin — the failure is cached and logged once,
    cosine search keeps working over a sequential scan (and the documented
    dimension-mismatch behavior applies). Idempotent, thread-safe,
    non-fatal; returns True when the index exists.
    """
    global _hnsw_ready, _hnsw_failed
    if not _is_postgres():
        return False
    if _hnsw_ready:
        return True
    if _hnsw_failed:
        return False
    if not isinstance(dim, int) or dim <= 0 or dim > _HNSW_MAX_DIM:
        logger.warning(
            "HNSW index skipped: embedding dimension %r outside pgvector's "
            "supported range (1-%d).", dim, _HNSW_MAX_DIM,
        )
        _hnsw_failed = f"unsupported dimension {dim!r}"
        return False
    with _hnsw_lock:
        if _hnsw_ready:
            return True
        if _hnsw_failed:
            return False
        try:
            with engine.begin() as conn:
                typmod = _embedding_column_typmod(conn)
                if typmod is None:
                    # Table/column missing (create_all pending) — retry later.
                    return False
                if typmod < 0:
                    conn.exec_driver_sql(
                        "ALTER TABLE knowledge_chunks "
                        f"ALTER COLUMN embedding TYPE vector({dim})"
                    )
                    logger.info(
                        "Pinned knowledge_chunks.embedding to vector(%d); "
                        "creating the HNSW cosine index.", dim,
                    )
                elif typmod != dim and typmod - 4 != dim:
                    # Already pinned; accept both typmod conventions (bare dim
                    # on current pgvector builds, dim+4 on older ones) so a
                    # correct pin is not misreported as a dimension mismatch.
                    raise RuntimeError(
                        f"knowledge_chunks.embedding typmod is {typmod} "
                        f"but the embedder produces {dim}-dim vectors; reindex "
                        f"or clear the product memory to re-pin"
                    )
                conn.exec_driver_sql(_hnsw_index_sql(dim))
            _hnsw_ready = True
            _hnsw_distance_expr = None
            return True
        except Exception as e:
            _hnsw_failed = str(e)
            logger.warning(
                "HNSW index not created on knowledge_chunks.embedding "
                "(cosine search falls back to a sequential scan; non-fatal): %s", e,
            )
            return False


def _ensure_hnsw_index() -> None:
    """Best-effort HNSW index on knowledge_chunks.embedding (Postgres only).

    pgvector cannot index the dimensionless ``vector`` column this app
    creates (by design: any embedder dimension, no migration on model
    change). When the column is still dimensionless the index creation is
    DEFERRED to the first memory index run
    (``ensure_embedding_dimension_and_hnsw`` pins the dimension once real
    embeddings exist); when the column is already typed (pinned by a previous
    run) the index is created here if missing. Non-fatal.
    """
    global _hnsw_ready
    if not _is_postgres():
        return
    try:
        from api.models import _PGVECTOR_AVAILABLE
        if not _PGVECTOR_AVAILABLE:
            return
    except Exception:
        return
    try:
        with engine.begin() as conn:
            typmod = _embedding_column_typmod(conn)
            if typmod is None or typmod < 0:
                logger.info(
                    "HNSW index deferred: knowledge_chunks.embedding is "
                    "dimensionless until the first memory index run pins "
                    "the embedder dimension (see "
                    "api.db.ensure_embedding_dimension_and_hnsw)."
                )
                return
            # Near/above the 2000-dim limit prefer the dim+4 typmod reading so
            # an over-limit pin gets the halfvec expression index here too.
            dim = typmod - 4 if typmod > _HNSW_VECTOR_INDEX_MAX_DIM else typmod
            conn.exec_driver_sql(_hnsw_index_sql(max(1, dim)))
        _hnsw_ready = True
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
