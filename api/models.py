"""SQLAlchemy 2.0 ORM models for Productarium persistence.

Defines the product-centric data model:

- ``UserORM``          — local + Keycloak users (admin/manager/viewer_global/user roles)
- ``ProductGrantORM``  — per-product access grants (ro/rw) for non-admin users
- ``ProductORM``       — top-level product (no ``type``; +summary, +owner_id)
- ``CodebaseORM``      — git repository artifact (repo clone, page tree, generated docs)
- ``SpecORM``          — OpenAPI/AsyncAPI spec artifact (single yaml/json)
- ``LinksORM``         — curated external links (kv pairs)
- ``DatabaseORM``      — reverse-engineered database artifact (masked DSN, MCP)
- ``KnowledgeNodeORM`` — Confluence-like tree of knowledge pages per product
- ``SettingORM``       — admin config key/value store (optionally encrypted)
- ``ApiTokenORM``      — public API tokens for external integrations
- ``ChatSessionORM``   — expert-agent chat session per product (Wave B)
- ``ChatMessageORM``   — one transcript message of a chat session
- ``KnowledgeChunkORM`` — embedded text chunks for the pgvector-direct memory
- ``McpServerORM``       — admin registry of external MCP servers (Wave C)
- ``ProductMcpServerORM`` — per-product binding to an MCP server (Wave C)

String primary keys (``prod_..``, ``art_..``, ``user_..``, ``node_..``,
``tok_..``) keep frontend compatibility. All tables share the same
Postgres+pgvector database (see ``api/db.py``). ``init_db`` is idempotent and
non-fatal.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

logger = logging.getLogger(__name__)

# pgvector may be absent in minimal venvs. Import guarded so the module
# always imports; the real Vector column type is only used on Postgres
# (load_dialect_impl selects it by dialect).
try:
    from pgvector.sqlalchemy import Vector as _PgVector  # type: ignore
    _PGVECTOR_AVAILABLE = True
except Exception:  # pragma: no cover - dep missing in minimal venv
    _PgVector = None  # type: ignore
    _PGVECTOR_AVAILABLE = False


class VectorType(TypeDecorator):
    """Dialect-adaptive embedding column: pgvector ``Vector`` on Postgres, ``Text`` elsewhere.

    Uses a dimensionless pgvector ``Vector()`` (no fixed dim) so a change of
    embedder model / dimension does not require a migration. Cosine search
    works as long as the query vector and stored vectors share a dimension;
    otherwise the operator raises at query time (the caller returns "" on any
    error). On SQLite (tests) the column degrades to ``Text`` and vector
    operations are skipped by the pgvector backend.

    ``process_bind_param`` serializes list/tuple embeddings to a
    ``"[1.0,2.0,...]"`` string literal on the Text fallback dialect so the
    column is writable on SQLite (tests) and when pgvector is absent. On
    Postgres with pgvector the list is passed through to ``_PgVector`` which
    binds it natively (and also accepts the string literal form).
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name in ("postgresql", "postgres") and _PGVECTOR_AVAILABLE:
            return dialect.type_descriptor(_PgVector())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # On Postgres + pgvector, delegate to _PgVector's own bind processor
        # (it accepts lists and string literals).
        if dialect.name in ("postgresql", "postgres") and _PGVECTOR_AVAILABLE:
            return value
        # Text fallback (SQLite / pgvector absent): serialize lists to the
        # "[1.0,2.0]" string literal so the column is writable. Non-list values
        # (already a string) are stored as-is.
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(str(float(x)) for x in value) + "]"
        return value


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models (used by db.init_db)."""
    pass


# The spec subtype enum (openapi|asyncapi) carried on SpecORM.kind.
SPEC_KINDS: tuple[str, ...] = ("openapi", "asyncapi")

# Valid user roles (P0-2 role model) — see UserORM.role.
USER_ROLES: tuple[str, ...] = ("user", "admin", "manager", "viewer_global")


class UserORM(Base):
    """ORM model for the ``users`` table (local + Keycloak users)."""

    __tablename__ = "productarium_users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    email: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, unique=True)
    # Null for Keycloak users (no local password).
    password_hash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 'user' | 'admin' | 'manager' | 'viewer_global' (P0-2 role model):
    #   admin         — full access incl. the admin UI
    #   manager       — create + fill products (no admin UI)
    #   viewer_global — read-only on all products
    #   user          — own products + per-product grants only
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    # 'local' | 'keycloak'
    provider: Mapped[str] = mapped_column(String(16), nullable=False, default="local")
    # Keycloak `sub` claim (null for local users).
    provider_subject: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    # Password reset / temp-password flow (local users). The reset token is
    # stored as a sha256 hash (never plaintext); ``reset_token_expires`` is the
    # UTC expiry. ``must_change_password`` is set when an admin creates a user
    # with a temporary password so the UI can force a change on first login.
    reset_token_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    reset_token_expires: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<UserORM id={self.id!r} username={self.username!r} role={self.role!r}>"


class ProductORM(Base):
    """ORM model for the ``products`` table (no ``type`` column)."""

    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # AI-generated summary (item 4). Nullable until generated.
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Owner of the product (FK users.id, SET NULL on delete). Nullable, indexed.
    owner_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    codebases: Mapped[list["CodebaseORM"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )
    specs: Mapped[list["SpecORM"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )
    links: Mapped[list["LinksORM"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )
    databases: Mapped[list["DatabaseORM"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )
    knowledge_nodes: Mapped[list["KnowledgeNodeORM"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )
    mcp_bindings: Mapped[list["ProductMcpServerORM"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<ProductORM id={self.id!r} name={self.name!r}>"


class CodebaseORM(Base):
    """ORM model for the ``codebases`` table — a git repo documented from source.

    The complex artifact: repo cloning, a JSON tree of generated wiki pages,
    and the generated docs blob. Owned by exactly one Product.
    """

    __tablename__ = "codebases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    repo_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    repo_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    generated_docs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # JSON tree of generated wiki pages, keyed by page id.
    pages: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verified_by: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="SET NULL"),
        nullable=True,
    )
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    # Active documentation version (productarium_doc_versions.version);
    # NULL until the first snapshot (pre-versioning legacy artifacts).
    current_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    product: Mapped["ProductORM"] = relationship(back_populates="codebases")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<CodebaseORM id={self.id!r} name={self.name!r} repo_url={self.repo_url!r}>"


class SpecORM(Base):
    """ORM model for the ``specs`` table — a single OpenAPI/AsyncAPI spec.

    Simple artifact: one yaml/json ``content`` string, rendered by the UI.
    ``kind`` distinguishes openapi from asyncapi.
    """

    __tablename__ = "specs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False, default="openapi")
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verified_by: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="SET NULL"),
        nullable=True,
    )
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    # Active documentation version (specs snapshot their single `content`).
    current_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    product: Mapped["ProductORM"] = relationship(back_populates="specs")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<SpecORM id={self.id!r} name={self.name!r} kind={self.kind!r}>"


class LinksORM(Base):
    """ORM model for the ``links`` table — curated external link pairs.

    Simplest artifact: ``content`` holds a JSON array of {url, description}.
    """

    __tablename__ = "links"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verified_by: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="SET NULL"),
        nullable=True,
    )
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    product: Mapped["ProductORM"] = relationship(back_populates="links")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<LinksORM id={self.id!r} name={self.name!r}>"


class DatabaseORM(Base):
    """ORM model for the ``databases`` table — a database documented via MCP
    reverse-engineering (Wave E).

    Mirrors :class:`CodebaseORM`: the reverse-engineering flow (MCP
    introspection tools → deterministic schema skeleton → LLM enrichment →
    verification) writes ``generated_docs`` (the full markdown blob) and
    ``pages`` (the JSON page tree rendered by the viewer).

    Secrets: the connection DSN is accepted from the client and — on the
    preset path (``db_type`` set, ``api/mcp/presets.py``) — is NEVER
    persisted at all: it lives only in the dedicated preset MCP server row's
    Fernet-encrypted ``env``, and ``dsn_masked`` stays NULL. On the legacy
    path it is masked via ``api.docgen.verification.mask_dsn`` and only the
    masked form (``dsn_masked``) is persisted; the raw DSN is never stored,
    logged, or returned. ``mcp_server_id`` optionally pins the registry MCP
    server whose introspection tools the flow uses (NULL = all of the
    product's bound enabled servers); the FK is SET NULL so deleting the
    server row does not cascade into the documented database — EXCEPT for
    preset servers, which the databases router deletes explicitly together
    with the database row.
    """

    __tablename__ = "databases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    # Preset database type (postgresql|mysql|mariadb|sqlserver|sqlite|oracle)
    # when the DB was added via the preset flow; NULL = legacy manual path.
    db_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # Masked connection DSN (secret part replaced with ***REDACTED***).
    dsn_masked: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Optional pin to a registered MCP server (introspection source).
    mcp_server_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("mcp_servers.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    generated_docs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # JSON tree of generated pages, keyed by page id (viewer contract).
    pages: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verified_by: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="SET NULL"),
        nullable=True,
    )
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    # Active documentation version (productarium_doc_versions.version).
    current_version: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    product: Mapped["ProductORM"] = relationship(back_populates="databases")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<DatabaseORM id={self.id!r} name={self.name!r}>"


class DocVersionORM(Base):
    """Immutable documentation snapshot of one artifact (Vault KV-v2 style).

    Every successful generation, manual edit and rollback APPENDS a row; the
    artifact row's ``current_version`` points at the active snapshot. Payload
    columns are filled per entity type: codebase/database store
    ``generated_docs`` + ``pages``; spec stores its single ``content``. The
    vector memory stays keyed to the artifact (delete-then-insert by
    source_id), so only the CURRENT version is searchable.
    """

    __tablename__ = "productarium_doc_versions"
    __table_args__ = (
        UniqueConstraint(
            "entity_type", "entity_id", "version", name="uq_doc_version_entity"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # codebase | database | spec
    entity_type: Mapped[str] = mapped_column(String(16), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    # baseline | generate | edit | rollback
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    generated_docs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pages: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )


class KnowledgeNodeORM(Base):
    """Confluence-like tree node of knowledge pages scoped to a product."""

    __tablename__ = "knowledge_nodes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Self-referential parent (subtree deleted via DB ON DELETE CASCADE).
    parent_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("knowledge_nodes.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False)
    content_md: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # page|folder|branch
    node_type: Mapped[str] = mapped_column(String(32), nullable=False, default="page")
    # Plain nullable id (no FK) pointing at a codebase/spec/links id. A DB-level
    # FK was dropped when the polymorphic ``artifacts`` table was split into
    # codebases/specs/links; the app already tolerates stale refs (orphan
    # handling in the tree builder), so no meaningful integrity is lost.
    artifact_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verified_by: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="SET NULL"),
        nullable=True,
    )
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    product: Mapped["ProductORM"] = relationship(back_populates="knowledge_nodes")
    # Self-referential adjacency list. foreign_keys is explicit so SQLAlchemy can
    # disambiguate the single self-FK (no cascade: subtree deletion is handled by
    # the DB-level ON DELETE CASCADE on parent_id).
    children: Mapped[list["KnowledgeNodeORM"]] = relationship(
        "KnowledgeNodeORM",
        back_populates="parent",
        foreign_keys="KnowledgeNodeORM.parent_id",
    )
    parent: Mapped[Optional["KnowledgeNodeORM"]] = relationship(
        "KnowledgeNodeORM",
        back_populates="children",
        remote_side="KnowledgeNodeORM.id",
        foreign_keys="KnowledgeNodeORM.parent_id",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<KnowledgeNodeORM id={self.id!r} title={self.title!r} "
            f"node_type={self.node_type!r}>"
        )


class ProductGrantORM(Base):
    """Per-product access grant (P0-2): user -> product at level ``ro`` | ``rw``.

    Grants complement ownership: a plain ``user`` (or ``manager``) gets read or
    write access to products they do not own. Composite PK (product_id,
    user_id) so a user has at most one grant per product. Both FKs cascade:
    deleting the product or the user removes the grant.
    """

    __tablename__ = "product_grants"

    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    # 'ro' | 'rw'
    level: Mapped[str] = mapped_column(String(8), nullable=False, default="ro")
    granted_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<ProductGrantORM product={self.product_id!r} user={self.user_id!r} "
            f"level={self.level!r}>"
        )


class SettingORM(Base):
    """Admin config key/value store (optionally encrypted; see settings_store)."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    encrypted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<SettingORM key={self.key!r} encrypted={self.encrypted}>"


class ApiTokenORM(Base):
    """Public API token (hashed) for external integrations."""

    __tablename__ = "api_tokens"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<ApiTokenORM id={self.id!r} name={self.name!r} user_id={self.user_id!r}>"


class ChatSessionORM(Base):
    """ORM model for the ``chat_sessions`` table — a persistent expert-agent
    chat conversation scoped to a product (Wave B).

    Each session owns an ordered list of :class:`ChatMessageORM` rows and is
    the stable key for the LangGraph checkpointer thread
    (``thread_id = session.id``), so the agent's internal message state and the
    user-visible transcript persist together.
    """

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # The user who owns the conversation (auth ``get_current_user``).
    user_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    messages: Mapped[list["ChatMessageORM"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessageORM.created_at",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<ChatSessionORM id={self.id!r} product_id={self.product_id!r}>"
        )


class ChatMessageORM(Base):
    """ORM model for the ``chat_messages`` table — one turn in a chat session.

    Stores the user-visible transcript (user queries + assistant answers + a
    short summary of each tool call) so the UI history endpoints can render
    the conversation without replaying the LangGraph checkpoint state.
    """

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # user | assistant | tool
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Tool-call metadata for role='tool': tool name + args/result summary.
    tool_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    tool_args: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    session: Mapped["ChatSessionORM"] = relationship(back_populates="messages")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<ChatMessageORM id={self.id!r} session_id={self.session_id!r} "
            f"role={self.role!r}>"
        )


class ChatAttachmentORM(Base):
    """ORM model for the ``chat_attachments`` table — a file attached to an
    expert chat turn, stored as its converted Markdown rendition.

    Conversation-context only: the rendition is inlined into the ask-turn
    runner query (never indexed into the product memory). ``session_id`` /
    ``message_id`` are linked best-effort when the turn persists its user
    row; both cascade so deleting a session cleans its attachments.
    """

    __tablename__ = "chat_attachments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    message_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(256), nullable=False)
    mime: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_md: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<ChatAttachmentORM id={self.id!r} product_id={self.product_id!r} "
            f"filename={self.filename!r}>"
        )


class McpServerORM(Base):
    """ORM model for the ``mcp_servers`` table — admin registry of external
    MCP (Model Context Protocol) servers whose tools can be attached to
    products (Wave C).

    ``transport`` is either ``http`` (streamable HTTP endpoint at ``url``),
    ``sse`` (legacy SSE endpoint at ``url``) or ``stdio`` (subprocess launched
    from ``command`` + ``args``).

    Secrets: ``headers`` (http) and ``env`` (stdio) hold Fernet-ENCRYPTED JSON
    ciphertext (see ``api/mcp/secrets.py``) — never plaintext, never returned
    to API clients (responses expose masked key-only views).

    ``status`` is the outcome of the last admin health-check
    (``ok`` | ``error`` | ``unknown``), stamped by ``status_checked_at`` and
    detailed (short, sanitized) in ``status_error``.
    """

    __tablename__ = "mcp_servers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    # Preset key (postgresql|mysql|mariadb|sqlserver|sqlite|oracle) for
    # SYSTEM-MANAGED rows created by the databases preset flow
    # (``api/mcp/presets.py``); NULL = a regular admin-registered server.
    # Preset rows are immutable via the admin API and validated at connect
    # time by exact registry match instead of the interpreter ban.
    preset_key: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # 'http' | 'stdio'
    transport: Mapped[str] = mapped_column(String(16), nullable=False)
    url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    command: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    args: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Encrypted JSON dicts (ciphertext strings inside the JSON column).
    headers: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    env: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # 'ok' | 'error' | 'unknown' (last health-check outcome)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="unknown")
    status_checked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    bindings: Mapped[list["ProductMcpServerORM"]] = relationship(
        back_populates="server",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<McpServerORM id={self.id!r} name={self.name!r} "
            f"transport={self.transport!r}>"
        )


class ProductMcpServerORM(Base):
    """ORM model for the ``product_mcp_servers`` table — a binding between a
    product and a registered MCP server (Wave C).

    ``allowed_tools`` is a JSON list of tool names exposed to the product's
    expert agent; ``None`` means ALL tools of the server are allowed.
    Both sides cascade: deleting the product or the server removes bindings.
    """

    __tablename__ = "product_mcp_servers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    mcp_server_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # JSON list[str] | None — None means all tools of the server are allowed.
    allowed_tools: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    product: Mapped["ProductORM"] = relationship(back_populates="mcp_bindings")
    server: Mapped["McpServerORM"] = relationship(back_populates="bindings")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<ProductMcpServerORM id={self.id!r} product_id={self.product_id!r} "
            f"mcp_server_id={self.mcp_server_id!r}>"
        )


class HttpIntegrationORM(Base):
    """ORM model for the ``http_integrations`` table — admin registry of
    templated read-only GET integrations exposed to agents as named tools
    (issue #3).

    Each row is one HTTP GET endpoint: ``url_template`` may carry
    ``{placeholders}`` matching the declared ``variables`` (plus the implicit
    ``{product_name}``). ``headers`` holds a Fernet-ENCRYPTED ciphertext
    string (see ``api/mcp/secrets.py``) — never plaintext, never returned to
    API clients (responses expose a masked key-only view).

    ``variables`` is a JSON list of ``{name, description, default}`` dicts;
    the agent-facing tool parameters are generated from it.
    """

    __tablename__ = "http_integrations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url_template: Mapped[str] = mapped_column(String(512), nullable=False)
    # Fernet ciphertext string (api/mcp/secrets.encrypt_secret_dict).
    headers: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # JSON list[{name, description, default}] — the tool parameter contract.
    variables: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<HttpIntegrationORM id={self.id!r} name={self.name!r} "
            f"enabled={self.enabled!r}>"
        )


class KnowledgeChunkORM(Base):
    """ORM model for the ``knowledge_chunks`` table — a single embedded text
    chunk scoped to a product, used by the pgvector-direct memory backend.

    Each chunk is produced by chunking a source document (codebase generated
    docs, spec content, knowledge node markdown, integration-pulled text) with
    the shared ``TextSplitter`` config and embedding it via the configured
    embedder. The ``embedding`` column is a pgvector ``Vector`` on Postgres
    (dimensionless — any embedder dim) and degrades to ``Text`` on SQLite.

    Product isolation is enforced by ``product_id`` filtering in every query;
    the HNSW index on ``embedding`` (dimension pinned + index created lazily
    by ``api.db.ensure_embedding_dimension_and_hnsw`` once the first real
    embedding batch reveals the embedder dimension) accelerates the
    cosine-distance ``ORDER BY`` within a product.
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # codebase | spec | links | knowledge_node | integration
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="codebase")
    # The id of the codebase / spec / links / knowledge_node that produced this
    # chunk (nullable: raw integration text has no owning entity row). Used by
    # the upsert path to delete-and-reinsert chunks for a single source.
    source_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    chunk_index: Mapped[int] = mapped_column(nullable=False, default=0)
    # Citation/provenance columns (Wave D). All nullable so pre-existing rows
    # and pre-existing DATABASES (create_all adds them only to fresh tables)
    # stay valid: the pgvector backend writes them only when the columns are
    # actually present in the live schema (see
    # ``api.memory.pgvector_backend._citation_columns_available``).
    # Stable citation id resolvable back to the chunk ("c:<source_id>:<idx>").
    chunk_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    # Repo-relative path of the file the chunk was produced from (when known).
    source_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # [start, end] character offsets of the chunk within the source document
    # (JSON; None when the offsets could not be located).
    char_span: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[object]] = mapped_column(VectorType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<KnowledgeChunkORM id={self.id!r} product_id={self.product_id!r} "
            f"source_type={self.source_type!r} chunk_index={self.chunk_index!r}>"
        )
