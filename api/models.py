"""SQLAlchemy 2.0 ORM models for Productarium persistence.

Defines the product-centric data model:

- ``UserORM``          — local + Keycloak users (admin/user roles)
- ``ProductORM``       — top-level product (no ``type``; +summary, +owner_id)
- ``CodebaseORM``      — git repository artifact (repo clone, page tree, generated docs)
- ``SpecORM``          — OpenAPI/AsyncAPI spec artifact (single yaml/json)
- ``LinksORM``         — curated external links (kv pairs)
- ``KnowledgeNodeORM`` — Confluence-like tree of knowledge pages per product
- ``SettingORM``       — admin config key/value store (optionally encrypted)
- ``ApiTokenORM``      — public API tokens for external integrations
- ``KnowledgeChunkORM`` — embedded text chunks for the pgvector-direct memory

String primary keys (``prod_..``, ``art_..``, ``user_..``, ``node_..``,
``tok_..``) keep frontend compatibility. All tables share the same
Postgres+pgvector database used by cognee (see ``api/db.py``). ``init_db`` is
idempotent and non-fatal.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

logger = logging.getLogger(__name__)

# pgvector is a transitive dep (via cognee) but may be absent in minimal venvs.
# Import guarded so the module always imports; the real Vector column type is
# only used on Postgres (load_dialect_impl selects it by dialect).
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


class UserORM(Base):
    """ORM model for the ``users`` table (local + Keycloak users)."""

    __tablename__ = "productarium_users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    email: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, unique=True)
    # Null for Keycloak users (no local password).
    password_hash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # 'user' | 'admin'
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
    knowledge_nodes: Mapped[list["KnowledgeNodeORM"]] = relationship(
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


class KnowledgeChunkORM(Base):
    """ORM model for the ``knowledge_chunks`` table — a single embedded text
    chunk scoped to a product, used by the pgvector-direct memory backend.

    Each chunk is produced by chunking a source document (codebase generated
    docs, spec content, knowledge node markdown, integration-pulled text) with
    the shared ``TextSplitter`` config and embedding it via the configured
    embedder. The ``embedding`` column is a pgvector ``Vector`` on Postgres
    (dimensionless — any embedder dim) and degrades to ``Text`` on SQLite.

    Product isolation is enforced by ``product_id`` filtering in every query;
    the HNSW index on ``embedding`` (created in ``init_db``) accelerates the
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
