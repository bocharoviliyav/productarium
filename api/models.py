"""SQLAlchemy 2.0 ORM models for Productarium persistence.

Defines the product-centric data model:

- ``UserORM``          — local + Keycloak users (admin/user roles)
- ``ProductORM``       — top-level product (no ``type``; +summary, +owner_id)
- ``ArtifactORM``      — codebase|spec|links|documentation|guides (+kind, +verified*, +source)
- ``KnowledgeNodeORM`` — Confluence-like tree of knowledge pages per product
- ``SettingORM``       — admin config key/value store (optionally encrypted)
- ``ApiTokenORM``      — public API tokens for external integrations

String primary keys (``prod_..``, ``art_..``, ``user_..``, ``node_..``,
``tok_..``) keep frontend compatibility. All tables share the same
Postgres+pgvector database used by cognee (see ``api/db.py``). ``init_db`` is
idempotent and non-fatal (see the one-shot migration in ``api/db.py``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, Boolean, ForeignKey, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models (used by db.init_db)."""
    pass


# --- Legacy artifact type -> new (type, kind) mapping -----------------------
# Applied by the one-shot migration in api/db.py AND by _artifact_orm_from_pydantic
# in api/api.py so the data stays consistent regardless of which client wrote it.
LEGACY_ARTIFACT_TYPE_MAP: dict[str, tuple[str, str]] = {
    "openapi": ("spec", "openapi"),
    "asyncapi": ("spec", "asyncapi"),
    "testcase": ("documentation", "testcase"),
}
# The new artifact type enum (contract J).
ARTIFACT_TYPES: tuple[str, ...] = (
    "codebase", "spec", "links", "documentation", "guides",
)


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

    artifacts: Mapped[list["ArtifactORM"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )
    knowledge_nodes: Mapped[list["KnowledgeNodeORM"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<ProductORM id={self.id!r} name={self.name!r}>"


class ArtifactORM(Base):
    """ORM model for the ``artifacts`` table (belongs to a Product)."""

    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    # New enum: codebase|spec|links|documentation|guides
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    # Subtype (e.g. openapi/asyncapi for spec). Nullable.
    kind: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    repo_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    repo_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    allure_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    generated_docs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Stored in a JSON column; serialized/deserialized transparently.
    pages: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Verified flag (item 5) + audit.
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verified_by: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("productarium_users.id", ondelete="SET NULL"),
        nullable=True,
    )
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Provenance: manual|generated|api|mcp
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

    product: Mapped["ProductORM"] = relationship(back_populates="artifacts")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"<ArtifactORM id={self.id!r} name={self.name!r} "
            f"type={self.type!r} kind={self.kind!r}>"
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
    artifact_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("artifacts.id", ondelete="SET NULL"),
        nullable=True,
    )
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
