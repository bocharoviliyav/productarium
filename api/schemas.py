"""Shared Pydantic models — the Productarium API contract (contract J).

Wave 2 routers import these instead of ``api.api`` to avoid circular imports.
``api.api`` re-imports ``Product`` / ``Artifact`` from here so the existing
product/artifact endpoints keep their ``response_model`` working.

Shapes:
- ``Product``  — no ``type``; +summary, +owner_id
- ``Artifact`` — type enum codebase|spec|links|documentation|guides; +kind,
                 +verified, +verified_by, +verified_at, +source
- ``User``     — id/username/email/role/provider
- ``KnowledgeNode`` — Confluence-like tree node
- ``ApiToken`` / ``Setting`` — admin/public API helpers
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


# --- Artifact / Product -----------------------------------------------------
class Artifact(BaseModel):
    id: str
    name: str
    # codebase|spec|links|documentation|guides (legacy openapi/asyncapi/testcase
    # are normalized to spec/documentation + kind on write).
    type: str
    kind: Optional[str] = None  # subtype, e.g. openapi/asyncapi for spec
    repo_url: Optional[str] = None
    repo_type: Optional[str] = None
    token: Optional[str] = None
    content: Optional[str] = None
    allure_url: Optional[str] = None
    generated_docs: Optional[str] = None
    pages: Optional[Dict[str, Any]] = None
    verified: bool = False
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    source: str = "manual"


class Product(BaseModel):
    id: str
    name: str
    description: str = ""
    summary: Optional[str] = None
    owner_id: Optional[str] = None
    artifacts: List[Artifact] = []


# --- User -------------------------------------------------------------------
class UserBase(BaseModel):
    username: str
    email: Optional[str] = None
    role: str = "user"  # user|admin
    provider: str = "local"  # local|keycloak


class UserCreate(UserBase):
    password: Optional[str] = None  # local users only


class UserOut(UserBase):
    id: str
    created_at: Optional[datetime] = None
    # True when the user must change their password on next login (admin-created
    # temp password). Surfaced so the UI can force a change-password prompt.
    must_change_password: bool = False


class LoginRequest(BaseModel):
    username: str
    password: str


class SetupStatus(BaseModel):
    """Tells the UI whether the first-run admin setup flow is needed."""
    setup_required: bool
    auth_provider: str


class SetupRequest(BaseModel):
    """First-run admin creation (only allowed when no local users exist)."""
    username: str
    password: str
    email: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    """Authenticated password change (old -> new)."""
    old_password: str
    new_password: str


class ResetPasswordRequest(BaseModel):
    """Public password reset via a one-time reset token."""
    token: str
    new_password: str


class UserCreateAdmin(BaseModel):
    """Admin creates a local user (optionally with a temp password)."""
    username: str
    email: Optional[str] = None
    role: str = "user"  # user|admin
    # Optional temp password; if omitted a random one is generated and returned.
    password: Optional[str] = None
    must_change_password: bool = True


class UserCreateResult(BaseModel):
    """Result of admin user creation: the new user + credentials shown once."""
    user: UserOut
    temp_password: Optional[str] = None
    reset_token: Optional[str] = None


# --- KnowledgeNode ----------------------------------------------------------
class KnowledgeNode(BaseModel):
    id: str
    product_id: str
    parent_id: Optional[str] = None
    title: str
    slug: str
    content_md: Optional[str] = None
    node_type: str = "page"  # page|folder|branch
    artifact_id: Optional[str] = None
    source: str = "manual"
    verified: bool = False
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class KnowledgeNodeCreate(BaseModel):
    parent_id: Optional[str] = None
    title: str
    slug: Optional[str] = None  # derived from title if omitted
    content_md: Optional[str] = None
    node_type: str = "page"
    artifact_id: Optional[str] = None
    source: str = "manual"


class KnowledgeNodeUpdate(BaseModel):
    title: Optional[str] = None
    slug: Optional[str] = None
    content_md: Optional[str] = None
    node_type: Optional[str] = None
    # Move the node under a different parent (drag-and-drop). None moves it to
    # the product root. Validated against same-product + cycle rules in the
    # router; an empty string is treated as None for client convenience.
    parent_id: Optional[str] = None


# --- ApiToken ---------------------------------------------------------------
class ApiTokenCreate(BaseModel):
    name: str


class ApiTokenOut(BaseModel):
    id: str
    name: str
    created_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    # The raw token is only populated once, at creation time.
    token: Optional[str] = None


# --- Setting ----------------------------------------------------------------
class SettingOut(BaseModel):
    key: str
    value: Optional[str] = None
    encrypted: bool = False


class SettingUpdate(BaseModel):
    value: Optional[str] = None
    encrypt: bool = False
