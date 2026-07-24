"""One-shot bootstrap admin creation on startup (non-fatal).

Called from ``api.api.startup_event``. If ``AUTH_PROVIDER != none`` and both
``BOOTSTRAP_ADMIN_USERNAME`` and ``BOOTSTRAP_ADMIN_PASSWORD`` are set, creates
an admin user (or promotes an existing user with that username to admin) when
no admin exists yet. Idempotent, logged, never raises.
"""

from __future__ import annotations

import logging
import os
import uuid

from api.auth import AUTH_PROVIDER
from api.auth.local import hash_password

logger = logging.getLogger(__name__)


def bootstrap_admin() -> bool:
    """Create/promote an admin user from BOOTSTRAP_ADMIN_* env. One-shot, non-fatal."""
    if AUTH_PROVIDER == "none":
        logger.info("AUTH_PROVIDER=none; skipping bootstrap admin.")
        return False
    username = os.environ.get("BOOTSTRAP_ADMIN_USERNAME")
    password = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD")
    if not (username and password):
        logger.info(
            "BOOTSTRAP_ADMIN_USERNAME/PASSWORD not set; skipping bootstrap admin."
        )
        return False
    try:
        from api.db import SessionLocal
        from api.models import UserORM
        with SessionLocal() as db:
            existing_admin = db.query(UserORM).filter(UserORM.role == "admin").first()
            if existing_admin is not None:
                logger.info(
                    "Bootstrap admin: an admin already exists (%s); skipping.",
                    existing_admin.username,
                )
                return False
            existing_user = (
                db.query(UserORM).filter(UserORM.username == username).first()
            )
            if existing_user is not None:
                existing_user.role = "admin"
                existing_user.password_hash = hash_password(password)
                db.commit()
                logger.info(
                    "Bootstrap admin: promoted existing user %r to admin.", username
                )
                return True
            user = UserORM(
                id=f"user_{uuid.uuid4().hex[:24]}",
                username=username,
                password_hash=hash_password(password),
                role="admin",
                provider="local",
            )
            db.add(user)
            db.commit()
            logger.warning(
                "Bootstrap admin: created admin user %r. Please change the password.",
                username,
            )
            return True
    except Exception as e:
        logger.warning("Bootstrap admin failed (non-fatal): %s", e)
        return False
