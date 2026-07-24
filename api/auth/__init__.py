"""Productarium auth package (local + Keycloak, configurable via AUTH_PROVIDER)."""

import os

# AUTH_PROVIDER: local (default) | keycloak | both | none
# - local:    username/password login (passlib bcrypt + JWT session cookie)
# - keycloak: OIDC redirect login via Keycloak
# - both:     local + keycloak endpoints both enabled
# - none:     auth disabled; get_current_user returns a bootstrap/system admin
AUTH_PROVIDER = (os.environ.get("AUTH_PROVIDER") or "local").lower()
