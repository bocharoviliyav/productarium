"""GitHub connector (plan section G).

Wraps :func:`api.clients.git.download_repo` for cloning and the GitHub REST
API (public ``api.github.com`` or GitHub Enterprise ``/api/v3``) for listing
repositories. Credentials come from ``settings_store.get_git_creds("github")``.

No live credentials are required to import this module — network calls happen
only inside ``test``/``list_spaces``/``pull`` and degrade gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from api.integrations._git_base import GitConnector
from api.integrations.base import IntegrationConnector

logger = logging.getLogger(__name__)


class GitHubConnector(IntegrationConnector, GitConnector):
    name = "github"
    display_name = "GitHub"
    description = "Clone and document GitHub repositories (public, Enterprise, or via token)."
    repo_type = "github"
    default_api_base = "https://api.github.com"

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        from api.config.settings import get_git_creds

        return get_git_creds("github")

    def is_configured(self) -> bool:
        # GitHub works on any public repo even without config; configured when
        # either an enterprise URL or a token is present.
        return bool(self.config.get("url") or self.config.get("token"))

    # ---- listing ---------------------------------------------------------
    def list_repos_url(self, api_base: str) -> str:
        # GitHub Enterprise exposes the same /user/repos endpoint under /api/v3.
        base = api_base.rstrip("/")
        if "api.github.com" in base:
            return f"{base}/user/repos?per_page={self.list_per_page}&type=all"
        # Enterprise: api_base is the enterprise root (no /api/v3 yet).
        if not base.endswith("/api/v3"):
            base = base + "/api/v3"
        return f"{base}/user/repos?per_page={self.list_per_page}&type=all"

    def auth_headers(self) -> Dict[str, str]:
        token = self._token()
        # GitHub uses `token <PAT>` (also accepts `Bearer <PAT>` for fine-grained).
        return {"Authorization": f"token {token}"} if token else {}

    # ---- IntegrationConnector interface ----------------------------------
    def test(self) -> Dict[str, Any]:
        try:
            import requests
        except Exception as e:  # pragma: no cover
            return {"success": False, "message": f"requests unavailable: {e}"}
        from api.config.ssl import requests_verify

        token = self._token()
        if not token:
            return {"success": True, "message": "No token configured; public repos only."}
        api_base = self._api_base()
        if "api.github.com" not in api_base and not api_base.endswith("/api/v3"):
            api_base = api_base.rstrip("/") + "/api/v3"
        url = f"{api_base.rstrip('/')}/user"
        try:
            from api.config.timeout import resolve_integration_http_timeout
            resp = requests.get(
                url, headers=self.auth_headers(), timeout=resolve_integration_http_timeout(),
                verify=requests_verify(),
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "message": f"Authenticated as {data.get('login', '?')}",
                }
            return {"success": False, "message": f"GitHub API returned HTTP {resp.status_code}"}
        except Exception as e:
            return {"success": False, "message": f"GitHub connection failed: {e}"}

    def list_spaces(self) -> List[Dict[str, Any]]:
        return self.git_list_spaces()

    def pull(self, source_id: str, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.git_pull(source_id, opts)
