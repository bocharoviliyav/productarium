"""GitLab connector (plan section G).

Wraps :func:`api.data_pipeline.download_repo` for cloning and the GitLab REST
API v4 (``gitlab.com`` or a self-hosted instance) for listing projects.
Credentials come from ``settings_store.get_git_creds("gitlab")``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from api.integrations._git_base import GitConnector
from api.integrations.base import IntegrationConnector

logger = logging.getLogger(__name__)


class GitLabConnector(IntegrationConnector, GitConnector):
    name = "gitlab"
    display_name = "GitLab"
    description = "Clone and document GitLab repositories (cloud or self-hosted)."
    repo_type = "gitlab"
    default_api_base = "https://gitlab.com"

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        from api.settings_store import get_git_creds

        return get_git_creds("gitlab")

    def is_configured(self) -> bool:
        return bool(self.config.get("url") or self.config.get("token"))

    # ---- listing ---------------------------------------------------------
    def list_repos_url(self, api_base: str) -> str:
        base = api_base.rstrip("/")
        return f"{base}/api/v4/projects?membership=true&per_page={self.list_per_page}"

    def auth_headers(self) -> Dict[str, str]:
        token = self._token()
        # GitLab uses the PRIVATE-TOKEN header.
        return {"PRIVATE-TOKEN": token} if token else {}

    def _parse_repo_entry(self, item: Dict[str, Any]) -> Dict[str, Any]:
        entry = super()._parse_repo_entry(item)
        # GitLab's clone URL for the authenticated user uses the web_url base.
        if not entry.get("url") and item.get("http_url_to_repo"):
            entry["url"] = item["http_url_to_repo"]
        return entry

    # ---- IntegrationConnector interface ----------------------------------
    def test(self) -> Dict[str, Any]:
        try:
            import requests
        except Exception as e:  # pragma: no cover
            return {"success": False, "message": f"requests unavailable: {e}"}
        from api.ssl_config import requests_verify

        token = self._token()
        if not token:
            return {"success": True, "message": "No token configured; public repos only."}
        base = self._api_base().rstrip("/")
        url = f"{base}/api/v4/user"
        try:
            from api.timeout_config import resolve_integration_http_timeout
            resp = requests.get(
                url, headers=self.auth_headers(), timeout=resolve_integration_http_timeout(),
                verify=requests_verify(),
            )
            if resp.status_code == 200:
                data = resp.json()
                return {"success": True, "message": f"Authenticated as {data.get('username', '?')}"}
            return {"success": False, "message": f"GitLab API returned HTTP {resp.status_code}"}
        except Exception as e:
            return {"success": False, "message": f"GitLab connection failed: {e}"}

    def list_spaces(self) -> List[Dict[str, Any]]:
        return self.git_list_spaces()

    def pull(self, source_id: str, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.git_pull(source_id, opts)
