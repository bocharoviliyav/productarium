"""Shared base for Git-host connectors (GitHub / GitLab).

This module is underscore-prefixed so the registry's pkgutil auto-discovery
skips it (it is not itself a connector). The two concrete connectors
(:mod:`api.integrations.github`, :mod:`api.integrations.gitlab`) subclass
:class:`GitConnector` and only supply the host-specific API listing details +
the ``repo_type`` slug.

The heavy lifting (authenticated clone with token injection per host) reuses
:func:`api.data_pipeline.download_repo` — we import and wrap it, we do NOT
duplicate or delete the original function (per the foundation contract).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GitConnector:
    """Mixin/base providing clone + markdown-build + repo-name extraction.

    Subclasses set:
        - ``name`` / ``display_name`` / ``description`` (from IntegrationConnector)
        - ``repo_type``: ``"github" | "gitlab"``
        - ``default_api_base``: default REST API base URL for the host
        - ``list_repos_url(api_base, config)``: build the "list my repos" URL
        - ``auth_headers(token)``: build the auth headers for the list API
    """

    repo_type: str = ""
    default_api_base: str = ""
    # Max number of pages to follow when listing repos (keeps listing bounded).
    list_max_pages: int = 5
    list_per_page: int = 100

    # ---- config ----------------------------------------------------------
    def _api_base(self) -> str:
        """Resolve the API base URL from config (enterprise URL) or default."""
        url = (self.config or {}).get("url") or ""
        if url:
            base = url.rstrip("/")
            # GitHub Enterprise API lives under /api/v3; GitLab under /api/v4
            # (handled by subclasses via _api_path).
            return base
        return self.default_api_base

    def _token(self) -> Optional[str]:
        return (self.config or {}).get("token")

    # ---- listing (host-specific hooks, overridden by subclasses) ---------
    def list_repos_url(self, api_base: str) -> str:
        """Build the 'list repositories for the authenticated user' URL."""
        raise NotImplementedError

    def auth_headers(self) -> Dict[str, str]:
        """Auth headers for the listing API (default: Bearer token)."""
        token = self._token()
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _parse_repo_entry(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single repo API result into {id, title, type, url}."""
        # GitHub: {id, full_name, html_url, ...}
        # GitLab: {id, path_with_namespace, web_url, ...}
        title = item.get("full_name") or item.get("path_with_namespace") or item.get("name") or ""
        url = (
            item.get("html_url")
            or item.get("web_url")
            or (item.get("links") or {}).get("html", {}).get("href")
            or ""
        )
        rid = str(item.get("id") or item.get("uuid") or item.get("name") or title)
        return {"id": url or title, "title": title, "type": "repo", "url": url, "repo_id": rid}

    # ---- repo name / clone path ------------------------------------------
    @staticmethod
    def extract_repo_name(repo_url: str, repo_type: str) -> str:
        """owner_repo slug from a URL (mirrors data_pipeline's extraction)."""
        parts = repo_url.rstrip("/").split("/")
        if repo_type in ("github", "gitlab") and len(parts) >= 5:
            owner = parts[-2]
            repo = parts[-1].replace(".git", "")
            return f"{owner}_{repo}"
        return parts[-1].replace(".git", "")

    def _clone_dir(self, repo_url: str) -> str:
        """The local clone path under ~/.adalflow/repos (mirrors data_pipeline)."""
        try:
            from adalflow.utils import get_adalflow_default_root_path

            root = get_adalflow_default_root_path()
        except Exception:
            root = os.path.expanduser("~/.adalflow")
        repo_name = self.extract_repo_name(repo_url, self.repo_type)
        return os.path.join(root, "repos", repo_name)

    # ---- markdown building from a clone ----------------------------------
    @staticmethod
    def _find_readme(repo_dir: str) -> Optional[str]:
        for cand in ("README.md", "README.rst", "README.txt", "README", "readme.md"):
            p = os.path.join(repo_dir, cand)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        return f.read()
                except Exception:
                    continue
        return None

    @staticmethod
    def _file_tree(repo_dir: str, max_depth: int = 2, max_entries: int = 200) -> str:
        """A shallow text tree of the repo (bounded for readability)."""
        lines: List[str] = []
        count = 0

        def walk(dir_path: str, prefix: str, depth: int) -> None:
            nonlocal count
            if depth > max_depth or count >= max_entries:
                return
            try:
                entries = sorted(os.listdir(dir_path))
            except Exception:
                return
            # Skip common noise dirs.
            entries = [e for e in entries if e not in (".git", "node_modules", "__pycache__")]
            for e in entries:
                if count >= max_entries:
                    lines.append(f"{prefix}... (truncated)")
                    return
                full = os.path.join(dir_path, e)
                is_dir = os.path.isdir(full)
                lines.append(f"{prefix}{e}/" if is_dir else f"{prefix}{e}")
                count += 1
                if is_dir:
                    walk(full, prefix + "    ", depth + 1)

        walk(repo_dir, "", 0)
        return "\n".join(lines)

    def _build_markdown(self, repo_url: str, repo_dir: str) -> str:
        title = self.extract_repo_name(repo_url, self.repo_type).replace("_", "/", 1)
        readme = self._find_readme(repo_dir)
        tree = self._file_tree(repo_dir)
        parts = [f"# {title}", "", f"Source: {repo_url}", ""]
        if readme:
            parts += ["## README", "", readme, ""]
        else:
            parts += ["<!-- No README found at repo root -->", ""]
        if tree:
            parts += ["## File tree", "", "```", tree, "```", ""]
        return "\n".join(parts)

    # ---- shared pull -----------------------------------------------------
    def git_pull(self, source_id: str, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Clone a repo via data_pipeline.download_repo and build markdown.

        ``source_id`` is a git URL. ``opts`` may carry ``"token"`` to override
        the configured token (e.g. a per-request token from the caller).
        """
        from api.clients.git import download_repo

        opts = opts or {}
        token = opts.get("token") or self._token()
        local_path = self._clone_dir(source_id)
        # download_repo is idempotent: if the dir exists & is non-empty it reuses it.
        download_repo(source_id, local_path, repo_type=self.repo_type, access_token=token)
        markdown = self._build_markdown(source_id, local_path)
        title = self.extract_repo_name(source_id, self.repo_type).replace("_", "/", 1)
        return {
            "title": title,
            "markdown": markdown,
            "attachments": [],
            "repo_url": source_id,
            "repo_type": self.repo_type,
            "local_path": local_path,
        }

    # ---- shared listing with pagination ----------------------------------
    def git_list_spaces(self) -> List[Dict[str, Any]]:
        """List repos for the authenticated user (bounded pagination)."""
        try:
            import requests  # local import keeps the module import-safe
        except Exception as e:  # pragma: no cover - requests is a core dep
            logger.warning("requests unavailable; cannot list repos for %s: %s", self.repo_type, e)
            return []
        from api.ssl_config import requests_verify

        token = self._token()
        if not token:
            logger.info("No token configured for %s; listing no repos.", self.repo_type)
            return []
        api_base = self._api_base()
        url = self.list_repos_url(api_base)
        headers = self.auth_headers()
        out: List[Dict[str, Any]] = []
        try:
            for _ in range(self.list_max_pages):
                if not url:
                    break
                from api.timeout_config import resolve_integration_http_timeout
                resp = requests.get(
                    url, headers=headers, timeout=resolve_integration_http_timeout(),
                    verify=requests_verify(),
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "%s list repos %s -> HTTP %s: %s",
                        self.repo_type, url, resp.status_code, resp.text[:200],
                    )
                    break
                data = resp.json()
                results = data.get("results") if isinstance(data, dict) else data
                if not isinstance(results, list):
                    break
                for item in results:
                    if isinstance(item, dict):
                        out.append(self._parse_repo_entry(item))
                # Follow next-link pagination if present.
                url = ""
                if isinstance(data, dict):
                    nxt = data.get("next") or (data.get("links") or {}).get("next")
                    if isinstance(nxt, dict):
                        nxt = nxt.get("href")
                    url = nxt or ""
        except Exception as e:
            logger.warning("Error listing repos for %s: %s", self.repo_type, e)
            return out
        return out
