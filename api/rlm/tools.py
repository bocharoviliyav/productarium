"""Self-contained Python tools injected into the fast-rlm root REPL.

These callables are registered with ``fast_rlm.run(..., tools=[...])`` so the
RLM agent can pull exactly what it needs (a specific file, a knowledge slice,
the spec list) instead of having the whole corpus stringified into the prompt.

CRITICAL — fast-rlm runs these inside an isolated Pyodide REPL, NOT the host
process. That means:
- They CANNOT import host modules (SQLAlchemy, psycopg, adalflow, the api.*
  package). Only modules available in the Pyodide environment are importable.
- They CANNOT close over call-site variables (no closures over outer state).
- They reach product data by calling back to Productarium over HTTP via
  ``pyodide.http.open_url`` (sync), which hits the localhost-only internal
  endpoints in ``api/routers/rlm_tools.py``.

That is why every function:
- does ALL its imports INSIDE the body (Pyodide lazily resolves them at call
  time);
- reads the API base / product id / codebase id from ``os.environ`` (injected
  by fast-rlm via ``env_variables``, so no closure over call-site state);
- returns a plain ``list`` / ``str`` / ``dict`` (JSON-decoded) so the agent
  can index the result directly;
- catches every exception and returns ``[]`` / ``""`` so a tool failure never
  breaks the RLM run (the agent just sees an empty result and moves on).

``build_expert_tools`` / ``build_docgen_tools`` assemble the tool lists for the
two scenarios and are called from the host process (``api.expert.generate`` /
``api.docgen.codebase``) — they only reference the function objects by name,
never calling them in the host.
"""

from __future__ import annotations

import logging
from typing import Callable, List

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tool functions. These are intentionally module-level (not nested) so their
# source can be extracted by fast-rlm and re-created inside the Pyodide REPL.
# Each function is fully self-contained: imports inside the body, reads env,
# defensive try/except.
# --------------------------------------------------------------------------- #
def search_knowledge(query: str, top_k: int = 20) -> str:
    """Semantic recall over ALL product knowledge (codebases, specs, links,
    knowledge tree, cognee/pgvector vectors).

    Call this FIRST to find relevant context for the user's question, then
    drill into specific files / specs / nodes as needed via the other tools.
    Returns the recalled text (possibly empty if nothing is indexed).
    """
    try:
        import os as _os
        from pyodide.http import open_url as _open_url
        from urllib.parse import urlencode as _urlencode

        _base = _os.environ.get("RLM_API_BASE", "").rstrip("/")
        _pid = _os.environ.get("PRODUCT_ID", "")
        if not _base or not _pid:
            return ""
        _qs = _urlencode({"product_id": _pid, "query": query, "top_k": int(top_k)})
        _resp = _open_url(f"{_base}/api/internal/rlm/knowledge?{_qs}")
        import json as _json
        _data = _json.loads(_resp.getvalue() if hasattr(_resp, "getvalue") else str(_resp))
        return _data.get("text", "") or ""
    except Exception as _e:
        try:
            import os as _os2
            _os2.environ.get("_RLM_TOOL_ERR", "")
        except Exception:
            pass
        return ""


def list_codebase_files(codebase_id: str = "") -> list:
    """Return the file tree (list of relative paths) of the codebase's cloned
    repo. Use this to discover what files exist before reading specific ones
    with ``read_codebase_file``. Defaults to the env CODEBASE_ID when omitted.
    """
    try:
        import os as _os
        from pyodide.http import open_url as _open_url
        from urllib.parse import urlencode as _urlencode

        _base = _os.environ.get("RLM_API_BASE", "").rstrip("/")
        _cb = codebase_id or _os.environ.get("CODEBASE_ID", "")
        if not _base or not _cb:
            return []
        _qs = _urlencode({"codebase_id": _cb})
        _resp = _open_url(f"{_base}/api/internal/rlm/files?{_qs}")
        import json as _json
        _data = _json.loads(_resp.getvalue() if hasattr(_resp, "getvalue") else str(_resp))
        return _data.get("files", []) or []
    except Exception:
        return []


def read_codebase_file(path: str, codebase_id: str = "") -> str:
    """Return the FULL contents of one file in the codebase's repo.

    This is the key to covering 100% of a repository of any size: instead of a
    pre-chunked blob in the prompt, the agent reads files ON DEMAND. Call
    ``list_codebase_files`` first to discover paths, then read the ones
    relevant to the current section. Defaults to the env CODEBASE_ID.
    """
    try:
        import os as _os
        from pyodide.http import open_url as _open_url
        from urllib.parse import urlencode as _urlencode

        _base = _os.environ.get("RLM_API_BASE", "").rstrip("/")
        _cb = codebase_id or _os.environ.get("CODEBASE_ID", "")
        if not _base or not _cb or not path:
            return ""
        _qs = _urlencode({"codebase_id": _cb, "path": path})
        _resp = _open_url(f"{_base}/api/internal/rlm/file?{_qs}")
        import json as _json
        _data = _json.loads(_resp.getvalue() if hasattr(_resp, "getvalue") else str(_resp))
        return _data.get("content", "") or ""
    except Exception:
        return ""


def search_code(pattern: str, codebase_id: str = "") -> list:
    """Grep ``pattern`` (regex, case-insensitive) across the codebase's repo.
    Returns a list of ``{path, line, text}`` matches. Use this to locate where
    a symbol / endpoint / config is defined or used. Defaults to env CODEBASE_ID.
    """
    try:
        import os as _os
        from pyodide.http import open_url as _open_url
        from urllib.parse import urlencode as _urlencode

        _base = _os.environ.get("RLM_API_BASE", "").rstrip("/")
        _cb = codebase_id or _os.environ.get("CODEBASE_ID", "")
        if not _base or not _cb or not pattern:
            return []
        _qs = _urlencode({"codebase_id": _cb, "pattern": pattern})
        _resp = _open_url(f"{_base}/api/internal/rlm/search_code?{_qs}")
        import json as _json
        _data = _json.loads(_resp.getvalue() if hasattr(_resp, "getvalue") else str(_resp))
        return _data.get("matches", []) or []
    except Exception:
        return []


def get_specs() -> list:
    """Return all specs for the product (from env PRODUCT_ID) as
    ``[{id, name, kind, content}]``. Call this to read OpenAPI/AsyncAPI specs."""
    try:
        import os as _os
        from pyodide.http import open_url as _open_url
        from urllib.parse import urlencode as _urlencode

        _base = _os.environ.get("RLM_API_BASE", "").rstrip("/")
        _pid = _os.environ.get("PRODUCT_ID", "")
        if not _base or not _pid:
            return []
        _qs = _urlencode({"product_id": _pid})
        _resp = _open_url(f"{_base}/api/internal/rlm/specs?{_qs}")
        import json as _json
        _data = _json.loads(_resp.getvalue() if hasattr(_resp, "getvalue") else str(_resp))
        return _data.get("specs", []) or []
    except Exception:
        return []


def get_links() -> list:
    """Return all links for the product (from env PRODUCT_ID) as
    ``[{id, name, content}]``."""
    try:
        import os as _os
        from pyodide.http import open_url as _open_url
        from urllib.parse import urlencode as _urlencode

        _base = _os.environ.get("RLM_API_BASE", "").rstrip("/")
        _pid = _os.environ.get("PRODUCT_ID", "")
        if not _base or not _pid:
            return []
        _qs = _urlencode({"product_id": _pid})
        _resp = _open_url(f"{_base}/api/internal/rlm/links?{_qs}")
        import json as _json
        _data = _json.loads(_resp.getvalue() if hasattr(_resp, "getvalue") else str(_resp))
        return _data.get("links", []) or []
    except Exception:
        return []


def get_knowledge_nodes() -> list:
    """Return all knowledge-tree nodes for the product (from env PRODUCT_ID) as
    ``[{id, title, slug, node_type, content_md}]``."""
    try:
        import os as _os
        from pyodide.http import open_url as _open_url
        from urllib.parse import urlencode as _urlencode

        _base = _os.environ.get("RLM_API_BASE", "").rstrip("/")
        _pid = _os.environ.get("PRODUCT_ID", "")
        if not _base or not _pid:
            return []
        _qs = _urlencode({"product_id": _pid})
        _resp = _open_url(f"{_base}/api/internal/rlm/nodes?{_qs}")
        import json as _json
        _data = _json.loads(_resp.getvalue() if hasattr(_resp, "getvalue") else str(_resp))
        return _data.get("nodes", []) or []
    except Exception:
        return []


def get_codebases() -> list:
    """Return all codebases for the product (from env PRODUCT_ID) as
    ``[{id, name, repo_url, generated_docs}]``. Use this to discover which
    codebases exist and read their already-generated docs."""
    try:
        import os as _os
        from pyodide.http import open_url as _open_url
        from urllib.parse import urlencode as _urlencode

        _base = _os.environ.get("RLM_API_BASE", "").rstrip("/")
        _pid = _os.environ.get("PRODUCT_ID", "")
        if not _base or not _pid:
            return []
        _qs = _urlencode({"product_id": _pid})
        _resp = _open_url(f"{_base}/api/internal/rlm/codebases?{_qs}")
        import json as _json
        _data = _json.loads(_resp.getvalue() if hasattr(_resp, "getvalue") else str(_resp))
        return _data.get("codebases", []) or []
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Host-side assembly (these run in the host process, only referencing the
# function objects — they are NOT executed in the host).
# --------------------------------------------------------------------------- #
def build_expert_tools() -> List[Callable]:
    """Tool set for the expert scenario: exhaustive search across ALL product
    knowledge. The agent uses these to pull knowledge slices, files, specs,
    links, and nodes on demand instead of from a stringified prompt blob.
    """
    return [
        search_knowledge,
        get_codebases,
        list_codebase_files,
        read_codebase_file,
        search_code,
        get_specs,
        get_links,
        get_knowledge_nodes,
    ]


def build_docgen_tools() -> List[Callable]:
    """Tool set for the codebase-docgen scenario: the agent reads files ON
    DEMAND to cover 100% of the repo regardless of size. Specs/links/nodes are
    included so the agent can enrich the docs with product-level context.
    """
    return [
        list_codebase_files,
        read_codebase_file,
        search_code,
        get_specs,
        get_links,
        get_knowledge_nodes,
        search_knowledge,
    ]


def resolve_env_variables(
    product_id: str,
    codebase_id: str = None,
) -> dict:
    """Build the ``env_variables`` dict passed to ``fast_rlm.run``.

    Carries the loopback API base + the product/codebase ids so the tools
    (which read ``os.environ`` inside the Pyodide REPL) can call back to the
    Productarium internal endpoints. No secret: the endpoints are
    localhost-only (see ``api/routers/rlm_tools.py``).
    """
    # Default to the backend's own port (PORT env, default 8001). The RLM REPL
    # calls back to 127.0.0.1:<port>, which the loopback guard allows.
    import os

    port = os.environ.get("PORT", "8001")
    api_base = os.environ.get("RLM_API_BASE", f"http://127.0.0.1:{port}")
    env: dict = {
        "RLM_API_BASE": api_base,
        "PRODUCT_ID": product_id or "",
    }
    if codebase_id:
        env["CODEBASE_ID"] = codebase_id
    return env
