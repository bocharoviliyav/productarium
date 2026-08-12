"""Links / documentation / guides artifact documentation (item 2).

Split out of the former ``api/artifact_docgen.py`` (Step 4). These artifact
types are light-weight: links render a Markdown index (no heavy gen);
documentation/guides pass their content through with optional LLM enrichment.
All paths index into cognee and persist ``generated_docs`` + ``pages``.

Shared LLM/persistence helpers live in ``api.docgen._common``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from api.utils import setup_logging
from api.utils.llm_helpers import (  # noqa: E402
    safe_replace as _safe_replace,
)
from api.prompts import load_prompt_file
from api.docgen._common import (
    _with_verification_guard,
    _resolve_docgen_model,
    _llm_or_none,
    _persist_artifact,
    _cognee_dataset,
    _index_in_background,
    _product_name,
)

setup_logging()
logger = logging.getLogger(__name__)


def _render_links_index(content: str, artifact: Any) -> str:
    """Render a links artifact as a Markdown index page.

    ``artifact.content`` may be JSON (a list of ``{url, title?, description?}``
    objects, or ``{"links": [...]}``, or a single link object) or free-form
    Markdown. A clean bullet index is rendered when JSON parses; otherwise the
    content is passed through verbatim. No heavy generation.
    """
    name = getattr(artifact, "name", None) or "Links"
    text = (content or "").strip()
    items: List[Dict[str, Any]] = []
    if text:
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                items = [i for i in loaded if isinstance(i, dict)]
            elif isinstance(loaded, dict):
                links_field = loaded.get("links")
                if isinstance(links_field, list):
                    items = [i for i in links_field if isinstance(i, dict)]
                else:
                    items = [loaded]
        except Exception:
            items = []

    md: List[str] = [f"# Links: {name}"]
    if items:
        md.append("")
        for it in items:
            url = (it.get("url") or it.get("link") or "").strip()
            title = (it.get("title") or it.get("name") or url or "link").strip()
            desc = (it.get("description") or it.get("desc") or "").strip()
            if url:
                md.append(f"- [{title}]({url})" + (f" — {desc}" if desc else ""))
            elif desc:
                md.append(f"- {title} — {desc}")
    elif text:
        # Not JSON -> treat as pre-formatted Markdown.
        md.append("")
        md.append(text)
    else:
        md.append("")
        md.append("_(Ссылки не предоставлены.)_")
    return "\n".join(md)


async def generate_links_docs(
    artifact: Any, product: Any, provider: str = None,
    model: Optional[str] = None, language: str = "ru",
) -> str:
    """Generate a links index page. No heavy LLM gen; content holds URL+description."""
    content = (getattr(artifact, "content", "") or "").strip()
    docs = _render_links_index(content, artifact)
    pages = {
        "page_links": {
            "id": "page_links",
            "title": getattr(artifact, "name", None) or "Links",
            "content": docs,
            "filePaths": [],
            "importance": "medium",
            "relatedPages": [],
        }
    }
    _persist_artifact(artifact, docs, pages)
    _index_in_background(content, _cognee_dataset(product))
    return docs


async def generate_documentation_docs(
    artifact: Any, product: Any, provider: str = None,
    model: Optional[str] = None, language: str = "ru",
) -> str:
    """Documentation artifact (non-testcase): manual/generated MD passthrough + optional LLM enrichment.

    The artifact's ``content`` is the source of truth. A best-effort LLM
    enrichment (``refs/prompts/documentation_doc.md``) polishes the markdown; on
    any LLM failure the original content is returned unchanged.
    """
    content = (getattr(artifact, "content", "") or "").strip()
    if not content:
        raise ValueError("Documentation artifact has empty content.")

    # Resolve admin docgen config (models.docgen.*) for the LLM enrichment.
    r_provider, r_model, r_base_url, r_api_key = _resolve_docgen_model(model)
    provider = provider or r_provider
    model = model or r_model

    from api.utils import get_model_context_window, clamp_text_by_tokens
    ctx_win = get_model_context_window(provider=provider, base_url=r_base_url, model_name=model, api_key=r_api_key, task="docgen")
    content_token_limit = max(1024, ctx_win - 2048)
    clamped_content = clamp_text_by_tokens(content, content_token_limit)

    enriched = ""
    template = load_prompt_file("documentation_doc.md", "")
    if template:
        prompt = _with_verification_guard(_safe_replace(
            template,
            {
                "artifact_name": getattr(artifact, "name", None) or "Documentation",
                "content": clamped_content,
            },
        ))
        enriched = await _llm_or_none(
            prompt, provider, model, base_url=r_base_url, api_key=r_api_key
        )
    docs = enriched or content

    pages = {
        "page_documentation": {
            "id": "page_documentation",
            "title": getattr(artifact, "name", None) or "Documentation",
            "content": docs,
            "filePaths": [],
            "importance": "high",
            "relatedPages": [],
        }
    }
    _persist_artifact(artifact, docs, pages)
    _index_in_background(content, _cognee_dataset(product))
    return docs


async def generate_guides_docs(
    artifact: Any, product: Any, provider: str = None,
    model: Optional[str] = None, language: str = "ru",
) -> str:
    """Guides artifact: manual or generated Markdown passthrough."""
    content = (getattr(artifact, "content", "") or "").strip()
    if not content:
        raise ValueError("Guides artifact has empty content.")
    name = getattr(artifact, "name", None) or "Guides"
    docs = f"# {name}\n\n{content}"
    pages = {
        "page_guides": {
            "id": "page_guides",
            "title": name,
            "content": docs,
            "filePaths": [],
            "importance": "medium",
            "relatedPages": [],
        }
    }
    _persist_artifact(artifact, docs, pages)
    _index_in_background(content, _cognee_dataset(product))
    return docs
