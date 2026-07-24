"""AI product summary generator (item 4).

Generates a concise summary of a Product by concatenating its artifacts'
``generated_docs`` and its knowledge nodes' ``content_md`` and asking the
standard local LLM for a summary. The result is stored onto
``ProductORM.summary`` by the caller (the knowledge router).

Decoupling notes (per the Wave 2 plan):
- This module does NOT depend on ``api.expert_agent`` (built in parallel).
- This module does NOT edit ``api.artifact_docgen``; it replicates the minimal
  ``_StandardLLM`` wrapper pattern from there so it stays self-contained.
- Provider/model are resolved from the settings store task ``summary``
  (``api.settings_store.get_model_for_task``) with env fallback, so the admin
  panel can configure the summary model without touching this file.

All LLM/cognee/DB dependencies are imported lazily so this module imports
cleanly even when no live Ollama/Postgres is available.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, Iterable, Optional

from api.prompts import load_prompt_file

logger = logging.getLogger(__name__)

# Cap the concatenated context handed to the LLM so very large products stay
# within a single (non-RLM) prompt. The summary is intentionally concise, so a
# truncated context is acceptable. ~20k chars tokenizes to ~6k tokens, leaving
# headroom for the model response inside an 8192-token context window (the
# previous 60_000 cap overflowed to ~18.5k tokens and raised
# "n_keep >= n_ctx" on the default served model).
SUMMARY_CONTEXT_MAX_CHARS = 20_000

# Inline fallback prompt used only if refs/prompts/product_summary.md is missing.
_SUMMARY_PROMPT_FALLBACK = (
    "Сформируй краткое саммари продукта {product_name} (4–8 предложений) на "
    "основе контекста ниже. Не выдумывай факты. Технические термины на английском.\n"
    "<context>\n{content}\n</context>"
)


# --------------------------------------------------------------------------- #
# Standard (non-RLM) LLM wrapper -- replicated from api.artifact_docgen._StandardLLM
# (kept local so this module never imports/edits artifact_docgen).
# --------------------------------------------------------------------------- #
class _SummaryLLM:
    """Thin non-streaming text generator over the configured local LLM."""

    def __init__(
        self,
        provider: str,
        model: Optional[str],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        import adalflow as adal
        from api.config import get_model_config

        generator_config = get_model_config(provider, model)
        model_client_class = generator_config["model_client"]
        # Thread admin base_url/api_key through to the client so the summary
        # LLM hits the configured endpoint (e.g. corporate AI gateway) rather
        # than a dead env-default LM Studio :1234. Mirrors
        # _StandardLLM/_ExpertLLM: Ollama takes host=; the OpenAIClient takes
        # base_url=/api_key= (both already wire SSL verify via ssl_config).
        client_kwargs: Dict[str, Any] = {}
        if provider == "ollama":
            if base_url:
                client_kwargs["host"] = base_url
        elif provider == "openai_local":
            if base_url:
                client_kwargs["base_url"] = base_url
            if api_key:
                client_kwargs["api_key"] = api_key
        self.model_client = model_client_class(**client_kwargs)
        self.model_kwargs = generator_config["model_kwargs"]
        self.generator = adal.Generator(
            template="{{input_str}}",
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
        )

    async def generate(self, prompt: str) -> str:
        def _call() -> str:
            # adalflow 1.x Generator.call() takes ``prompt_kwargs`` (the dict
            # that fills the ``{{input_str}}`` template placeholder), NOT a
            # bare ``input_str=`` kwarg -- passing that raises TypeError.
            result = self.generator(prompt_kwargs={"input_str": prompt})
            for attr in ("response", "answer", "raw_response", "output"):
                val = getattr(result, attr, None)
                if val:
                    return str(val)
            return str(result)

        return await asyncio.to_thread(_call)


def _safe_build_summary_llm(
    provider: str,
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[_SummaryLLM]:
    try:
        return _SummaryLLM(provider, model, base_url=base_url, api_key=api_key)
    except Exception as e:  # pragma: no cover - depends on live config/Ollama
        logger.warning(
            "Could not initialise summary LLM (%s/%s): %s. "
            "Falling back to a deterministic stub summary.",
            provider, model, e,
        )
        return None


def _clean_text(text: Optional[str]) -> str:
    """Strip surrounding whitespace, a wrapping ```markdown fence, and any inline
    line-number prefixes the LLM emitted inside code blocks."""
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
        if t.endswith("```"):
            t = t[:-3]
    try:
        from api.artifact_docgen import _strip_inline_line_numbers
        t = _strip_inline_line_numbers(t)
    except Exception:  # pragma: no cover - import-safe
        pass
    return t.strip()


def _cap(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (обрезано для контекста LLM)\n"


# --------------------------------------------------------------------------- #
# Context collection + prompt building
# --------------------------------------------------------------------------- #
def _collect_summary_content(artifacts: Iterable[Any], nodes: Iterable[Any]) -> str:
    """Concatenate artifact generated_docs + knowledge node content_md."""
    parts = []
    for a in artifacts or []:
        docs = getattr(a, "generated_docs", None) or ""
        if docs and docs.strip():
            name = getattr(a, "name", None) or getattr(a, "id", "artifact")
            parts.append(f"## Артефакт: {name}\n\n{docs.strip()}")
    for n in nodes or []:
        md = getattr(n, "content_md", None) or ""
        if md and md.strip():
            title = getattr(n, "title", None) or getattr(n, "id", "node")
            parts.append(f"## Страница базы знаний: {title}\n\n{md.strip()}")
    if not parts:
        return ""
    return _cap("\n\n".join(parts), SUMMARY_CONTEXT_MAX_CHARS)


def _build_summary_prompt(product_name: str, content: str) -> str:
    template = load_prompt_file("product_summary.md", _SUMMARY_PROMPT_FALLBACK)
    out = template
    out = out.replace("{product_name}", product_name)
    out = out.replace("{content}", content)
    return out


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
async def generate_product_summary(
    product: Any,
    artifacts: Iterable[Any],
    nodes: Iterable[Any],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Generate an AI summary over the product's artifacts + knowledge nodes.

    Returns the cleaned summary text (possibly empty on LLM failure or when the
    product has no content to summarize). NEVER raises: callers store/return the
    result directly.
    """
    product_name = getattr(product, "name", "") or "product"
    content = _collect_summary_content(artifacts, nodes)
    if not content.strip():
        logger.info("Product %r has no content to summarize.", product_name)
        return ""

    # Resolve provider/model/base_url/api_key from the settings store 'summary'
    # task, falling back to env-driven defaults. Non-fatal if the store/DB is
    # unavailable. base_url/api_key are read here (NOT in _SummaryLLM) so the
    # summary LLM reaches the configured corporate gateway instead of the dead
    # env-default LM Studio :1234 -- previously _SummaryLLM built its client
    # with NO base_url/api_key, so the summary silently hit a non-existent
    # endpoint and returned an empty/garbage result ("выдаёт запрос вместо
    # результата").
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    if provider is None or model is None:
        try:
            from api.settings_store import get_model_for_task
            cfg = get_model_for_task("summary")
            provider = provider or cfg.get("provider") or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local")
            model = model or cfg.get("model")
            base_url = cfg.get("base_url")
            api_key = cfg.get("api_key")
        except Exception as e:  # pragma: no cover - depends on live DB
            logger.debug("settings_store summary task lookup failed: %s", e)
            provider = provider or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local")

    llm = _safe_build_summary_llm(
        provider or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local"),
        model,
        base_url=base_url,
        api_key=api_key,
    )
    if llm is None:
        return ""
    try:
        return _clean_text(await llm.generate(_build_summary_prompt(product_name, content)))
    except Exception as e:  # pragma: no cover - depends on live Ollama
        logger.warning("Summary LLM generation failed: %s", e)
        return ""


__all__ = ["generate_product_summary"]
