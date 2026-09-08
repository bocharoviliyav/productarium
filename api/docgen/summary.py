"""AI product summary generator (item 4).

Generates a concise summary of a Product by concatenating its artifacts'
``generated_docs`` and its knowledge nodes' ``content_md`` and asking the
standard local LLM for a summary. The result is stored onto
``ProductORM.summary`` by the caller (the knowledge router).

Decoupling notes (per the Wave 2 plan):
- This module does NOT depend on ``api.expert`` (built in parallel).
- The LLM is ``api.llm.GenerateLLM`` (langchain ``ChatOpenAI``) shared with
  the rest of the stack.
- Provider/model are resolved from the settings store task ``summary``
  (``api.config.settings.get_model_for_task``) with env fallback, so the admin
  panel can configure the summary model without touching this file.

All LLM/DB dependencies are imported lazily so this module imports
cleanly even when no live LLM/Postgres is available.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional

from api.prompts import load_prompt_file
from api.utils.llm_helpers import (  # noqa: E402
    cap as _cap,
    strip_inline_line_numbers as _strip_inline_line_numbers,
)
from api.docgen._common import _safe_aclose, _with_verification_guard
from api.docgen.verification import mask_secrets

logger = logging.getLogger(__name__)

# Cap the concatenated context handed to the LLM so very large products stay
# within a single prompt. The summary is intentionally concise, so a truncated
# context is acceptable. ~20k chars tokenizes to ~6k tokens, leaving headroom
# for the model response inside an 8192-token context window (the previous
# 60_000 cap overflowed to ~18.5k tokens and raised "n_keep >= n_ctx" on the
# default served model).
SUMMARY_CONTEXT_MAX_CHARS = 20_000

# Inline fallback prompt used only if refs/prompts/product_summary.md is missing.
_SUMMARY_PROMPT_FALLBACK = (
    "Сформируй краткое саммари продукта {product_name} (4–8 предложений) на "
    "основе контекста ниже. Не выдумывай факты. Технические термины на английском.\n"
    "<context>\n{content}\n</context>"
)


# --------------------------------------------------------------------------- #
# Standard LLM -- api.llm.GenerateLLM (langchain ChatOpenAI).
# --------------------------------------------------------------------------- #
class _SummaryLLM:
    """Thin non-streaming text generator over the configured local LLM."""

    def __init__(
        self,
        model: Optional[str],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        from api.llm import GenerateLLM

        # GenerateLLM threads admin base_url/api_key through to the
        # OpenAI-compatible endpoint (corporate AI gateway, LM Studio, ...)
        # and returns "" on any failure, so a model error never leaks the
        # prompt into the stored product summary.
        self._llm = GenerateLLM(model=model, base_url=base_url, api_key=api_key)

    async def generate(self, prompt: str) -> str:
        return await self._llm.generate(prompt)

    async def aclose(self) -> None:
        """Close the wrapped generator's httpx client."""
        await _safe_aclose(self._llm)


def _safe_build_summary_llm(
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[_SummaryLLM]:
    try:
        return _SummaryLLM(model, base_url=base_url, api_key=api_key)
    except Exception as e:  # pragma: no cover - depends on live config/LLM
        logger.warning(
            "Could not initialise summary LLM (%s): %s. "
            "Falling back to a deterministic stub summary.",
            model, e,
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
    t = _strip_inline_line_numbers(t)
    return t.strip()


# --------------------------------------------------------------------------- #
# Context collection + prompt building
# --------------------------------------------------------------------------- #
def _collect_summary_content(
    codebases: Iterable[Any],
    specs: Iterable[Any],
    nodes: Iterable[Any],
    max_tokens: int = 6000,
) -> str:
    """Concatenate codebase generated_docs + spec content + knowledge node content_md."""
    parts = []
    for c in codebases or []:
        docs = getattr(c, "generated_docs", None) or ""
        if docs and docs.strip():
            name = getattr(c, "name", None) or getattr(c, "id", "codebase")
            parts.append(f"## Codebase: {name}\n\n{docs.strip()}")
    for s in specs or []:
        content = getattr(s, "content", None) or ""
        if content and content.strip():
            name = getattr(s, "name", None) or getattr(s, "id", "spec")
            kind = getattr(s, "kind", None) or "spec"
            parts.append(f"## Спецификация ({kind}): {name}\n\n{content.strip()}")
    for n in nodes or []:
        md = getattr(n, "content_md", None) or ""
        if md and md.strip():
            title = getattr(n, "title", None) or getattr(n, "id", "node")
            parts.append(f"## Страница базы знаний: {title}\n\n{md.strip()}")
    if not parts:
        return ""

    full_text = "\n\n".join(parts)
    return _cap(full_text, SUMMARY_CONTEXT_MAX_CHARS)


def _build_summary_prompt(product_name: str, content: str) -> str:
    template = load_prompt_file("product_summary.md", _SUMMARY_PROMPT_FALLBACK)
    out = template
    out = out.replace("{product_name}", product_name)
    out = out.replace("{content}", content)
    # Same deterministic verification guard as the docgen flows (Wave D):
    # no secrets, no invented facts, source-bound claims.
    return _with_verification_guard(out)


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
async def generate_product_summary(
    product: Any,
    codebases: Iterable[Any],
    specs: Iterable[Any],
    nodes: Iterable[Any],
    *,
    model: Optional[str] = None,
) -> str:
    """Generate an AI summary over the product's codebases + specs + knowledge nodes.

    Returns the cleaned summary text (possibly empty on LLM failure or when the
    product has no content to summarize). NEVER raises: callers store/return the
    result directly.
    """
    product_name = getattr(product, "name", "") or "product"

    # Resolve model/base_url/api_key from the settings store 'summary' task,
    # falling back to env-driven defaults.
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    if model is None:
        try:
            from api.config.settings import get_model_for_task
            cfg = get_model_for_task("summary")
            model = cfg.get("model")
            base_url = cfg.get("base_url")
            api_key = cfg.get("api_key")
        except Exception as e:  # pragma: no cover - depends on live DB
            logger.debug("settings_store summary task lookup failed: %s", e)

    try:
        from api.utils import get_model_context_window
        ctx_win = get_model_context_window(base_url=base_url, model_name=model, api_key=api_key, task="summary")
    except Exception:
        ctx_win = 8192

    max_summary_tokens = max(1024, ctx_win - 2048)
    content = _collect_summary_content(codebases, specs, nodes, max_tokens=max_summary_tokens)
    if not content.strip():
        logger.info("Product %r has no content to summarize.", product_name)
        return ""

    llm = _safe_build_summary_llm(
        model,
        base_url=base_url,
        api_key=api_key,
    )
    if llm is None:
        return ""
    try:
        raw = _clean_text(await llm.generate(_build_summary_prompt(product_name, content)))
    except Exception as e:  # pragma: no cover - depends on live LLM
        logger.warning("Summary LLM generation failed: %s", e)
        return ""
    finally:
        await _safe_aclose(llm)
    # Deterministic secret guard: the summary lands in ProductORM.summary and
    # the products API — the same masked-text contract as the docgen flows.
    masked, findings = mask_secrets(raw)
    if findings:
        logger.warning(
            "Product summary guard: masked %d secret-like value(s) (%s)",
            len(findings), ", ".join(sorted(set(findings))),
        )
    return masked


__all__ = ["generate_product_summary"]
