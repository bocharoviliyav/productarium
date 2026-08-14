"""Shared helpers for the docgen pipeline (event loop, LLM wrapper, persistence).

Moved out of the former ``api/artifact_docgen.py`` during the Step 4 split so the
codebase / spec / simple generators can share one LLM wrapper, one cognee
indexing handoff, and one set of prompt/naming helpers without cross-importing
each other.

The three LLM wrapper classes (``_StandardLLM`` / ``_ExpertLLM`` / ``_SummaryLLM``)
are NOT identical (retry vs streaming vs simple) and stay in their domain
packages; only ``_StandardLLM`` (the docgen retry wrapper) lives here. Likewise
``_clean_llm_text`` differs between modules (expert strips ``<r>`` blocks) so
the docgen variant lives here and expert keeps its own.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

from api.utils import setup_logging
from api.utils.llm_helpers import (  # noqa: E402
    safe_replace as _safe_replace,
    cap as _cap,
    strip_inline_line_numbers as _strip_inline_line_numbers,
    strip_number_prefixes_from_block as _strip_number_prefixes_from_block,
    LINE_NUM_PREFIX_RE as _LINE_NUM_PREFIX_RE,
    LINE_NUM_ONLY_RE as _LINE_NUM_ONLY_RE,
)

setup_logging()
logger = logging.getLogger(__name__)


# Long-lived main FastAPI event loop, captured at startup so the docgen worker
# threads (which run their own short-lived loops) can hand off fire-and-forget
# cognee indexing via ``asyncio.run_coroutine_threadsafe``. This lets the
# long-running cognify survive the worker loop teardown instead of being
# cancelled when ``_run_docgen_job`` finishes. Set by ``set_main_event_loop``
# from ``api.api.startup_event``.
_main_event_loop: Optional[asyncio.AbstractEventLoop] = None


def set_main_event_loop(loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """Record the long-lived main event loop for cross-thread task handoff.

    Called once from ``api.api.startup_event``. Docgen worker threads then use
    ``get_main_event_loop`` to schedule cognee indexing so the cognify coroutine
    is NOT cancelled when the worker's own loop closes.
    """
    global _main_event_loop
    _main_event_loop = loop


def get_main_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Return the captured main loop, or None if startup has not run yet."""
    return _main_event_loop


def _with_verification_guard(prompt: str) -> str:
    """Append the unified verification guard to a built LLM prompt.

    The guard (grounding/citation/no-line-numbers/unverified-flag rules) is the
    single source of truth in ``refs/prompts/_verification_guard.md``. It is read
    fresh from ``api.prompts`` at call time so a hot-reload via the admin panel
    takes effect without a process restart. Returns the prompt unchanged if the
    guard is empty/unavailable.
    """
    if not prompt:
        return prompt
    try:
        from api.prompts import VERIFICATION_GUARD as _guard
    except Exception:  # pragma: no cover - import-safe
        _guard = ""
    if _guard:
        return prompt + "\n\n" + _guard
    return prompt


# Regex constants + strip_inline_line_numbers / strip_number_prefixes_from_block /
# _safe_replace / _cap now live in api.utils.llm_helpers (dedup). _clean_llm_text
# stays here (docgen variant) and calls the shared strip helper.
def _clean_llm_text(text: Optional[str]) -> str:
    """Strip surrounding whitespace, a single wrapping ```markdown fence, and
    any inline line-number prefixes the LLM emitted inside code blocks."""
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
        if t.endswith("```"):
            t = t[:-3]
    t = _strip_inline_line_numbers(t)
    return t.strip()


def _repo_name_from_url(repo_url: str) -> str:
    name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    return name or repo_url


def _product_name(product: Any, artifact: Any) -> str:
    if product is not None and getattr(product, "name", None):
        return product.name
    if getattr(artifact, "repo_url", None):
        return _repo_name_from_url(artifact.repo_url)
    return getattr(artifact, "name", "") or "product"


# ---------------------------------------------------------------------------
# Standard (non-RLM) LLM wrapper -- adalflow Generator over local Ollama /
# OpenAI-local. Built lazily from api.config.get_model_config (no cloud keys).
# ---------------------------------------------------------------------------
class _StandardLLM:
    """Thin non-streaming text generator over the configured local LLM.

    Honors admin-configured ``base_url`` / ``api_key`` from
    ``api.config.settings.get_model_for_task("docgen")`` so docgen reaches the
    corporate AI gateway instead of falling back to a dead local env default.
    """

    def __init__(
        self,
        model: Optional[str],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        import adalflow as adal
        from api.config import get_model_config

        if not model:
            model = "qwen/qwen3.6-27b"
        generator_config = get_model_config(model)
        model_client_class = generator_config["model_client"]
        # Thread admin base_url/api_key through to the OpenAI-compatible client
        # so docgen hits the configured endpoint (corporate gateway, LM Studio,
        # Ollama :11434, ...) rather than the env default. Every supported
        # server exposes the OpenAI-compatible /v1 API, so OpenAIClient covers
        # all cases (Ollama included). SSL verify is wired via ssl_config.
        client_kwargs: Dict[str, Any] = {}
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
        async def _call_with_retry() -> str:
            def _single_call():
                try:
                    result = self.generator(prompt_kwargs={"input_str": prompt})
                    if getattr(result, "error", None):
                        return "", Exception(str(result.error))
                    for attr in ("data", "response", "answer", "raw_response", "output"):
                        val = getattr(result, attr, None)
                        if val:
                            return str(val), None
                    return "", None
                except Exception as ex:
                    return "", ex

            max_retries = 3
            for attempt in range(max_retries):
                res_str, exc = await asyncio.to_thread(_single_call)
                if res_str:
                    return res_str
                if exc:
                    emsg = str(exc).lower()
                    if ("429" in emsg or "rate limit" in emsg or "too many requests" in emsg) and attempt < max_retries - 1:
                        backoff = (attempt + 1) * 2.5
                        logger.warning("Standard LLM hit rate limit (attempt %d/%d). Sleeping %.1fs: %s", attempt + 1, max_retries, backoff, exc)
                        await asyncio.sleep(backoff)
                    else:
                        if exc:
                            logger.warning("Standard LLM returned an error: %s", exc)
                        break
            return ""

        return await _call_with_retry()


def _resolve_docgen_model(
    model: Optional[str],
) -> Tuple[str, Optional[str], Optional[str]]:
    """Resolve (model, base_url, api_key) for the docgen task.

    Reads admin-configured ``models.docgen.*`` from the Config Abstraction Layer
    (with env fallbacks) so docgen hits the corporate AI gateway when configured.
    """
    try:
        from api.config.abstraction import get_task_config

        cfg = get_task_config("docgen") or {}
        resolved_model = model or cfg.get("model") or "qwen/qwen3.6-27b"
        return resolved_model, cfg.get("base_url"), cfg.get("api_key")
    except Exception as e:  # pragma: no cover - settings store is import-safe
        logger.debug("get_task_config(docgen) failed; using defaults: %s", e)
        return (
            model or "qwen/qwen3.6-27b",
            None,
            None,
        )


def _safe_build_llm(
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[_StandardLLM]:
    try:
        return _StandardLLM(model, base_url=base_url, api_key=api_key)
    except Exception as e:  # pragma: no cover - depends on live config/Ollama
        logger.warning(
            "Could not initialise standard LLM (%s): %s. "
            "Falling back to RLM/skeleton where possible.", model, e,
        )
        return None


async def _llm_or_none(
    prompt: str,
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Run the standard LLM on ``prompt``; return cleaned text or "" on failure."""
    if not prompt:
        return ""
    llm = _safe_build_llm(model, base_url=base_url, api_key=api_key)
    if llm is None:
        return ""
    try:
        return _clean_llm_text(await llm.generate(prompt))
    except Exception as e:  # pragma: no cover - depends on live Ollama
        logger.warning("Standard LLM generation failed: %s", e)
        return ""


def _make_repair_llm(
    model: Optional[str],
    existing: Optional[_StandardLLM] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional["object"]:
    """Build an async ``(prompt) -> str`` callable for the mermaid repair loop.

    Reuses an already-built ``_StandardLLM`` when available (so the codebase
    path doesn't construct a second client); otherwise builds one from the same
    model/base_url/api_key. Returns None if no LLM could be built
    (repairs are then skipped and broken diagrams are surfaced with a marker).
    """
    if existing is not None:
        llm = existing
    else:
        llm = _safe_build_llm(model, base_url=base_url, api_key=api_key)
    if llm is None:
        return None

    async def _call(prompt: str) -> str:
        try:
            return await llm.generate(prompt)
        except Exception as e:  # pragma: no cover - depends on live Ollama
            logger.warning("Mermaid repair LLM call failed: %s", e)
            return ""

    return _call


# ---------------------------------------------------------------------------
# Persistence + cognee indexing helpers
# ---------------------------------------------------------------------------
def _persist_artifact(artifact: Any, markdown: str, pages: Dict[str, Any]) -> None:
    """Write generated_docs + pages onto the artifact (ORM or Pydantic)."""
    try:
        artifact.generated_docs = markdown
        artifact.pages = pages
    except Exception as e:  # pragma: no cover - defensive over attribute setting
        logger.warning("Could not write generated_docs/pages onto artifact: %s", e)


def _cognee_dataset(product: Any) -> str:
    """Product-scoped cognee dataset name (item 1 cognee-first): ``prod_{product_id}``.

    All generated artifact/page content is indexed into a single per-product
    cognee dataset so the expert agent and Ask can recall across every
    artifact of the product. Falls back to ``unknown`` if the product has no id.
    """
    pid = getattr(product, "id", None) or getattr(product, "product_id", None) or "unknown"
    return f"prod_{pid}"


def _index_in_background(content_or_path: str, dataset_name: str) -> None:
    """Fire-and-forget cognee indexing. Failures are logged, never fatal.

    The indexing coroutine is handed off to the long-lived MAIN FastAPI event
    loop (captured at startup) via ``asyncio.run_coroutine_threadsafe`` so it
    survives the docgen worker thread's own short-lived loop teardown. This is
    essential because cognee's ``cognify`` can legitimately run 20-30 min — if
    it were scheduled on the worker loop, closing that loop when the docgen job
    finishes would CANCEL the still-running cognify and the graph would never
    finish building. By moving it to the main loop, the job can return
    immediately (display is decoupled from the knowledge graph) while indexing
    continues in the background.
    """

    async def _run() -> None:
        try:
            from api.cognee import add_and_index_document  # lazy: cognee optional
            await add_and_index_document(content_or_path, dataset_name=dataset_name)
        except Exception as e:  # pragma: no cover - depends on live cognee/DB
            logger.warning("Cognee indexing failed for %r: %s", dataset_name, e)

    main_loop = get_main_event_loop()
    if main_loop is not None and main_loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(_run(), main_loop)
            logger.info(
                "Scheduled cognee indexing for %r onto the main event loop.",
                dataset_name,
            )
            return
        except RuntimeError as e:  # loop closed between the check and the call
            logger.warning(
                "Could not schedule cognee indexing on the main loop for %r (%s); "
                "falling back to a local task.",
                dataset_name, e,
            )

    # Fallback: no main loop captured (startup not run / tests) — schedule on the
    # current loop so the old non-worker callers (websocket wiki, inline edits)
    # still work. The worker-thread drain in ``_run_docgen_job`` is best-effort
    # and non-fatal, so a cancellation here no longer marks the job as failed.
    try:
        asyncio.create_task(_run())
    except RuntimeError:
        # No running event loop -- best-effort skip.
        logger.warning(
            "No running event loop; skipping background cognee indexing for %r.",
            dataset_name,
        )
