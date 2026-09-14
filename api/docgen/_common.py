"""Shared helpers for the docgen pipeline (event loop, LLM, persistence).

Moved out of the former ``api/artifact_docgen.py`` during the Step 4 split so the
codebase / spec / simple generators can share one LLM path, one memory-backend
indexing handoff, and one set of prompt/naming helpers without cross-importing
each other.

The LLM itself is ``api.llm.GenerateLLM`` (langchain ``ChatOpenAI`` with retry);
the domain wrappers keep their own cleaning/prompt logic. Likewise
``_clean_llm_text`` differs between modules (expert strips ``<r>`` blocks) so
the docgen variant lives here and expert keeps its own.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, Dict, Optional, Tuple

from api.utils import setup_logging
from api.utils.llm_helpers import (  # noqa: E402
    safe_replace as _safe_replace,
    cap as _cap,
    aclose_llm as _aclose_llm,
    strip_inline_line_numbers as _strip_inline_line_numbers,
    strip_llm_preamble as _strip_llm_preamble,
    strip_number_prefixes_from_block as _strip_number_prefixes_from_block,
    LINE_NUM_PREFIX_RE as _LINE_NUM_PREFIX_RE,
    LINE_NUM_ONLY_RE as _LINE_NUM_ONLY_RE,
)

setup_logging()
logger = logging.getLogger(__name__)


# Long-lived main FastAPI event loop, captured at startup so the docgen worker
# threads (which run their own short-lived loops) can hand off fire-and-forget
# memory indexing via ``asyncio.run_coroutine_threadsafe``. This lets the
# long-running indexing survive the worker loop teardown instead of being
# cancelled when ``_run_docgen_job`` finishes. Set by ``set_main_event_loop``
# from ``api.api.startup_event``.
_main_event_loop: Optional[asyncio.AbstractEventLoop] = None


def set_main_event_loop(loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """Record the long-lived main event loop for cross-thread task handoff.

    Called once from ``api.api.startup_event``. Docgen worker threads then use
    ``get_main_event_loop`` to schedule memory indexing so the coroutine
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
    t = _strip_llm_preamble(t)
    return t.strip()


def _repo_name_from_url(repo_url: str) -> str:
    name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    return name or repo_url


class JobCancelledError(RuntimeError):
    """Cooperative cancellation — raised at pipeline checkpoints once the
    job's cancel flag is set (the artifact keeps its pre-run version)."""


def _check_cancel(should_cancel: Optional[Callable[[], bool]]) -> None:
    """Raise :class:`JobCancelledError` when cancellation was requested."""
    if should_cancel is not None and should_cancel():
        raise JobCancelledError("Generation cancelled by user")


def emit_progress(progress: Optional[Any], **fields: Any) -> None:
    """Invoke a docgen progress callback; a no-op when absent, never raises.

    ``progress`` is the optional ``Callable[..., None]`` threaded from
    ``api.docgen.jobs.report_progress`` through the generate_* entry points
    (None for legacy/test callers). Accepted fields: ``phase``,
    ``sections_total``, ``sections_done``, ``current_section``,
    ``section_done`` (+ ``section_seconds``). Broken progress plumbing must
    never fail a generation run.
    """
    if progress is None:
        return
    try:
        progress(**fields)
    except Exception:
        logger.debug("docgen progress callback failed", exc_info=True)


def _product_name(product: Any, artifact: Any) -> str:
    if product is not None and getattr(product, "name", None):
        return product.name
    if getattr(artifact, "repo_url", None):
        return _repo_name_from_url(artifact.repo_url)
    return getattr(artifact, "name", "") or "product"


# ---------------------------------------------------------------------------
# Standard LLM -- api.llm.GenerateLLM (langchain ChatOpenAI) over the
# OpenAI-compatible local server. Retry/backoff lives in api.llm.generate.
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
        from api.llm import GenerateLLM

        if not model:
            model = "qwen/qwen3.6-27b"
        self._llm = GenerateLLM(model=model, base_url=base_url, api_key=api_key)

    async def generate(self, prompt: str) -> str:
        return await self._llm.generate(prompt)

    async def aclose(self) -> None:
        """Close the wrapped generator's httpx client (docgen pool hygiene)."""
        await _safe_aclose(self._llm)


async def _safe_aclose(llm: Any) -> None:
    """Close the httpx client behind an LLM wrapper (best-effort, never raises).

    Works with anything exposing ``aclose()`` (``GenerateLLM``, the docgen
    ``_StandardLLM``/``_SummaryLLM`` wrappers); silently ignores objects
    without one (fakes in tests, factory changes).
    """
    if llm is None:
        return
    aclose = getattr(llm, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        logger.debug("could not close an LLM httpx client", exc_info=True)


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
    except Exception as e:  # pragma: no cover - depends on live config/LLM
        logger.warning(
            "Could not initialise standard LLM (%s): %s. "
            "Falling back to skeleton where possible.", model, e,
        )
        return None


async def _llm_or_none(
    prompt: str,
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Run the standard LLM on ``prompt``; return cleaned text or "" on failure.

    Builds, uses and CLOSES its own client (one per call): the caller never
    has to manage the lifecycle.
    """
    if not prompt:
        return ""
    llm = _safe_build_llm(model, base_url=base_url, api_key=api_key)
    if llm is None:
        return ""
    try:
        return _clean_llm_text(await llm.generate(prompt))
    except Exception as e:  # pragma: no cover - depends on live LLM
        logger.warning("Standard LLM generation failed: %s", e)
        return ""
    finally:
        # P1-14: release httpx pools. Duck-typed close — the factory is a
        # patch point and may return objects without ``aclose``.
        await _safe_aclose(llm)


def _make_repair_llm(
    model: Optional[str],
    existing: Optional[_StandardLLM] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional["object"]:
    """Build an async ``(prompt) -> str`` callable for the mermaid repair loop.

    Reuses an already-built ``_StandardLLM`` when available (so the codebase
    path doesn't construct a second client — its owner closes it); otherwise
    builds one from the same model/base_url/api_key and marks OWNERSHIP via
    the ``_owned_llm`` attribute: the caller MUST then close it through
    ``_close_owned_llm`` (spec flow). Returns None if no LLM could be built
    (repairs are then skipped and broken diagrams are surfaced with a marker).
    """
    def _build() -> Optional[_StandardLLM]:
        return existing if existing is not None else _safe_build_llm(
            model, base_url=base_url, api_key=api_key
        )

    owns_llm = existing is None
    llm = _build()
    if llm is None:
        return None

    async def _call(prompt: str) -> str:
        # P1-14: when this closure owns the LLM (no ``existing`` passed), build
        # per call and close it in ``finally`` so no httpx pool leaks. When an
        # ``existing`` LLM is reused, its owner is responsible for closing.
        owned = existing is None
        llm = _build()
        if llm is None:
            return ""
        try:
            return await llm.generate(prompt)
        except Exception as e:  # pragma: no cover - depends on live LLM
            logger.warning("Mermaid repair LLM call failed: %s", e)
            return ""
        finally:
            if owned:
                await _aclose_llm(llm)  # P1-14 (duck-typed; see _llm_or_none)

    _call._owned_llm = llm if owns_llm else None  # type: ignore[attr-defined]
    return _call


async def _close_owned_llm(repair_llm: Any) -> None:
    """Close the LLM built by ``_make_repair_llm`` when it owns one."""
    owned = getattr(repair_llm, "_owned_llm", None)
    if owned is not None:
        await _safe_aclose(owned)


# ---------------------------------------------------------------------------
# Persistence + memory indexing helpers
# ---------------------------------------------------------------------------
def _persist_artifact(artifact: Any, markdown: str, pages: Dict[str, Any]) -> None:
    """Write generated_docs + pages onto the artifact (ORM or Pydantic)."""
    try:
        artifact.generated_docs = markdown
        artifact.pages = pages
    except Exception as e:  # pragma: no cover - defensive over attribute setting
        logger.warning("Could not write generated_docs/pages onto artifact: %s", e)


def _checkpoint_partial_docs(
    artifact_id: Any,
    model: Any,
    markdown: str,
    pages: Dict[str, Any],
) -> bool:
    """Persist a PARTIAL doc set onto the artifact row mid-run (item 2.3a).

    Short-lived session, immediate commit: a crash later in the run (worker
    kill, LLM outage at section 6 of 7) keeps every already-verified section
    durable, and the rerun picks those sections up via diff regeneration
    (``plan_regeneration`` reuse) instead of paying for them again — the
    checkpoint stores exactly the provenance/fingerprint payload the final
    persist would. On success the worker's own session later overwrites the
    row with the complete doc set; on failure its rollback leaves the
    checkpoints intact (that is the point).

    Best-effort by contract: any error is logged and swallowed — a checkpoint
    must never fail a generation run.
    """
    if not artifact_id or model is None:
        return False
    from api.db import SessionLocal

    session = SessionLocal()
    try:
        row = session.get(model, artifact_id)
        if row is None:
            logger.debug(
                "checkpoint target %s %s not found; skipping", model, artifact_id,
            )
            return False
        row.generated_docs = markdown
        row.pages = pages
        session.commit()
        return True
    except Exception as e:  # pragma: no cover - checkpoint must never break gen
        try:
            session.rollback()
        except Exception:
            pass
        logger.warning(
            "Partial docs checkpoint failed for %s %s: %s", model, artifact_id, e,
        )
        return False
    finally:
        try:
            session.close()
        except Exception:
            pass


def _product_dataset(product: Any) -> str:
    """Product-scoped dataset key for memory indexing: ``prod_{product_id}``.

    ``_index_in_background`` extracts the ``product_id`` back out of this key
    (the active backend is selected by the ``memory.backend`` admin setting, so
    callers stay backend-agnostic). Falls back to ``unknown`` if the product
    has no id.
    """
    pid = getattr(product, "id", None) or getattr(product, "product_id", None) or "unknown"
    return f"prod_{pid}"


def _product_id_from_dataset(dataset_name: str) -> str:
    """Extract the product id from a ``prod_{id}`` dataset/key string."""
    if not dataset_name:
        return ""
    return dataset_name[len("prod_"):] if dataset_name.startswith("prod_") else dataset_name


def _index_in_background(
    content_or_path: str,
    dataset_name: str,
    *,
    source_type: str = "codebase",
    source_id: Optional[str] = None,
) -> None:
    """Fire-and-forget memory-backend indexing. Failures are logged, never fatal.

    Delegates to ``api.memory.index_document`` (active backend: pgvector). The
    indexing coroutine is handed off to the long-lived MAIN FastAPI event loop
    (captured at startup) via ``asyncio.run_coroutine_threadsafe`` so it
    survives the docgen worker thread's own short-lived loop teardown —
    indexing a large artifact can legitimately run for many minutes, and
    scheduling it on the worker loop would CANCEL it when the docgen job
    finishes. By moving it to the main loop, the job can return immediately
    while indexing continues in the background.
    """
    product_id = _product_id_from_dataset(dataset_name)

    async def _run() -> None:
        try:
            from api.memory import index_document  # lazy: backend optional
            await index_document(
                content_or_path, product_id,
                source_type=source_type, source_id=source_id,
            )
        except Exception as e:  # pragma: no cover - depends on live backend/DB
            logger.warning("Memory indexing failed for %r: %s", dataset_name, e)

    main_loop = get_main_event_loop()
    if main_loop is not None and main_loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(_run(), main_loop)
            logger.info(
                "Scheduled memory indexing for %r onto the main event loop.",
                dataset_name,
            )
            return
        except RuntimeError as e:  # loop closed between the check and the call
            logger.warning(
                "Could not schedule memory indexing on the main loop for %r (%s); "
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
            "No running event loop; skipping background memory indexing for %r.",
            dataset_name,
        )
