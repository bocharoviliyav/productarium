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
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

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
def _strip_wrapper_lead(text: str) -> str:
    """Drop a short chatter lead-in glued to a whole-output fenced wrapper.

    The one wrapper shape the preamble stripper cannot see: a presentation
    sentence ("The section is complete. Here is the final Markdown output:")
    directly before a ``` fence. Structural rule, not phrase matching: at
    most 2 short non-heading lines before the FIRST fence, the LAST lead
    line must end with ":" (the introducing-output signature), and the
    fence must CLOSE the text (whole-output wrapper). Anything else is
    document content and stays.
    """
    lines = text.split("\n")
    fence_idx = next(
        (i for i, ln in enumerate(lines[:3]) if ln.lstrip().startswith("```")),
        None,
    )
    if not fence_idx:
        return text
    lead = [ln for ln in lines[:fence_idx] if ln.strip()]
    if not lead or len(lead) > 2:
        return text
    if any(len(ln.strip()) > 200 or ln.lstrip().startswith("#") for ln in lead):
        return text
    if not lead[-1].rstrip().endswith(":"):
        return text
    if not lines or lines[-1].strip() != "```":
        return text
    return "\n".join(lines[fence_idx + 1:-1]).strip()


def _clean_llm_text(text: Optional[str]) -> str:
    """Deterministic docgen cleaning: strip agent artifacts, a whole-output
    fenced wrapper (with its chatter lead-in), meta preamble and a wrapping
    ```lang fence to a fixed point, then inline line-number prefixes inside
    code blocks. The loop matters because each strip can reveal the other
    shape ("preamble:\n```markdown\n…```") → fence survives a single pass)."""
    if not text:
        return ""
    t = _strip_agent_artifacts(text.strip())
    for _ in range(3):
        prev = t
        t = _strip_wrapper_lead(t)
        t = _strip_llm_preamble(t)
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
            if t.endswith("```"):
                t = t[:-3]
        t = t.strip()
        if t == prev:
            break
    return _strip_inline_line_numbers(t).strip()


def extract_json(text: Optional[str]) -> Optional[Any]:
    """Tolerant JSON extraction from an LLM answer: object OR array.

    One parser for every "model answered with prose/fences around the JSON"
    case (S-11): cleans artifacts + code fences via ``_clean_llm_text``, then
    tries the whole text and the outermost ``{...}``/``[...]`` slices ordered
    leftmost-first (prose around an array must not yield its inner dict).
    Returns the first ``json.loads`` hit (dict or list) or None; specialized
    callers keep only their expected type.
    """
    cleaned = _clean_llm_text(text or "")
    if not cleaned:
        return None
    spans = sorted(
        (cleaned.find(open_ch), cleaned.rfind(close_ch))
        for open_ch, close_ch in (("{", "}"), ("[", "]"))
    )
    candidates = [cleaned] + [
        cleaned[start:end + 1] for start, end in spans if 0 <= start < end
    ]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, (dict, list)):
            return data
    return None


# --- SPEC-004: page quality gate (agent-echo sanitizer + validator) ----------
# Local models sometimes echo their own tool-calling protocol into the final
# text: endless ``<tool_call>…</tool_call>`` blocks (including an UNCLOSED
# tail that runs to the end of the output), chat special tokens and the
# fence stubs left between stripped blocks. Removal is deterministic/LLM-free.
_TOOL_CALL_RE = re.compile(
    r"<tool_call>.*?</tool_call>\n?|<tool_call>[^\n]*\n?",
    re.DOTALL,
)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^<>\n|]{1,64}\|>")
_EMPTY_FENCE_RE = re.compile(r"```[^\n`]*\n(?:[^\S\n]*\n)*```")
#: Extra gate-driven attempts after the first run (SPEC-004).
PAGE_GATE_MAX_RETRIES = 2
_STRUCTURE_RE = re.compile(
    r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|\||`{3}\s*mermaid)", re.MULTILINE
)
_ARTIFACT_MARKERS = (
    "<tool_call", "<|im_start|>", "<|im_end|>", "We cannot invoke subagent",
)


def _strip_agent_artifacts(text: Optional[str]) -> str:
    """Drop tool-call echo (closed or unclosed), chat special tokens and
    empty fence stubs (to a fixed point — strips can reveal new stubs)."""
    if not text:
        return ""
    t = _TOOL_CALL_RE.sub("", text)
    t = _SPECIAL_TOKEN_RE.sub("", t)
    prev: Optional[str] = None
    while prev != t:
        prev = t
        t = _EMPTY_FENCE_RE.sub("", t)
    return t.strip()


def _has_open_fence(text: str) -> bool:
    """True when a ``` fence is left unclosed (token-limit truncation)."""
    open_ = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            open_ = not open_
    return open_


def _looks_truncated(text: str) -> bool:
    """Heuristic truncation detector for a generated unit body.

    Signals: an unclosed code fence (cut mid-block) or an abrupt mid-sentence
    ending — the last non-empty line ends with a letter or comma while the
    body is long enough for the ending to be meaningful.
    """
    if not text or not text.strip():
        return False
    if _has_open_fence(text):
        return True
    if len(text) < 500:
        return False
    last = [ln for ln in text.splitlines() if ln.strip()][-1].rstrip()
    return bool(last) and (last[-1].isalpha() or last[-1] == ",")


def _drop_dangling_fence_tail(text: str) -> str:
    """Remove a trailing unterminated ``` fence fragment (cut mid-block)."""
    lines = text.splitlines()
    last_open = -1
    open_ = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            open_ = not open_
            if open_:
                last_open = i
    if not open_ or last_open < 0:
        return text
    return "\n".join(lines[:last_open]).rstrip()


def validate_page(
    title: str,
    text: str,
    *,
    min_chars: int = 200,
    require_structure: bool = True,
) -> List[str]:
    """LLM-free O(text) page-quality validator (SPEC-004).

    Returns human-readable violations (empty list = page passes): empty
    title, too few meaningful chars, residual agent-echo markers after the
    sanitizer, a truncated-looking body (unclosed fence / abrupt ending,
    SPEC-015) and — when ``require_structure`` — a wall of prose without a
    single heading/list/table/mermaid signal. Entity pages whose contract
    IS a short paragraph call with ``require_structure=False`` and their own
    ``min_chars``.
    """
    violations: List[str] = []
    if not (title or "").strip():
        violations.append("пустой заголовок страницы")
    meaningful = len(re.sub(r"\s+", "", text or ""))
    if meaningful < min_chars:
        violations.append(
            f"текст короче {min_chars} значимых символов ({meaningful})"
        )
    lowered = (text or "").lower()
    leaked = [m for m in _ARTIFACT_MARKERS if m.lower() in lowered]
    if leaked:
        violations.append("остатки служебной разметки: " + ", ".join(leaked))
    if _looks_truncated(text or ""):
        violations.append("текст выглядит обрезанным (незакрытый код-фенс или обрыв предложения)")
    if (
        require_structure
        and meaningful >= min_chars
        and not _STRUCTURE_RE.search(text or "")
    ):
        violations.append(
            "нет ни одного заголовка, списка, таблицы или mermaid-диаграммы"
        )
    return violations


def page_gate_feedback(violations: List[str]) -> str:
    """Retry-prompt feedback block naming the gate violations (SPEC-004)."""
    items = "".join(f"- {v}.\n" for v in violations)
    return (
        "\n\n<quality_gate>\n"
        "The previous draft failed the automatic quality check:\n"
        f"{items}"
        "Write the page again from scratch: final clean Markdown only — no "
        "tool markup, tool dialogs or logs, and at least one heading, list, "
        "table or Mermaid diagram where appropriate.\n"
        "</quality_gate>"
    )


async def run_with_page_gate(
    run_once: Callable[[str], Awaitable[str]],
    *,
    title: str,
    label: str = "",
    progress: Optional[Any] = None,
    min_chars: int = 200,
    require_structure: bool = True,
) -> Tuple[str, bool, int]:
    """Run one page generation under the SPEC-004 quality gate.

    ``run_once(feedback)`` must return the cleaned candidate text; hard
    errors (incl. :class:`JobCancelledError`) propagate to the caller.
    Garbage results are retried with the violations as prompt feedback up to
    ``PAGE_GATE_MAX_RETRIES`` extra attempts; on exhaustion the SANITIZED
    last candidate is returned with ``degraded=True`` so the job still
    saves the page. Returns ``(text, degraded, attempts)``.
    """
    feedback = ""
    text = ""
    attempts = 0
    for attempt in range(1 + PAGE_GATE_MAX_RETRIES):
        attempts = attempt + 1
        text = await run_once(feedback)
        violations = validate_page(
            title, text, min_chars=min_chars, require_structure=require_structure,
        )
        if not violations:
            return text, False, attempts
        logger.warning(
            "page quality gate: %s attempt %d/%d failed: %s",
            label or title, attempts, 1 + PAGE_GATE_MAX_RETRIES,
            "; ".join(violations),
        )
        emit_progress(
            progress, phase="sections", gate_section=label or title,
            gate_attempt=attempts, gate_violations=violations,
        )
        feedback = page_gate_feedback(violations)
    logger.warning(
        "page quality gate: %s exhausted after %d attempt(s); saving "
        "sanitized degraded page",
        label or title, attempts,
    )
    return _strip_agent_artifacts(text), True, attempts


async def run_with_unit_budget(call: Awaitable[Any], *, seconds: float, label: str) -> Any:
    """SPEC-015 anti-stall watchdog: bound one unit agent run in wall clock.

    A unit that never finishes (hung tool call, infinite agent loop) is
    cancelled after ``seconds``; the TimeoutError propagates so the caller's
    existing retry/fallback machinery takes over while the job continues
    with the remaining units.
    """
    try:
        return await asyncio.wait_for(call, timeout=seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "unit %s stalled: cancelled after the %.0fs wall-clock budget",
            label, seconds,
        )
        raise


def utc_now_iso() -> str:
    """UTC now, ISO-8601 (SPEC-010 ``source_state`` timestamps)."""
    return datetime.now(timezone.utc).isoformat()


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


def collect_entity_parts(
    codebases: Iterable[Any] = (),
    databases: Iterable[Any] = (),
    specs: Iterable[Any] = (),
    nodes: Iterable[Any] = (),
    links: Iterable[Any] = (),
    *,
    budget_chars: int = 20_000,
    min_part_chars: int = 1_500,
) -> List[str]:
    """Fair-budget Markdown context parts across ALL product entities.

    One ``## <Kind>: <name>`` part per entity; every part is capped at
    ``budget_chars // len(parts)`` (floor ``min_part_chars`` while it fits —
    the joined total never exceeds ``budget_chars``) so every entity
    kind stays represented — the old global tail-cut starved everything after
    the first codebase. Shared by the product summary and the HLD pipeline.
    Databases/codebases fall back to concatenated page bodies when
    ``generated_docs`` is empty; links render their JSON url/description items.
    """

    def _docs_text(entity: Any) -> str:
        docs = getattr(entity, "generated_docs", None) or ""
        if docs.strip():
            return docs.strip()
        pages = getattr(entity, "pages", None)
        if isinstance(pages, dict):
            bodies = [
                (p or {}).get("content") or ""
                for p in pages.values()
                if isinstance(p, dict)
            ]
            joined = "\n\n".join(b for b in bodies if b.strip())
            if joined:
                return joined
        return ""

    def _name(entity: Any, fallback: str) -> str:
        return getattr(entity, "name", None) or getattr(entity, "id", None) or fallback

    parts: List[str] = []
    for c in codebases or ():
        text = _docs_text(c)
        if text:
            parts.append(f"## Codebase: {_name(c, 'codebase')}\n\n{text}")
    for d in databases or ():
        text = _docs_text(d)
        if text:
            parts.append(f"## Database: {_name(d, 'database')}\n\n{text}")
    for s in specs or ():
        content = (getattr(s, "content", None) or "").strip()
        if content:
            kind = getattr(s, "kind", None) or "spec"
            parts.append(f"## Спецификация ({kind}): {_name(s, 'spec')}\n\n{content}")
    for n in nodes or ():
        md = (getattr(n, "content_md", None) or "").strip()
        if md:
            title = getattr(n, "title", None) or getattr(n, "id", None) or "node"
            parts.append(f"## Страница базы знаний: {title}\n\n{md}")
    for lk in links or ():
        raw = (getattr(lk, "content", None) or "").strip()
        lines: List[str] = []
        if raw.startswith("["):
            try:
                for item in json.loads(raw):
                    if isinstance(item, dict):
                        url = str(item.get("url", "")).strip()
                        desc = str(item.get("description", "")).strip()
                        if url or desc:
                            lines.append(f"- {url}: {desc}".rstrip(": "))
            except (ValueError, TypeError):
                lines = []
        elif raw:
            lines = [raw]
        if lines:
            parts.append(
                f"## Ссылки: {_name(lk, 'links')}\n\n" + "\n".join(lines)
            )
    if not parts:
        return []
    fair = max(1, budget_chars // len(parts))
    # The floor keeps small entity sets richly quoted, but yields once N
    # parts would push the JOINED context past budget_chars — the sum bound
    # is what prevents context-window overflow on entity-heavy products.
    per_part = (
        max(min_part_chars, fair)
        if len(parts) * min_part_chars <= budget_chars
        else fair
    )
    return [_cap(p, per_part) for p in parts]


# ---------------------------------------------------------------------------
# Context-overflow detection + halving retry (shared docgen budget path)
# ---------------------------------------------------------------------------
# Error-text markers of a context-window overflow on OpenAI-compatible
# servers (400 + one of these strings).
_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "context length",
    "message too long",
    "maximum context length",
    "prompt is too long",
    "too many tokens",
    "max_tokens is too large",
)

# Halving stops here: below 8k a docgen prompt cannot stay useful.
_OVERFLOW_BUDGET_FLOOR_TOKENS = 8192


def _is_context_overflow(exc: BaseException) -> bool:
    """True when the error text says the request exceeded the context window."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _CONTEXT_OVERFLOW_MARKERS)


async def halve_budget_on_overflow(
    call: Callable[[int], Awaitable[Any]],
    budget: int,
    *,
    label: str = "LLM call",
) -> Any:
    """Run ``await call(budget)``; on a context-overflow error retry with the
    prompt budget halved (floor ``_OVERFLOW_BUDGET_FLOOR_TOKENS``).

    A window the server overstates then degrades the prompt to a smaller fit
    instead of failing the whole unit. Re-raises the last error when the
    overflow persists at the floor or the error is not an overflow.
    """
    while True:
        try:
            return await call(budget)
        except Exception as e:
            halved = max(_OVERFLOW_BUDGET_FLOOR_TOKENS, budget // 2)
            if halved >= budget or not _is_context_overflow(e):
                raise
            logger.warning(
                "%s exceeded the model context window; retrying with a "
                "halved prompt budget (%d -> %d tokens)",
                label, budget, halved,
            )
            budget = halved


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


def _carry_page_verify_flags(
    new_pages: Dict[str, Any], old_pages: Dict[str, Any]
) -> None:
    """Carry per-page verification flags onto regenerated pages whose content
    is byte-identical (verification binds to exact content; a regenerated page
    resets naturally). Mutates ``new_pages`` in place (fresh dicts at persist)."""
    if not isinstance(new_pages, dict) or not isinstance(old_pages, dict):
        return
    for page_id, new_page in new_pages.items():
        old_page = old_pages.get(page_id)
        if (
            isinstance(new_page, dict)
            and isinstance(old_page, dict)
            and old_page.get("verified")
            and new_page.get("content") == old_page.get("content")
        ):
            for key in ("verified", "verified_by", "verified_at"):
                new_page[key] = old_page.get(key)


def plain_agent_tools(tools: List[Any]) -> List[Any]:
    """Rewrap MCP adapter tools (``response_format=content_and_artifact``)
    as plain string tools for deepagents subagents.

    langchain-mcp-adapters' error handler returns a bare list on
    ``isError=True`` results, which langchain_core rejects with a ValueError
    — killing the whole agent instead of degrading one call. The wrapper
    invokes the original tool and flattens ANY result or exception to text.
    Non-adapter tools pass through untouched.
    """
    from langchain_core.tools import StructuredTool

    def _text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for block in value:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                else:
                    parts.append(str(block))
            return "".join(parts)
        content = getattr(value, "content", None)
        return _text(content) if isinstance(content, (str, list)) else str(value)

    out: List[Any] = []
    for original in tools:
        if getattr(original, "response_format", None) != "content_and_artifact":
            out.append(original)
            continue

        async def _run(tool: Any = original, **kwargs: Any) -> str:
            try:
                return _cap(_text(await tool.ainvoke(dict(kwargs))), 100_000)
            except Exception as e:
                return _cap(f"ERROR: {type(e).__name__}: {e}".split("\n")[0], 4000)

        out.append(StructuredTool(
            name=original.name,
            description=original.description,
            args_schema=getattr(original, "args_schema", None),
            coroutine=_run,
        ))
    return out


# Matches a markdown heading line; used to bound the provenance block.
_ANY_HEADING_RE = re.compile(r"^(#{1,6})\s")
_PROV_HEADING_RE = re.compile(r"^(#{2,4})\s*(.+?)\s*$")


def _is_provenance_heading(line: str) -> Optional[int]:
    """Heading level when ``line`` is a provenance-block heading, else None.

    Matches the block mandated by ``docgen_agent_section.md`` —
    ``### Провенанс и проверка`` — plus natural EN variants
    (``### Provenance and verification``), at heading levels 2-4.
    """
    m = _PROV_HEADING_RE.match(line)
    if not m:
        return None
    title = m.group(2).strip().lower().rstrip("#").strip()
    has_prov = "провенанс" in title or "provenance" in title
    has_ver = "проверка" in title or "verification" in title
    return len(m.group(1)) if has_prov and has_ver else None


def _split_provenance_block(content: str) -> Tuple[str, Optional[str]]:
    """Split the LLM-written provenance block out of a generated page.

    Every unit ends with a compact ``### Провенанс и проверка`` block (key
    sources, assumptions, gaps, confidence). The report is rendered inside
    the verification panel instead, so it is extracted from the persisted
    text into ``provenance["report"]``. The LAST matching heading wins (the
    contract puts the block at the end); the block spans to the next heading
    of the same or higher level. Pages without the block (or consisting of
    only the block) return unchanged.
    """
    if not content:
        return content, None
    lines = content.split("\n")
    start = start_level = None
    for i in range(len(lines) - 1, -1, -1):
        level = _is_provenance_heading(lines[i])
        if level is not None:
            start, start_level = i, level
            break
    if start is None:
        return content, None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = _ANY_HEADING_RE.match(lines[j])
        if m and len(m.group(1)) <= start_level:
            end = j
            break
    head = "\n".join(lines[:start]).rstrip()
    report = "\n".join(lines[start + 1:end]).strip()
    tail = "\n".join(lines[end:]).strip()
    if not head and not tail:
        return content, None
    kept = "\n\n".join(p for p in (head, tail) if p)
    return kept, report or None


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

    _schedule_indexing(_run, dataset_name)


def _index_pages_in_background(
    pages: Dict[str, Any],
    dataset_name: str,
    *,
    source_type: str,
    source_id: Optional[str],
    only: Optional[Iterable[str]] = None,
) -> None:
    """Fire-and-forget PAGE-granular memory indexing (SPEC-010).

    Each non-empty page is indexed under its own
    ``{source_id}::page::{page_id}`` source, so the backend's
    delete-then-insert upsert re-embeds exactly one page per call.
    ``only`` limits re-indexing to those page ids (incremental runs) — the
    other pages keep their stored chunks; a first run over a legacy
    blob-indexed artifact (no page sources yet) bootstraps ALL pages so the
    prune below cannot drop unchanged pages from recall. The prune deletes
    legacy artifact-level chunks and page sources that no longer have a
    non-empty page. No-op when there is nothing indexable (page-less legacy
    artifacts keep their blob-level indexing).
    """
    if not source_id or not isinstance(pages, dict):
        return
    product_id = _product_id_from_dataset(dataset_name)
    indexable: Dict[str, str] = {}
    for pid, page in pages.items():
        content = page.get("content") if isinstance(page, dict) else page
        if isinstance(content, str) and content.strip():
            indexable[pid] = content
    if not indexable:
        return
    wanted = set(only) if only is not None else set(indexable)

    async def _run() -> None:
        try:
            from api.memory import (  # lazy: backend optional
                index_document,
                list_page_sources,
                prune_stale_sources,
            )
            if only is not None and not await list_page_sources(
                product_id, source_type, source_id
            ):
                wanted.update(indexable)
            await prune_stale_sources(
                product_id, source_type, source_id,
                [f"{source_id}::page::{pid}" for pid in indexable],
            )
            for pid in sorted(wanted):
                await index_document(
                    indexable[pid], product_id,
                    source_type=source_type,
                    source_id=f"{source_id}::page::{pid}",
                )
            # SPEC-008: extract terms from the SAME changed pages into the
            # product knowledge graph (gated + budgeted inside; non-fatal).
            from api.config.timeout import resolve_timeout_bool

            if resolve_timeout_bool("knowledge_graph_enabled"):
                from api.memory.knowledge_graph import update_graph_from_pages

                await update_graph_from_pages(
                    product_id, source_type, source_id, pages, only=set(wanted),
                )
        except Exception as e:  # pragma: no cover - depends on live backend/DB
            logger.warning("Page memory indexing failed for %r: %s", dataset_name, e)

    _schedule_indexing(_run, dataset_name)


def _schedule_indexing(
    coro_factory: Callable[[], Any], dataset_name: str
) -> None:
    """Hand an indexing coroutine factory to the long-lived main event loop.

    Shared by the blob and page-granular indexing helpers: the coroutine is
    scheduled via ``asyncio.run_coroutine_threadsafe`` so it survives the
    docgen worker thread's loop teardown (see ``_index_in_background``).
    """
    main_loop = get_main_event_loop()
    if main_loop is not None and main_loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(coro_factory(), main_loop)
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
        asyncio.create_task(coro_factory())
    except RuntimeError:
        # No running event loop -- best-effort skip.
        logger.warning(
            "No running event loop; skipping background memory indexing for %r.",
            dataset_name,
        )


def reindex_entity(
    product_id: str,
    indexed_text: Optional[str],
    entity_id: str,
    *,
    source_type: str = "codebase",
    page_id: Optional[str] = None,
    pages: Optional[dict] = None,
) -> None:
    """Re-index router-edited text into the product memory (fire-and-forget).

    Shared by the products/databases routers (Q-9): a single-page edit targets
    ``{entity_id}::page::{page_id}`` (the same source_id generation writes); a
    wholesale ``pages`` replace re-indexes every page and prunes stale page
    sources exactly like generation; a single-page edit that CLEARED the
    content (empty ``indexed_text`` + ``page_id``) prunes only that page's
    chunks, keeping the legacy blob and the other pages (S-12). Everything
    else (spec/links content, the generated_docs blob) stays artifact-level
    with ``source_id`` = entity id.
    """
    try:
        dataset = f"prod_{product_id}"
        if page_id and indexed_text and indexed_text.strip():
            _index_in_background(
                indexed_text, dataset,
                source_type=source_type,
                source_id=f"{entity_id}::page::{page_id}",
            )
        elif isinstance(pages, dict) and pages:
            _index_pages_in_background(
                pages, dataset, source_type=source_type, source_id=entity_id,
            )
        elif page_id:
            stale = f"{entity_id}::page::{page_id}"

            async def _prune_page() -> None:
                try:
                    from api.memory import list_page_sources, prune_stale_sources

                    stored = await list_page_sources(product_id, source_type, entity_id)
                    await prune_stale_sources(
                        product_id, source_type, entity_id,
                        [entity_id] + [s for s in stored if s != stale],
                    )
                except Exception as e:  # pragma: no cover - live backend
                    logger.warning("Page chunk prune failed for %r: %s", dataset, e)

            _schedule_indexing(_prune_page, dataset)
        elif indexed_text and indexed_text.strip():
            _index_in_background(
                indexed_text, dataset,
                source_type=source_type, source_id=entity_id,
            )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Memory re-index failed for entity %s: %s", entity_id, e)
