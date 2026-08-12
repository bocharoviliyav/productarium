"""Mermaid diagram verifier + bounded LLM repair loop.

Validates Mermaid diagrams produced during documentation generation by running
them through the headless Node validator at ``api/_mermaid_validate.mjs``
(which calls ``mermaid.parse()`` — no browser/puppeteer needed). Broken
diagrams are queued and repaired via a focused LLM call within a bounded retry
budget; diagrams that cannot be repaired (or that cannot be judged headlessly,
e.g. C4) are left in place with a visible error marker so the user still sees
the original.

Public API
----------
- :data:`MERMAID_VERIFY_ENABLED` — env-only master switch. Per-diagram verify /
  repair timeouts and the repair-attempt budget are resolved per-call through
  :mod:`api.timeout_config` (admin store > env var > default).
- :func:`extract_mermaid_blocks` — split markdown into fenced mermaid blocks.
- :func:`verify_diagram` — validate one diagram body via the Node subprocess.
- :func:`run_repair_loop` — extract → verify → enqueue → repair → splice; the
  single entry point wired into both generation paths
  (``api/docgen/codebase.py`` and ``api/websocket_wiki.py``).

Design rules (from the plan / user flow):
- Verification of all diagrams in a page runs in parallel (``asyncio.gather``)
  alongside generation; repairs drain the queue after the page's diagrams are
  all verified.
- The repair budget is keyed by a normalized hash of the diagram body so that a
  semantically-identical re-suggestion counts toward the SAME budget unit
  (avoids burning tokens on the same broken diagram).
- A diagram whose verifier error signature is "unverifiable" (e.g. C4's
  ``Bt.addHook is not a function`` headless failure) is left untouched and is
  NOT queued — we cannot tell a valid C4 diagram from an invalid one without a
  DOM, so repairing would be noise.
- Everything is non-fatal: a missing Node / mermaid bundle degrades to
  "unverifiable" so generation never breaks because of the verifier.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from api.timeout_config import (
    resolve_mermaid_max_repair_attempts,
    resolve_mermaid_repair_timeout,
    resolve_mermaid_verify_timeout,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Master switch (env-only). ``false`` disables verification entirely and
# run_repair_loop returns the markdown unchanged — generation behaves exactly
# as before. The per-diagram verify timeout, per-LLM-repair-call timeout, and
# the per-unique-body repair budget are resolved per-call through
# api.timeout_config (admin store > env var > default); see the
# mermaid_verify / mermaid_repair / mermaid_max_repair_attempts entries in
# TIMEOUT_KEYS. Editing them in the admin panel takes effect on the next
# verify_diagram / repair_diagram / run_repair_loop call without a restart.
MERMAID_VERIFY_ENABLED: bool = os.environ.get("MERMAID_VERIFY", "true").strip().lower() in (
    "1", "true", "yes", "on",
)

# Path to the headless Node validator, resolved relative to the api/ package
# root so it works regardless of the process CWD. The validator lives at
# ``api/_mermaid_validate.mjs``; this module lives at ``api/formats/mermaid.py``,
# so we go up one level from this file's directory.
_VALIDATOR_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_mermaid_validate.mjs",
)

# Marker emitted on stderr by the Node script when the mermaid import fails.
_IMPORT_FAILED_MARKER = "MERMAID_IMPORT_FAILED"


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------
# Matches fenced ```mermaid blocks. Tolerant of:
# - leading language variants: ```mermaid, ```mermaid{.theme}, ```mermaid-123
# - CRLF/LF
# - the opening fence having extra trailing spaces
# A block captures the inner body (between the fences, fences excluded).
_MERMAID_FENCE_RE = re.compile(
    r"```mermaid[^\n`]*\n(?P<body>.*?)```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class MermaidBlock:
    """A fenced ```mermaid block found in a markdown document.

    Attributes
    ----------
    index:
        0-based ordinal of the block within the document (for stable splice
        ordering after repairs shift offsets).
    raw:
        The full original text including the opening/closing fences.
    body:
        The inner diagram text (between the fences).
    start, end:
        Character offsets of ``raw`` within the source markdown. Updated by
        :func:`run_repair_loop` after every splice so subsequent splices stay
        correct.
    """

    index: int
    raw: str
    body: str
    start: int
    end: int


def extract_mermaid_blocks(markdown: str) -> List[MermaidBlock]:
    """Return all fenced ```mermaid blocks in ``markdown`` (in document order)."""
    if not markdown:
        return []
    blocks: List[MermaidBlock] = []
    for i, m in enumerate(_MERMAID_FENCE_RE.finditer(markdown)):
        raw = m.group(0)
        body = m.group("body")
        blocks.append(
            MermaidBlock(
                index=i,
                raw=raw,
                body=body,
                start=m.start(),
                end=m.end(),
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# Verification (Node subprocess)
# ---------------------------------------------------------------------------
@dataclass
class VerifyResult:
    """Outcome of validating a single diagram body."""

    ok: bool
    error: Optional[str] = None
    diagram_type: Optional[str] = None
    unverifiable: bool = False  # True => cannot judge headlessly (e.g. C4)


def _has_node() -> bool:
    return shutil.which("node") is not None


async def verify_diagram(body: str, timeout: Optional[float] = None) -> VerifyResult:
    """Validate ``body`` headlessly via the Node validator subprocess.

    Returns a :class:`VerifyResult`. Any infrastructure failure (Node missing,
    script absent, non-zero exit, unparseable stdout, timeout) is downgraded to
    a conservative ``unverifiable=True`` result so generation is never blocked
    by the verifier — the diagram is simply left in place.
    """
    if not body or not body.strip():
        return VerifyResult(ok=True, diagram_type=None)
    if not _has_node() or not os.path.isfile(_VALIDATOR_SCRIPT):
        logger.warning(
            "Mermaid verifier unavailable (node=%s, script=%s); skipping "
            "validation. Install node or set MERMAID_VERIFY=false to silence.",
            _has_node(), os.path.isfile(_VALIDATOR_SCRIPT),
        )
        return VerifyResult(ok=True, unverifiable=True)

    timeout = timeout if timeout is not None else resolve_mermaid_verify_timeout()

    async def _run() -> VerifyResult:
        proc = await asyncio.create_subprocess_exec(
            "node", _VALIDATOR_SCRIPT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=body.encode("utf-8")),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            logger.warning("Mermaid verify timed out after %ss; treating as unverifiable.", timeout)
            return VerifyResult(ok=True, unverifiable=True)

        stderr = (stderr_b or b"").decode("utf-8", errors="replace")
        if proc.returncode != 0:
            # Exit code 3 => import failed (no mermaid in node_modules).
            if _IMPORT_FAILED_MARKER in stderr:
                logger.warning(
                    "Mermaid validator import failed (mermaid not in "
                    "node_modules); skipping validation. %s",
                    stderr.strip(),
                )
            else:
                logger.warning("Mermaid validator exited %s: %s", proc.returncode, stderr.strip())
            return VerifyResult(ok=True, unverifiable=True)

        line = (stdout_b or b"").decode("utf-8", errors="replace").strip()
        if not line:
            return VerifyResult(ok=True, unverifiable=True)
        # The script prints exactly one JSON object; be defensive and take the
        # last non-empty line (in case a stray newline sneaks in).
        lines = [ln for ln in line.splitlines() if ln.strip()]
        if not lines:
            return VerifyResult(ok=True, unverifiable=True)
        try:
            data = json.loads(lines[-1])
        except json.JSONDecodeError:
            logger.warning("Mermaid validator returned non-JSON: %r", lines[-1][:200])
            return VerifyResult(ok=True, unverifiable=True)

        return VerifyResult(
            ok=bool(data.get("ok", False)),
            error=(data.get("error") or None),
            diagram_type=(data.get("diagramType") or None),
            unverifiable=bool(data.get("unverifiable", False)),
        )

    return await _run()


async def _verify_many(bodies: List[str]) -> List[VerifyResult]:
    """Verify a list of diagram bodies in parallel."""
    return await asyncio.gather(*(verify_diagram(b) for b in bodies))


# ---------------------------------------------------------------------------
# Repair queue
# ---------------------------------------------------------------------------
def _normalize_body(body: str) -> str:
    """Normalize a diagram body for stable hashing.

    Trailing whitespace per line and overall trailing whitespace are stripped so
    that whitespace-only edits by the LLM do not count as a new unique diagram.
    """
    if not body:
        return ""
    lines = [ln.rstrip() for ln in body.splitlines()]
    return "\n".join(lines).strip()


def _body_hash(body: str) -> str:
    return hashlib.sha1(_normalize_body(body).encode("utf-8")).hexdigest()


@dataclass
class RepairJob:
    """A queued repair task for one broken diagram."""

    block_index: int
    body: str
    error: Optional[str]
    attempt: int = 0
    original_body: str = ""


# ---------------------------------------------------------------------------
# LLM repair
# ---------------------------------------------------------------------------
# Prompt body lives in refs/prompts/mermaid_repair.md (externalized, editable
# without code changes — matches the project pattern). It is loaded into the
# ``api.prompts.MERMAID_REPAIR_PROMPT`` module constant at import time and is
# hot-reloadable via the admin panel (``reload_prompt_file``), so we read the
# attribute fresh each call rather than caching locally — an admin edit takes
# effect on the next repair without a process restart.
_REPAIR_PROMPT_FALLBACK = (
    "<role>\n"
    "Ты — эксперт по Mermaid-диаграммам. Исправь сломанную диаграмму.\n"
    "</role>\n\n"
    "<broken_diagram>\n{broken_diagram}\n</broken_diagram>\n\n"
    "<error>\n{error}\n</error>\n\n"
    "<requirements>\n"
    "Верни ТОЛЬКО исправленную Mermaid-диаграмму в блоке ```mermaid ... ```.\n"
    "Не добавляй пояснений, комментариев или дополнительного текста.\n"
    "Сохрани смысл и элементы исходной диаграммы. Язык подписей — как в оригинале.\n"
    "</requirements>"
)


def _load_repair_prompt() -> str:
    try:
        from api import prompts as _prompts
        prompt = getattr(_prompts, "MERMAID_REPAIR_PROMPT", "") or ""
        # If the constant is empty (refs/prompts/mermaid_repair.md missing),
        # fall back to the inline skeleton so repair still works.
        return prompt or _REPAIR_PROMPT_FALLBACK
    except Exception:
        return _REPAIR_PROMPT_FALLBACK


def _extract_repaired_body(text: str) -> Optional[str]:
    """Pull the mermaid block body out of an LLM repair response.

    Accepts either a fenced ```mermaid block or a bare diagram (fallback).
    Returns the inner body, or None if nothing usable was found.
    """
    if not text:
        return None
    t = text.strip()
    m = _MERMAID_FENCE_RE.search(t)
    if m:
        return m.group("body")
    # Bare diagram fallback: strip a leading ``` fence if the model forgot the
    # closing one.
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    # Heuristic: a real mermaid body has a diagram keyword on the first line.
    first = (t.splitlines() or [""])[0].strip().lower()
    if any(kw in first for kw in ("flowchart", "graph", "sequence", "class", "state", "er", "gantt", "pie", "journey", "mindmap", "c4", "gitgraph")):
        return t
    return None


# Type alias: an async callable (prompt) -> str that runs the LLM. Both
# generation paths pass a thin wrapper over their existing LLM client so no new
# provider/model config path is needed.
LLMCallable = Callable[[str], Awaitable[str]]


async def repair_diagram(job: RepairJob, llm: LLMCallable) -> Optional[str]:
    """Ask the LLM to fix ``job``'s diagram; return the repaired body or None.

    The repair call is bounded by the mermaid-repair timeout, resolved per call
    through api.timeout_config. Any failure returns None so the caller can
    decide whether to re-enqueue (within the budget) or give up and mark the
    diagram.
    """
    template = _load_repair_prompt()
    prompt = template.replace("{broken_diagram}", job.body).replace(
        "{error}", job.error or "Неизвестная ошибка синтаксиса"
    )
    repair_timeout = resolve_mermaid_repair_timeout()
    try:
        raw = await asyncio.wait_for(llm(prompt), timeout=repair_timeout)
    except asyncio.TimeoutError:
        logger.warning("Mermaid repair LLM call timed out after %ss.", repair_timeout)
        return None
    except Exception as e:  # pragma: no cover - depends on live LLM
        logger.warning("Mermaid repair LLM call failed: %s", e)
        return None
    return _extract_repaired_body(_strip_llm_fences(raw))


def _strip_llm_fences(text: str) -> str:
    """Best-effort strip of a wrapping ```markdown fence around the LLM output."""
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```") and not t.startswith("```mermaid"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


# ---------------------------------------------------------------------------
# Error marker for unrecoverable diagrams
# ---------------------------------------------------------------------------
def _error_marker(attempts: int) -> str:
    """Markdown blockquote shown after a diagram that exhausted the repair budget."""
    return (
        f"\n\n> ⚠️ **Mermaid**: диаграмма не отрисовывается (syntax error). "
        f"Попыток исправления: {attempts}. Исходный текст сохранён выше.\n"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
async def run_repair_loop(
    page_markdown: str,
    llm: Optional[LLMCallable],
    *,
    on_progress: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
) -> Tuple[str, Dict[str, int]]:
    """Validate + repair all mermaid blocks in ``page_markdown``.

    Flow (per the user's design):
    1. Extract all ```mermaid blocks.
    2. Verify all of them in parallel.
    3. Enqueue the broken (but headlessly-judgeable) ones; skip unverifiable
       ones (e.g. C4) and leave them in place.
    4. Drain the queue: for each job, call the LLM, verify the result; if ok,
       splice the fixed body back into the markdown; if not ok, re-enqueue only
       if the unique-body budget (``MAX_REPAIR_ATTEMPTS``) is not exhausted.
    5. For diagrams that exhaust the budget, leave the original broken block in
       place and append a visible error marker after it.

    Parameters
    ----------
    page_markdown:
        The generated section/page markdown that may contain mermaid blocks.
    llm:
        Async callable ``(prompt) -> str`` used for repairs. If None, no repairs
        are attempted (broken diagrams are left in place with a marker).
    on_progress:
        Optional async callback receiving a stats dict ``{verified, broken,
        unverifiable, fixed, failed}`` — used by the websocket path to emit a
        progress message.

    Returns
    -------
    (patched_markdown, stats)
    """
    stats = {"verified": 0, "broken": 0, "unverifiable": 0, "fixed": 0, "failed": 0}

    if not MERMAID_VERIFY_ENABLED:
        return page_markdown, stats

    # Resolve the per-unique-body repair budget once for this loop run so the
    # budget stays consistent across the drain (admin edits take effect on the
    # next run_repair_loop call, not mid-drain).
    max_attempts = resolve_mermaid_max_repair_attempts()

    blocks = extract_mermaid_blocks(page_markdown)
    if not blocks:
        return page_markdown, stats

    # 1+2. Verify all blocks in parallel.
    results = await _verify_many([b.body for b in blocks])

    # 3. Enqueue broken-but-judgeable diagrams. Track per-unique-body attempt
    # counts so the budget is shared across semantically-identical suggestions.
    attempt_counts: Dict[str, int] = {}
    queue: List[RepairJob] = []
    for block, res in zip(blocks, results):
        if res.ok:
            stats["verified"] += 1
            continue
        if res.unverifiable:
            stats["unverifiable"] += 1
            continue
        stats["broken"] += 1
        h = _body_hash(block.body)
        attempt_counts[h] = attempt_counts.get(h, 0)
        if attempt_counts[h] < max_attempts:
            queue.append(
                RepairJob(
                    block_index=block.index,
                    body=block.body,
                    error=res.error,
                    original_body=block.body,
                )
            )
        # else: already at budget for this unique body — will be marked below.

    # 4. Drain the repair queue.
    # We keep a list of (block_index, new_body) splices to apply, and a set of
    # block indices that have been definitively fixed so they aren't remarked.
    fixed_bodies: Dict[int, str] = {}    # block_index -> repaired body
    # Cache successful repairs by broken-body hash so that duplicate identical
    # broken diagrams on the same page reuse the SAME fix instead of each
    # consuming its own repair budget (and LLM tokens). This also keeps the
    # per-unique-body budget meaningful: one unique broken diagram = one budget.
    fixed_by_hash: Dict[str, str] = {}   # body_hash -> repaired body
    marked: set = set()                  # block_index -> needs error marker

    while queue:
        job = queue.pop(0)
        h = _body_hash(job.body)
        # Reuse a previously-successful fix for an identical broken diagram.
        if h in fixed_by_hash:
            fixed_bodies[job.block_index] = fixed_by_hash[h]
            stats["fixed"] += 1
            continue
        if attempt_counts[h] >= max_attempts:
            marked.add(job.block_index)
            stats["failed"] += 1
            continue

        if llm is None:
            marked.add(job.block_index)
            stats["failed"] += 1
            continue

        attempt_counts[h] += 1
        job.attempt = attempt_counts[h]
        repaired = await repair_diagram(job, llm)
        if not repaired:
            # LLM gave nothing usable; re-enqueue if budget remains.
            if attempt_counts[h] < max_attempts:
                queue.append(job)
            else:
                marked.add(job.block_index)
                stats["failed"] += 1
            continue

        # Verify the repaired body.
        vr = await verify_diagram(repaired)
        if vr.ok:
            fixed_bodies[job.block_index] = repaired
            fixed_by_hash[h] = repaired
            stats["fixed"] += 1
            continue
        if vr.unverifiable:
            # Repaired into a type we can't judge headlessly (e.g. became C4).
            # Accept it optimistically — it's no worse than the original.
            fixed_bodies[job.block_index] = repaired
            fixed_by_hash[h] = repaired
            stats["fixed"] += 1
            continue
        # Repaired body is still broken. Re-enqueue under the NEW body's hash so
        # a different broken variant also gets its own budget.
        new_h = _body_hash(repaired)
        attempt_counts[new_h] = attempt_counts.get(new_h, 0)
        if attempt_counts[new_h] < max_attempts:
            queue.append(
                RepairJob(
                    block_index=job.block_index,
                    body=repaired,
                    error=vr.error,
                    original_body=job.original_body,
                )
            )
        else:
            marked.add(job.block_index)
            stats["failed"] += 1

    # 5. Apply splices + error markers. Splice from the LAST block to the first
    # so earlier offsets stay valid. Re-extract offsets defensively each time.
    patched = page_markdown
    # Combine: for each block index, either a fixed body or (if marked) the
    # original body + error marker.
    to_apply: List[Tuple[int, str]] = []  # (block_index, replacement_raw)
    for block in blocks:
        if block.index in fixed_bodies:
            to_apply.append((block.index, f"```mermaid\n{fixed_bodies[block.index]}```"))
        elif block.index in marked:
            attempts_used = attempt_counts.get(_body_hash(block.body), max_attempts)
            to_apply.append((block.index, block.raw + _error_marker(attempts_used)))
    # Apply in reverse document order.
    for block in reversed(blocks):
        for bi, repl in to_apply:
            if bi == block.index:
                patched = patched[: block.start] + repl + patched[block.end:]
                break

    if on_progress is not None:
        try:
            await on_progress(stats)
        except Exception:  # pragma: no cover - progress callback must not break gen
            logger.debug("mermaid on_progress callback raised; ignored.", exc_info=True)

    logger.info(
        "Mermaid repair loop: verified=%d broken=%d unverifiable=%d fixed=%d failed=%d",
        stats["verified"], stats["broken"], stats["unverifiable"],
        stats["fixed"], stats["failed"],
    )
    return patched, stats


__all__ = [
    "MERMAID_VERIFY_ENABLED",
    "MermaidBlock",
    "VerifyResult",
    "RepairJob",
    "extract_mermaid_blocks",
    "verify_diagram",
    "repair_diagram",
    "run_repair_loop",
]
