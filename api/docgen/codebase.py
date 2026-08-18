"""Codebase documentation generation (RLM long-context + standard-LLM fallback).

Split out of the former ``api/artifact_docgen.py`` (Step 4). Generates the 7
wiki sections for a codebase artifact: the repo is cloned via
``api.data_pipeline.DatabaseManager._create_repo``, its files are read into a
long-context blob, and each section is generated from the matching
``refs/prompts/<section>.md`` template (variable mapping reused from
``api.docgen.wiki.WikiGenerator._format_prompt``). RLM (fast-rlm) is used for
long context; the standard LLM (adalflow, local OpenAI-compatible server) is
used when the codebase is small or RLM fails. The repo is indexed into cognee
in the background after generation.

Shared LLM/persistence helpers live in ``api.docgen._common``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from api.utils import setup_logging
from api.formats.mermaid import run_repair_loop
from api.prompts import get_section_title
from api.docgen.wiki import (
    WikiGenerator,
    create_wiki_section_context,
)
from api.docgen._common import (
    _clean_llm_text,
    _with_verification_guard,
    _StandardLLM,
    _resolve_docgen_model,
    _safe_build_llm,
    _make_repair_llm,
    _persist_artifact,
    _cognee_dataset,
    _index_in_background,
    _repo_name_from_url,
)

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# RLM is for long context only (per the plan). Below this codebase-blob size we
# skip RLM and use the standard LLM directly.
RLM_MIN_CHARS = 20_000
# When using the standard LLM for a SMALL codebase, append the codebase blob to
# the section prompt only if it fits this limit (avoids blowing the context).
SMALL_CODEBASE_APPEND_LIMIT = 20_000
# Cap the long-context blob handed to RLM so very large repos stay manageable.
CODEBASE_BLOB_MAX_CHARS = 200_000
# Per-file cap inside the codebase blob.
PER_FILE_MAX_CHARS = 8_000
# Cap for {content} substituted into openapi/asyncapi/testcase LLM prompts.
LLM_CONTENT_MAX_CHARS = 50_000
# Per-section RLM timeout: a single long-context completion on a LOCAL model
# (e.g. LM Studio running qwen3.6-27b over a 200k-char codebase blob) can take
# several minutes, so the default must be generous. This is a safety net for a
# genuinely hung fast-rlm (first-run npm/pyodide bootstrap) -- the per-API-call
# timeout lives in rlm_runner.py (RLM_API_TIMEOUT_MS). When this section timeout
# fires, only the awaited result is discarded; the underlying RLM worker thread
# keeps running. Resolved at call time via the central timeout config
# (admin > env > default) so an admin save takes effect without a restart.
# Default 1800s (30 min), floor 60s.
# Max RLM failures within a single generate run before skipping RLM for the
# remaining sections. fast-rlm runs the model inside a Pyodide Python REPL;
# after this many failures we stop trying RLM and go straight to the standard
# LLM for the rest of the run. Default 2 failures.
RLM_MAX_FAILURES = int(os.environ.get("RLM_MAX_FAILURES", "2"))

# Placeholder used when a single section's generation produces no usable text.
# Kept as a named constant so callers can detect an all-placeholder result and
# surface a clear failure instead of committing placeholder-only docs as success.
_SECTION_UNAVAILABLE_PLACEHOLDER = (
    "_(Содержимое раздела временно недоступно. Вы можете перезапустить генерацию.)_"
)

# fast-rlm's default ``max_prompt_tokens`` (200000) is the hard ceiling a single
# RLM call must stay under (the whole codebase blob + fast-rlm's recursive
# subagent outputs count toward it). On large repos the full blob alone can
# approach that limit, leaving no room for the recursive accumulation -- which
# surfaces as ``Prompt token budget exceeded``. To keep each RLM call safe we
# reserve this many tokens (for the section prompt + fast-rlm recursion) and
# split the codebase into per-call chunks that each fit in what remains.
# Tunable via env. See ``_resolve_codebase_chunk_budget``.
RLM_PROMPT_RESERVE_TOKENS = int(os.environ.get("RLM_PROMPT_RESERVE_TOKENS", "40000"))


def _resolve_rlm_context_window() -> Optional[int]:
    """Resolve the model's actual context-window ceiling in tokens.

    Uses ``get_model_context_window(task="docgen")`` which checks explicit env
    vars, admin settings, live API metadata (/api/show or /v1/models), model name
    heuristics, and a safe default (8192).
    """
    try:
        from api.utils import get_model_context_window
        return get_model_context_window(task="docgen")
    except Exception as e:
        logger.debug("Could not resolve context window in codebase docgen: %s", e)
        return 8192


# Token-counting is approximate by design: we need a budget estimate, not an
# exact count. tiktoken cl100k_base (the path in data_pipeline.count_tokens)
# is a good fit for the local models used here; if tiktoken is unavailable we
# fall back to a len//4 character ratio so chunking still works (just less
# precise). The estimate is intentionally conservative (chunks end up slightly
# smaller than the budget, which is the safe direction).
_TIKTOKEN_ENC = None


def _count_tokens(text: str) -> int:
    """Approximate token count for a chunk-budget estimate.

    Uses tiktoken ``cl100k_base`` (matches the path in
    ``api.data_pipeline.count_tokens``) when available; otherwise falls back to
    a 4-chars-per-token ratio. Never raises: on any error the fallback is used.
    """
    if not text:
        return 0
    global _TIKTOKEN_ENC
    try:
        if _TIKTOKEN_ENC is None:
            import tiktoken  # type: ignore
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        return len(_TIKTOKEN_ENC.encode(text, disallowed_special=()))
    except Exception:
        # Conservative ratio so chunks err on the small side.
        return max(1, len(text) // 4)


def _resolve_codebase_chunk_budget() -> int:
    """Resolve the per-call codebase token budget for RLM.

    The codebase chunk budget must account for both:
      1. the model's actual context window (``num_ctx``) minus completion reserve;
      2. fast-rlm's REPL subagent recursion overhead (section prompt + system
         instructions + Pyodide code execution outputs).

    If the codebase chunk is too large (e.g. 26k tokens on a 32k window), fast-rlm
    subagents will exceed `max_prompt_tokens` (e.g. 24,576) on their first
    recursive step and raise `Prompt token budget exceeded: 28,887 used, limit 24,576`.

    We therefore reserve ~35-40% of max_prompt_tokens (or at least 6,000-12,000
    tokens) for fast-rlm recursion headroom, ensuring that codebase chunks leave
    ample room for fast-rlm subagents to complete.
    """
    max_prompt = None
    try:
        from api.config.settings import get_model_for_task
        max_prompt = (get_model_for_task("docgen") or {}).get("max_prompt_tokens")
    except Exception:  # pragma: no cover - settings store import-safe
        max_prompt = None
    if max_prompt is None and os.environ.get("RLM_MAX_PROMPT_TOKENS"):
        try:
            max_prompt = int(os.environ["RLM_MAX_PROMPT_TOKENS"])
        except ValueError:
            max_prompt = None

    context_window = _resolve_rlm_context_window() or 8192
    completion_reserve = max(1024, min(4096, context_window // 4))
    max_prompt_tokens = max(1024, context_window - completion_reserve)
    if max_prompt and max_prompt > 0:
        max_prompt_tokens = min(max_prompt_tokens, max_prompt)

    # Reserve headroom for fast-rlm's recursive subagent execution history (35-40%)
    recursion_reserve = max(3000, min(12000, int(max_prompt_tokens * 0.4)))
    budget = max_prompt_tokens - recursion_reserve
    return max(budget, 3000)


_CODEBASE_BLOCK_HEADER = (
    "\n\n<context_codebase>\n"
    "Ниже приведён исходный код проекта. Используй его как основной источник "
    "фактов при генерации раздела документации:\n"
)


# ---------------------------------------------------------------------------
# Codebase reading + lightweight analysis (feeds WikiSectionContext)
# ---------------------------------------------------------------------------
def _split_large_file_into_parts(path: str, text: str, max_tokens: int) -> List[str]:
    """Split a single large file into multi-part blocks with Part X of N headers.

    Preserves 100% of source lines without character truncation.
    """
    lines = text.splitlines(keepends=True)
    parts: List[List[str]] = []
    current_lines: List[str] = []
    base_header_tokens = _count_tokens(f"### File: {path} (Part 99 of 99)\n```\n\n```\n")
    current_tokens = base_header_tokens

    for line in lines:
        line_tokens = _count_tokens(line)
        if current_lines and current_tokens + line_tokens > max_tokens:
            parts.append(current_lines)
            current_lines = [line]
            current_tokens = base_header_tokens + line_tokens
        else:
            current_lines.append(line)
            current_tokens += line_tokens
    if current_lines:
        parts.append(current_lines)

    total_parts = len(parts)
    blocks: List[str] = []
    for i, part_lines in enumerate(parts, 1):
        part_text = "".join(part_lines)
        part_header = f" (Part {i} of {total_parts})" if total_parts > 1 else ""
        blocks.append(f"### File: {path}{part_header}\n```\n{part_text}\n```\n")
    return blocks


def _build_file_blocks(documents: List[Any], max_file_chunk_tokens: int = 8000) -> List[str]:
    """Build per-file code blocks without character truncation.

    If a single file exceeds max_file_chunk_tokens, it is split into multi-part
    file blocks with explicit (Part X of N) headers so zero code is lost.
    """
    blocks: List[str] = []
    for doc in documents:
        meta = getattr(doc, "meta_data", None) or {}
        path = meta.get("file_path", "unknown")
        text = getattr(doc, "text", "") or ""
        if not text or not text.strip():
            continue
        file_tokens = _count_tokens(text)
        if file_tokens > max_file_chunk_tokens:
            blocks.extend(_split_large_file_into_parts(path, text, max_file_chunk_tokens))
        else:
            blocks.append(f"### File: {path}\n```\n{text}\n```\n")
    return blocks


def _build_codebase_blob(documents: List[Any]) -> str:
    """Concatenate repo file contents into a single long-context string."""
    return "\n".join(_build_file_blocks(documents))


def _chunk_file_blocks(blocks: List[str], max_tokens: int) -> List[str]:
    """Group per-file blocks into chunks that each fit ``max_tokens``.

    Splits on block (file) boundaries -- a file is never split across two
    chunks -- so each chunk is a coherent, readable slice of the codebase.
    Each resulting chunk is a ``"\n"``-joined blob of one or more file blocks.

    A single block larger than the budget still becomes its own chunk (it was
    already capped at ``PER_FILE_MAX_CHARS`` upstream); we never drop content.
    Returns ``[]`` for empty input, and a single empty-string chunk is avoided
    (the caller treats a 1-element list as the single-call path).
    """
    if not blocks:
        return []
    if max_tokens <= 0:
        return ["\n".join(blocks)]
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    # Account for the "\n" separators that join() will insert between blocks.
    for block in blocks:
        block_tokens = _count_tokens(block)
        sep_tokens = 1 if current else 0  # one "\n" between blocks
        # If the block alone exceeds the budget, flush what we have then emit
        # the oversize block as its own chunk (never split mid-file).
        if block_tokens > max_tokens:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_tokens = 0
            chunks.append(block)
            continue
        if current and current_tokens + sep_tokens + block_tokens > max_tokens:
            chunks.append("\n".join(current))
            current = [block]
            current_tokens = block_tokens
        else:
            current.append(block)
            current_tokens += sep_tokens + block_tokens
    if current:
        chunks.append("\n".join(current))
    return chunks


def _build_file_tree(paths: List[str], max_lines: int = 200) -> str:
    clean = sorted({p for p in paths if p})
    if len(clean) > max_lines:
        clean = clean[:max_lines]
    return "\n".join(clean)


_LANG_MAP = {
    "py": "Python", "js": "JavaScript", "ts": "TypeScript", "tsx": "TypeScript",
    "jsx": "JavaScript", "java": "Java", "go": "Go", "rs": "Rust", "cs": "C#",
    "rb": "Ruby", "php": "PHP", "kt": "Kotlin", "swift": "Swift", "c": "C",
    "cpp": "C++", "h": "C/C++ header",
}

_CONFIG_BASENAMES = {
    "package.json", "requirements.txt", "pyproject.toml", "cargo.toml", "go.mod",
    "pom.xml", "build.gradle", "docker-compose.yml", "docker-compose.yaml",
    "dockerfile", "tsconfig.json", "vite.config.js", "next.config.js",
    ".env.example",
}


def _build_file_analysis(documents: List[Any]) -> Dict[str, Any]:
    """Build a lightweight file_analysis dict for ``create_wiki_section_context``.

    Mirrors the keys consumed by ``api.docgen.wiki.create_wiki_section_context``
    (main_directories, main_files, tech_stack, config_files, cicd_files,
    docker_files, api_endpoints, databases, entities, modules, primary_language,
    file_count). Statically-undetectable fields (api_endpoints, databases,
    entities) are left empty -- the codebase blob / file tree is the source of
    truth for RLM, and the section prompts handle empty values gracefully.
    """
    paths = [(getattr(d, "meta_data", None) or {}).get("file_path", "") for d in documents]
    paths = [p for p in paths if p]

    main_directories: List[str] = []
    for p in paths:
        parts = p.split(os.sep)
        if len(parts) > 1 and parts[0] not in main_directories:
            main_directories.append(parts[0])

    main_files = [os.path.basename(p) for p in paths][:30]

    ext_counts = Counter(os.path.splitext(p)[1].lower().lstrip(".") for p in paths if p)
    primary_language = "unknown"
    if ext_counts:
        top_ext = ext_counts.most_common(1)[0][0]
        primary_language = _LANG_MAP.get(top_ext, top_ext or "unknown")

    config_files = [
        p for p in paths
        if os.path.basename(p).lower() in _CONFIG_BASENAMES
        or p.endswith((".toml", ".cfg", ".ini", ".conf"))
    ][:15]
    cicd_files = [
        p for p in paths
        if any(x in p.lower() for x in (
            ".github/", ".gitlab-ci", "jenkinsfile", "azure-pipelines",
        ))
    ][:10]
    docker_files = [
        p for p in paths
        if "dockerfile" in os.path.basename(p).lower()
        or "docker-compose" in os.path.basename(p).lower()
    ][:10]

    modules = list(dict.fromkeys(main_directories))[:15]

    return {
        "main_directories": main_directories[:15],
        "main_files": main_files,
        "tech_stack": {},
        "config_files": config_files,
        "cicd_files": cicd_files,
        "docker_files": docker_files,
        "api_endpoints": [],
        "databases": [],
        "entities": [],
        "modules": modules,
        "primary_language": primary_language,
        "file_count": len(documents),
    }


def _read_readme(repo_dir: str) -> str:
    for name in ("README.md", "README.rst", "README.txt", "README", "readme.md"):
        p = os.path.join(repo_dir, name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    return ""


# ---------------------------------------------------------------------------
# Section generation (RLM with standard-LLM fallback)
# ---------------------------------------------------------------------------
def _resolve_use_rlm(use_rlm: Optional[bool], blob_len: int) -> bool:
    """Resolve the final RLM flag for codebase docgen from a mode setting.

    Mirrors ``api.expert.generate._resolve_use_rlm`` but for the ``docgen`` task.
    Precedence: explicit True/False wins; otherwise follow the admin
    ``rlm.docgen.mode`` setting (auto/rlm/llm). ``auto`` enables RLM only for
    large contexts (>= RLM_MIN_CHARS). ``get_rlm_mode`` returns ``llm`` when
    fast-rlm is not installed, so RLM never accidentally runs when unavailable.
    """
    if use_rlm is True:
        return True
    if use_rlm is False:
        return False
    try:
        from api.config.settings import get_rlm_mode
        mode = get_rlm_mode("docgen")
    except Exception:  # pragma: no cover - settings store import-safe
        mode = "auto"
    if mode == "llm":
        return False
    if mode == "rlm":
        return True
    return blob_len >= RLM_MIN_CHARS


async def _attempt_rlm(
    query: str,
    rlm_model: Optional[str],
    rlm_state: Optional[Dict[str, int]],
    product_id: Optional[str] = None,
    codebase_id: Optional[str] = None,
    file_tree: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> Optional[str]:
    """Run one fast-rlm call. Returns cleaned text on success, ``None`` otherwise.

    Shared by the single-call section path and the per-chunk map step. Honors
    the per-run circuit breaker in ``rlm_state``: once ``RLM_MAX_FAILURES`` is
    reached this run, returns ``None`` immediately without invoking RLM (so the
    remaining sections/chunks skip RLM and use the standard LLM instead of
    wasting ~3 min/call on a model that consistently fails in the Pyodide REPL).
    On any non-success path a failure is recorded into ``rlm_state`` and the
    breaker trip is logged.

    Two RLM paths:
    - **Structured path** (when ``codebase_id`` is supplied): calls
      ``run_rlm_structured`` with a dict query and the docgen retrieval tools.
      The agent reads repo files ON DEMAND via ``read_codebase_file`` /
      ``search_code`` so it can cover 100% of the repo regardless of size
      (instead of a pre-chunked blob in the prompt). A per-codebase session
      persists the agent's file index across the 7 sections (~2.6× cheaper per
      the docs: the first section builds the index, later sections reuse it).
      Falls back to the flat-string path on any structured failure.
    - **Flat-string path** (no ``codebase_id``, or structured fallback): the
      legacy ``run_rlm_task(query)`` with the codebase chunk already in the query.
    """
    rlm_failures = (rlm_state or {}).get("failures", 0)
    if rlm_failures >= RLM_MAX_FAILURES:
        return None
    # Resolve the per-section RLM timeout at call time (admin > env > default)
    # so an admin save takes effect without a restart.
    try:
        from api.config.timeout import resolve_rlm_section_timeout
        rlm_section_timeout = resolve_rlm_section_timeout()
    except Exception:
        rlm_section_timeout = 1800.0
    try:
        ctx_win = _resolve_rlm_context_window() or 8192
        completion_res = max(1024, min(4096, ctx_win // 4))
        max_prompt_limit = max(1024, ctx_win - completion_res)
        safe_query_limit = max(2000, max_prompt_limit - 6000)
        # Char-cap the query to the token budget (4 chars/token heuristic) so a
        # single RLM call cannot blow max_prompt_tokens. The codebase is already
        # split into per-call chunks upstream (_chunk_file_blocks), so this only
        # fires for a large section prompt + chunk combination.
        safe_query_chars = safe_query_limit * 4
        safe_query = query if len(query) <= safe_query_chars else (
            query[:safe_query_chars] + "\n... (контекст обрезан для RLM)"
        )

        from api.rlm.runner import (  # lazy: fast_rlm is optional
            run_rlm_structured,
            run_rlm_task,
            get_rlm_session_dir,
        )

        # Structured path: dict query + docgen tools + per-codebase session.
        # The agent reads files on demand, so the dict carries the task + repo
        # name + file tree (NOT the codebase blob). The first section builds the
        # file index in the session; later sections reuse it.
        if codebase_id:
            try:
                from api.rlm.tools import build_docgen_tools, resolve_env_variables

                dict_query = {
                    "task": query,
                    "repo": repo_name or codebase_id,
                    "file_tree": file_tree or "",
                }
                tools = build_docgen_tools()
                env_vars = resolve_env_variables(product_id or "", codebase_id)
                session_dir = get_rlm_session_dir(f"docgen_{codebase_id}")
                res = await asyncio.wait_for(
                    run_rlm_structured(
                        dict_query,
                        model_name=rlm_model,
                        tools=tools,
                        env_variables=env_vars,
                        session_dir=session_dir,
                        session_id=codebase_id,
                        task="docgen",
                    ),
                    timeout=rlm_section_timeout,
                )
                if res.get("success") and res.get("results"):
                    txt = _clean_llm_text(str(res["results"]))
                    if txt:
                        return txt
                # Structured did not succeed -> fall through to flat-string path.
                if not res.get("success"):
                    logger.info(
                        "RLM structured docgen path did not succeed (%s); "
                        "trying flat-string path.",
                        str(res.get("results"))[:120],
                    )
            except Exception as e:  # pragma: no cover - depends on live fast-rlm
                logger.warning(
                    "RLM structured docgen path failed (%s); falling back to flat-string.",
                    e,
                )

        # Flat-string path (legacy, or structured fallback).
        res = await asyncio.wait_for(
            run_rlm_task(safe_query, rlm_model), timeout=rlm_section_timeout
        )
        if res.get("success") and res.get("results"):
            txt = _clean_llm_text(str(res["results"]))
            if txt:
                return txt
    except asyncio.TimeoutError:
        logger.warning("RLM timed out after %ss.", rlm_section_timeout)
    except Exception as e:  # pragma: no cover - depends on live fast-rlm
        logger.warning("RLM generation failed: %s", e)
    # Non-success path (no usable result / timeout / exception): record it.
    if rlm_state is not None:
        rlm_state["failures"] = rlm_failures + 1
        if rlm_state["failures"] >= RLM_MAX_FAILURES:
            logger.info(
                "RLM failed %d time(s); skipping RLM for remaining sections/chunks "
                "(standard LLM will be used directly).",
                rlm_state["failures"],
            )
    return None


async def _generate_section_text(
    section_prompt: str,
    codebase_chunks: List[str],
    use_rlm: bool,
    llm: Optional[_StandardLLM],
    rlm_model: Optional[str],
    rlm_state: Optional[Dict[str, int]] = None,
    product_id: Optional[str] = None,
    codebase_id: Optional[str] = None,
    file_tree: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> str:
    """Generate a single section: RLM (long context) -> standard LLM fallback.

    Two RLM modes (selected automatically by ``codebase_id``):
    - **Structured** (``codebase_id`` supplied): the agent reads repo files ON
      DEMAND via the docgen tools, so it can cover 100% of the repo regardless
      of size. The per-codebase session persists the file index across the 7
      sections. Chunks are NOT used here — the agent fetches files itself.
    - **Flat-string / map-reduce** (no ``codebase_id``): ``codebase_chunks``
      is the codebase split into token-budget-sized pieces
      (see ``_chunk_file_blocks``). A single chunk takes the single-call path;
      multiple chunks take a map-reduce path:
        - MAP: one RLM draft per chunk. The shared ``rlm_state`` breaker
          short-circuits RLM once ``RLM_MAX_FAILURES`` is hit this run.
        - REDUCE: the standard LLM synthesizes the per-chunk drafts.

    When ``use_rlm`` is False (small codebase, or admin forced LLM) the caller
    passes a single chunk and the standard LLM is used directly, with the
    codebase blob appended only if it is small enough (unchanged behaviour).
    """
    chunks = [c for c in (codebase_chunks or []) if c]
    rlm_failures = (rlm_state or {}).get("failures", 0)
    try_rlm = use_rlm and rlm_failures < RLM_MAX_FAILURES

    # Structured RLM path: the agent reads files on demand, so chunks are
    # irrelevant. One call per section (the session carries state across
    # sections). Takes priority over the flat-string map-reduce path.
    if try_rlm and codebase_id:
        txt = await _attempt_rlm(
            section_prompt, rlm_model, rlm_state,
            product_id=product_id,
            codebase_id=codebase_id,
            file_tree=file_tree,
            repo_name=repo_name,
        )
        if txt:
            return txt
        logger.warning(
            "RLM structured path returned no usable result; falling back to "
            "flat-string / standard LLM for this section."
        )
        # Fall through to the flat-string / standard-LLM path below.

    # Flat-string map-reduce path (no codebase_id, or structured fallback).
    # Multi-chunk => map-reduce over RLM. Chunks are only produced for
    # large-codebase RLM runs, so ``use_rlm`` is True here.
    if use_rlm and len(chunks) > 1:
        return await _generate_section_mapreduce(
            section_prompt, chunks, llm, rlm_model, rlm_state,
            product_id=product_id,
            codebase_id=codebase_id,
            file_tree=file_tree,
            repo_name=repo_name,
        )

    # Single-chunk (or no-chunk / use_rlm False) => original single-call path.
    codebase_blob = chunks[0] if chunks else ""

    if try_rlm:
        query = section_prompt + _CODEBASE_BLOCK_HEADER + codebase_blob
        txt = await _attempt_rlm(query, rlm_model, rlm_state)
        if txt:
            return txt
        logger.warning(
            "RLM returned no usable result; falling back to standard LLM "
            "for this section."
        )

    if llm is not None:
        try:
            from api.utils.llm_helpers import cap as _cap_text
            ctx_win = _resolve_rlm_context_window() or 8192
            max_p_tokens = max(1024, ctx_win - 2048)

            # ALWAYS attach codebase_blob when available (capped to fit model
            # context) so the standard LLM fallback has actual source code to
            # generate the section from, even if RLM failed or was skipped.
            prompt = section_prompt
            if codebase_blob:
                prompt = prompt + _CODEBASE_BLOCK_HEADER + codebase_blob

            # Char-cap to the token budget (4 chars/token heuristic).
            prompt = _cap_text(prompt, max_p_tokens * 4)
            txt = _clean_llm_text(await llm.generate(prompt))
            if txt:
                return txt
        except Exception as e:  # pragma: no cover - depends on live LLM
            logger.warning("Standard LLM section generation failed: %s", e)

    return ""


async def _generate_section_mapreduce(
    section_prompt: str,
    chunks: List[str],
    llm: Optional[_StandardLLM],
    rlm_model: Optional[str],
    rlm_state: Optional[Dict[str, int]] = None,
    product_id: Optional[str] = None,
    codebase_id: Optional[str] = None,
    file_tree: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> str:
    """Map-reduce a section over multiple codebase chunks.

    MAP: one RLM draft per chunk (each chunk already fits the per-call token
    budget, so no single RLM call can blow ``max_prompt_tokens``). The shared
    ``rlm_state`` breaker skips RLM for the remaining chunks once the per-run
    failure budget is exhausted.

    REDUCE: the standard LLM merges the drafts into one section. The drafts are
    small (a section's worth each), so the reduce input always fits the standard
    LLM context. If reduce itself fails, the concatenated drafts are returned
    (still useful content). If every chunk produced no draft (RLM failed on all /
    breaker tripped immediately), degrade like the single-call RLM-failure path:
    standard LLM with the section prompt only.
    """
    drafts: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        query = section_prompt + _CODEBASE_BLOCK_HEADER + chunk
        draft = await _attempt_rlm(
            query, rlm_model, rlm_state,
            product_id=product_id,
            codebase_id=codebase_id,
            file_tree=file_tree,
            repo_name=repo_name,
        )
        if draft:
            drafts.append(draft)
        else:
            logger.info(
                "Section map: chunk %d/%d produced no draft (RLM skipped/failed).",
                i, len(chunks),
            )

    if drafts:
        merged = await _reduce_section_drafts(section_prompt, drafts, llm)
        if merged:
            return merged
        # Reduce failed: concatenated drafts are still useful section content.
        logger.warning("Section reduce produced no text; returning concatenated drafts.")
        return "\n\n".join(drafts)

    # No drafts at all: use Agentic Bottom-Up Synthesis Engine to analyze 100% of files
    logger.warning(
        "Section map produced no RLM drafts; using Agentic Bottom-Up Synthesis Engine for standard LLM."
    )
    if llm is not None:
        try:
            txt = await _agentic_bottom_up_docgen(section_prompt, chunks, llm)
            if txt:
                return txt
        except Exception as e:  # pragma: no cover - depends on live LLM
            logger.warning("Agentic bottom-up section generation failed: %s", e)
    return ""


async def _reduce_section_drafts(
    section_prompt: str,
    drafts: List[str],
    llm: Optional[_StandardLLM],
) -> str:
    """Merge per-chunk section drafts into one coherent section via the LLM.

    Used by the map-reduce path (``_generate_section_mapreduce``) and the
    agentic bottom-up engine (``_agentic_bottom_up_docgen``). The drafts are
    each a section's worth of text (small), so the merged input fits the
    standard LLM context. Returns cleaned text, or "" on LLM failure / empty
    input (the caller falls back to concatenated drafts).
    """
    if not drafts:
        return ""
    if llm is None:
        return "\n\n".join(drafts)
    if len(drafts) == 1:
        return drafts[0]
    from api.utils.llm_helpers import cap as _cap_text
    ctx_win = _resolve_rlm_context_window() or 8192
    max_p_tokens = max(1024, ctx_win - 2048)
    joined = "\n\n---\n\n".join(drafts)
    # Char-cap the joined drafts to the token budget (4 chars/token heuristic).
    joined = _cap_text(joined, max_p_tokens * 4)
    reduce_prompt = (
        "Ниже представлены частичные черновики одного раздела документации,\n"
        "полученные из разных частей кодовой базы. Объедини их в один\n"
        "согласованный, непротиворечивый раздел на русском языке (технические\n"
        "термины на английском). Убери дублирование, сохрани все технические\n"
        "факты и имена. Не добавляй новых фактов, которых нет в черновиках.\n\n"
        f"<section_instruction>\n{section_prompt}\n</section_instruction>\n\n"
        f"<drafts>\n{joined}\n</drafts>\n\n"
        "Готовый раздел:"
    )
    try:
        txt = _clean_llm_text(await llm.generate(_with_verification_guard(reduce_prompt)))
        if txt:
            return txt
    except Exception as e:  # pragma: no cover - depends on live LLM
        logger.warning("Section reduce LLM call failed: %s", e)
    return ""


async def _agentic_file_map_summary(
    block_chunk: str,
    llm: Optional[_StandardLLM],
    max_tokens: int,
) -> str:
    """Phase 1: Extract structured technical facts from a codebase block chunk."""
    if llm is None or not block_chunk:
        return ""
    prompt = (
        "Ты технический AI-агент. Проанализируй исходные файлы кодовой базы ниже "
        "и извлеки краткую, но исчерпывающую техническую сводку:\n"
        "1. Архитектурную роль и назначение каждого файла/модуля.\n"
        "2. Экспортируемые классы, интерфейсы, функции и их сигнатуры.\n"
        "3. API эндпоинты, методы, структуры запросов/ответов.\n"
        "4. Модели данных, базы данных, сущности и их поля.\n"
        "5. Зависимости, конфигурацию и CI/CD компоненты.\n"
        "Будь предельно точен, не упускай технические детали, сохрани все имена файлов, "
        "классов и методов.\n\n"
        f"<codebase_chunk>\n{block_chunk}\n</codebase_chunk>\n\nAssistant:"
    )
    from api.utils.llm_helpers import cap as _cap_text
    prompt = _with_verification_guard(_cap_text(prompt, max_tokens * 4))
    try:
        return _clean_llm_text(await llm.generate(prompt))
    except Exception as e:
        logger.warning("Agentic file map summary call failed: %s", e)
        return ""


async def _agentic_bottom_up_docgen(
    section_prompt: str,
    chunks: List[str],
    llm: Optional[_StandardLLM],
) -> str:
    """Agentic Bottom-Up Harness Engine for Standard LLM Fallback.

    Phase 1 (Map): Summarizes technical facts for 100% of files across chunks.
    Phase 2 (Reduce/Synthesize): Merges file summaries into the final section.
    Ensures 0% code loss on large codebases.
    """
    if llm is None or not chunks:
        return ""

    ctx_win = _resolve_rlm_context_window() or 8192
    max_p_tokens = max(1024, ctx_win - 2048)

    # Phase 1: Map all file chunks to technical file summaries
    file_summaries: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        summary = await _agentic_file_map_summary(chunk, llm, max_p_tokens)
        if summary:
            file_summaries.append(f"### Сводка файлов (часть {i}/{len(chunks)}):\n{summary}")

    if not file_summaries:
        # Fallback to direct prompt if map produced nothing
        combined_blob = "\n\n".join(chunks)
        from api.utils.llm_helpers import cap as _cap_text
        prompt = _cap_text(section_prompt + _CODEBASE_BLOCK_HEADER + combined_blob, max_p_tokens * 4)
        try:
            return _clean_llm_text(await llm.generate(prompt))
        except Exception:
            return ""

    # Phase 2: Synthesize the section from all file summaries
    return await _reduce_section_drafts(section_prompt, file_summaries, llm)


# ---------------------------------------------------------------------------
# Persistence + section helpers
# ---------------------------------------------------------------------------
def _section_pages(sections: Dict[str, str], language: str) -> Dict[str, Any]:
    pages: Dict[str, Any] = {}
    for section_type in WikiGenerator.SECTION_ORDER:
        sid = section_type.value
        page_id = f"page_{sid}"
        pages[page_id] = {
            "id": page_id,
            "title": get_section_title(sid, language),
            "content": sections.get(sid, ""),
            "filePaths": [],
            "importance": "high",
            "relatedPages": [],
        }
    return pages


def _raise_if_all_sections_unavailable(sections: Dict[str, str]) -> None:
    """Raise ValueError when EVERY section is the unavailable placeholder.

    Real generation produced no usable content in this case (the LLM/RLM were
    unreachable or rate-limited for the whole run). Committing placeholder-only
    docs as a "succeeded" job would make the UI show
    "Содержимое раздела временно недоступно" on every page while claiming
    success. Surfacing a genuine failure lets the user retry instead. Display
    being decoupled from cognee does NOT mean a total generation failure should
    be masked as a successful (empty) doc set.
    """
    if sections and all(
        (v or "").strip() == _SECTION_UNAVAILABLE_PLACEHOLDER for v in sections.values()
    ):
        raise ValueError(
            "Не удалось сгенерировать ни один раздел документации (LLM/RLM "
            "недоступны или превысили таймаут). Проверьте подключение к модели "
            "и перезапустите генерацию."
        )


# ---------------------------------------------------------------------------
# Codebase documentation (RLM long-context + standard-LLM fallback)
# ---------------------------------------------------------------------------
async def generate_codebase_docs(
    artifact: Any,
    product: Any,
    model: Optional[str] = None,
    language: str = "ru",
) -> str:
    """Generate the 7 wiki sections for a codebase artifact and return markdown.

    The repo is cloned via ``DatabaseManager._create_repo`` (reuses the same
    path scheme as the rest of DeepWiki). The codebase is read into a
    long-context blob; RLM generates each section sequentially using the
    ``refs/prompts/<section>.md`` template (variable mapping reused from
    ``WikiGenerator._format_prompt``). Falls back to the standard LLM when the
    codebase is small or RLM is unavailable. The repo is indexed into cognee
    in the background after generation.
    """
    from api.repositories.documents import DatabaseManager, read_all_documents  # lazy

    repo_url = (getattr(artifact, "repo_url", "") or "").strip()
    if not repo_url:
        raise ValueError("Codebase artifact has no repo_url; cannot generate docs.")
    repo_type = getattr(artifact, "repo_type", None) or "github"
    token = getattr(artifact, "token", None)
    # Server-side credential resolution: the per-artifact token is no longer
    # entered in the UI, so resolve the token from admin-configured git
    # accounts by matching the repo URL host (public github.com / self-hosted).
    try:
        from api.config.settings import resolve_git_token
        resolved_token = resolve_git_token(repo_url, repo_type)
        if resolved_token:
            token = resolved_token
    except Exception as e:  # pragma: no cover - import-safe
        logger.debug("resolve_git_token failed for %s: %s", repo_url, e)

    # Resolve the docgen LLM config from the admin store (models.docgen.*) so
    # codebase docgen reaches the corporate AI gateway when configured, instead
    # of the dead env-default LM Studio :1234. Per-request model override wins.
    resolved_model, resolved_base_url, resolved_api_key = _resolve_docgen_model(model)
    model = model or resolved_model

    db_manager = DatabaseManager()
    # force_refresh=True so every (re)generation fetches the latest remote tip
    # instead of reusing the stale first clone — otherwise regenerating an
    # already-documented codebase re-reads the original checkout and the UI
    # shows unchanged ("old") docs even though the job reported success.
    db_manager._create_repo(repo_url, repo_type, token, force_refresh=True)
    repo_dir = (db_manager.repo_paths or {}).get("save_repo_dir")
    if not repo_dir or not os.path.isdir(repo_dir):
        raise ValueError(f"Repository not available locally at {repo_dir!r} after clone.")

    documents = read_all_documents(repo_dir)
    if not documents:
        raise ValueError("No readable source files found in the repository.")

    codebase_blob = _build_codebase_blob(documents)
    file_analysis = _build_file_analysis(documents)
    file_tree = _build_file_tree(
        [(getattr(d, "meta_data", None) or {}).get("file_path", "") for d in documents]
    )
    readme = _read_readme(repo_dir)

    # Enrich docgen context with product-level knowledge (Confluence pages / specs)
    pid = getattr(product, "id", None) or getattr(product, "product_id", None)
    if pid:
        try:
            from api.expert.knowledge import _retrieve_product_knowledge
            p_knowledge = await _retrieve_product_knowledge(pid, "architecture functional API specifications")
            if p_knowledge and p_knowledge.strip():
                from api.utils.llm_helpers import cap as _cap_text
                clamped_kn = _cap_text(p_knowledge, 16000)  # ~4000 tokens
                readme = (readme or "") + f"\n\n### Дополнительный контекст продукта (Confluence / База знаний):\n{clamped_kn}\n"
        except Exception as e:
            logger.debug("Docgen product knowledge retrieval skipped for %s: %s", pid, e)

    # Decide whether the codebase fits a single RLM call. ``codebase_chunks`` is
    # used only on the RLM path; the standard-LLM path always uses the single
    # blob (capped by SMALL_CODEBASE_APPEND_LIMIT) as before.
    use_rlm = _resolve_use_rlm(None, len(codebase_blob))
    codebase_chunks: List[str] = [codebase_blob]
    chunk_budget = _resolve_codebase_chunk_budget()
    if use_rlm and codebase_blob:
        blocks = _build_file_blocks(documents)
        chunked = _chunk_file_blocks(blocks, chunk_budget)
        if len(chunked) > 1:
            codebase_chunks = chunked

    context = create_wiki_section_context(
        repo_url=repo_url,
        repo_type=repo_type,
        file_tree=file_tree,
        readme=readme,
        file_analysis=file_analysis,
    )

    # Reuse WikiGenerator for the per-section variable mapping (_format_prompt)
    # so this pipeline and the websocket/wiki path stay in sync. We drive the
    # loop ourselves because RLM is async and WikiGenerator.generate_section is
    # synchronous.
    gen = WikiGenerator(
        model=model,
        language=language,
    )
    gen.set_context(context)

    llm = _safe_build_llm(model, base_url=resolved_base_url, api_key=resolved_api_key)
    rlm_model = model  # run_rlm_task resolves the local model id itself
    # Shared mutable state across sections: once RLM fails RLM_MAX_FAILURES
    # times within this run, remaining sections skip RLM and go straight to
    # the standard LLM.
    rlm_state: Dict[str, int] = {"failures": 0}

    _resolved_ctx_window = _resolve_rlm_context_window()
    logger.info(
        "Codebase docgen: repo=%s files=%d blob_chars=%d use_rlm=%s chunks=%d "
        "chunk_budget_tokens=%d base_url=%s "
        "model_context_window=%s",
        repo_url, len(documents), len(codebase_blob), use_rlm, len(codebase_chunks),
        chunk_budget, (resolved_base_url or "<env default>"),
        _resolved_ctx_window or "unset (no clamp; relies on max_prompt_tokens)",
    )

    # codebase_id enables the structured RLM path (agent reads files on demand
    # via tools + per-codebase session). pid is the product id for the tools'
    # env_variables. repo_name is derived from the URL for the dict query.
    _codebase_id = getattr(artifact, "id", None)
    _repo_name = context.repo_name or _repo_name_from_url(repo_url)

    sections: Dict[str, str] = {}
    repair_llm = _make_repair_llm(
        model, llm, base_url=resolved_base_url, api_key=resolved_api_key
    )
    for section_type in gen.SECTION_ORDER:
        prompt = gen._format_prompt(section_type, context)
        content = await _generate_section_text(
            prompt, codebase_chunks, use_rlm, llm, rlm_model, rlm_state,
            product_id=pid,
            codebase_id=_codebase_id,
            file_tree=file_tree,
            repo_name=_repo_name,
        )
        if not content:
            content = _SECTION_UNAVAILABLE_PLACEHOLDER
        # Validate + repair any mermaid diagrams in this section before storing
        # it, so broken diagrams never reach the UI or ``previous_content``.
        # The repair loop is non-fatal: if the Node verifier is unavailable it
        # returns the content unchanged.
        try:
            content, _mstats = await run_repair_loop(content, repair_llm)
        except Exception as e:  # pragma: no cover - verifier must never break gen
            logger.warning("Mermaid repair loop failed for section %s: %s", section_type.value, e)
        # Feed this section back as previous_content for the next section.
        gen.generated_sections[section_type.value] = content
        sections[section_type.value] = content

    # If EVERY section came back as the unavailable placeholder, real generation
    # produced no usable content (LLM/RLM were unreachable or rate-limited for
    # the whole run). Committing placeholder-only docs as "succeeded" would make
    # the UI show "Содержимое раздела временно недоступно" on every page while
    # claiming success. Surface it as a genuine failure instead so the user can
    # retry — display is decoupled from cognee, but a total generation failure
    # must NOT be masked as a successful (empty) doc set.
    _raise_if_all_sections_unavailable(sections)

    repo_name = context.repo_name or _repo_name_from_url(repo_url)
    markdown = f"# Документация по кодовой базе: {repo_name}\n\n"
    for section_type in gen.SECTION_ORDER:
        sid = section_type.value
        title = get_section_title(sid, language)
        markdown += f"## {title}\n\n{sections[sid]}\n\n---\n\n"

    pages = _section_pages(sections, language)
    _persist_artifact(artifact, markdown, pages)
    # Index the cloned repo into the active memory backend AFTER generation
    # (non-blocking) into the product-scoped dataset. source_id = codebase id
    # so the pgvector upsert can delete the previous chunks for this codebase
    # before re-inserting (cognee ignores source_id).
    _index_in_background(
        repo_dir, _cognee_dataset(product),
        source_type="codebase", source_id=getattr(artifact, "id", None),
    )
    return markdown
