"""Artifact documentation-generation pipeline (Phase B).

Generates human-readable documentation for a Product's Artifacts and persists
the result onto ``artifact.generated_docs`` + a structured ``artifact.pages``:

- ``codebase``  -> 7 sequential wiki sections. The repository is cloned via
  ``api.data_pipeline.DatabaseManager._create_repo``, its files are read into a
  long-context blob, and each section is generated from the matching
  ``refs/prompts/<section>.md`` template (variable mapping reused from
  ``api.wiki_generator.WikiGenerator._format_prompt`` so the two paths stay
  consistent). RLM (fast-rlm) is used for long context; the standard LLM
  (adalflow, local Ollama) is used when the codebase is small or RLM fails.
- ``openapi``   -> structured stdlib render + standard-LLM enrichment via
  ``refs/prompts/openapi_doc.md``.
- ``asyncapi``  -> structured stdlib render + standard-LLM enrichment via
  ``refs/prompts/asyncapi_doc.md``.
- ``testcase``  -> standard-LLM doc via ``refs/prompts/testcase_doc.md``; an
  Allure URL is rendered as a LINK only (never fetched).

Design rules (from the Phase B plan / user decisions):
- Prompt substitution uses ``str.replace`` (NOT ``str.format``) so literal
  Mermaid/JSON braces in ``refs/prompts/*.md`` stay unescaped -- matches
  ``api/wiki_generator.py`` and ``api/websocket_wiki.py``.
- RLM (fast-rlm) is used ONLY for long-context codebase generation. OpenAPI /
  AsyncAPI / testcase and simple chat use the standard LLM (adalflow
  OllamaClient via ``api.config.get_model_config``).
- Cognee indexing is non-blocking: failures are logged, never fatal.
- No cloud API keys anywhere; everything points at local Ollama.

Resilience strategy: the heavy/optional dependencies (``cognee``,
``fast_rlm``, ``adalflow``, ``api.data_pipeline``) are imported LAZILY inside
the functions that need them and wrapped in try/except. That keeps
``import api.artifact_docgen`` cheap and lets every function fall back to a
deterministic result when a backend service is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# Optional YAML support. PyYAML is a transitive dep of adalflow/cognee and is
# declared explicitly in pyproject.toml; if it is ever missing we degrade
# gracefully to JSON-only parsing + a raw-text fallback.
try:  # pragma: no cover - import guard
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional at runtime
    yaml = None  # type: ignore

from api.logging_config import setup_logging
from api.mermaid_verifier import run_repair_loop
from api.models import LEGACY_ARTIFACT_TYPE_MAP
from api.prompts import get_section_title, load_prompt_file
from api.wiki_generator import (
    WikiGenerator,
    create_wiki_section_context,
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
# timeout lives in rlm_runner.py (RLM_API_TIMEOUT_MS, default 30 min). When this
# section timeout fires, only the awaited result is discarded; the underlying
# RLM worker thread keeps running. Tunable via env.
RLM_SECTION_TIMEOUT = float(os.environ.get("RLM_SECTION_TIMEOUT", "1200"))
# Max RLM failures within a single generate run before skipping RLM for the
# remaining sections. fast-rlm runs the model inside a Pyodide Python REPL;
# after this many failures we stop trying RLM and go straight to the standard
# LLM for the rest of the run. Default 2 failures.
RLM_MAX_FAILURES = int(os.environ.get("RLM_MAX_FAILURES", "2"))
# fast-rlm's default ``max_prompt_tokens`` (200000) is the hard ceiling a single
# RLM call must stay under (the whole codebase blob + fast-rlm's recursive
# subagent outputs count toward it). On large repos the full blob alone can
# approach that limit, leaving no room for the recursive accumulation -- which
# surfaces as ``Prompt token budget exceeded``. To keep each RLM call safe we
# reserve this many tokens (for the section prompt + fast-rlm recursion) and
# split the codebase into per-call chunks that each fit in what remains.
# Tunable via env. See ``_resolve_codebase_chunk_budget``.
RLM_PROMPT_RESERVE_TOKENS = int(os.environ.get("RLM_PROMPT_RESERVE_TOKENS", "40000"))
# The MODEL'S ACTUAL context window (``num_ctx``) is what caps a single LLM
# call -- NOT fast-rlm's ``max_prompt_tokens`` (that is just a budget fast-rlm
# thinks it can use). A local Ollama model's effective ``num_ctx`` defaults to
# 2048/4096/8192/32768 depending on the model, which is FAR below fast-rlm's
# 200k default. Sending a 160k-token chunk (200k - 40k reserve) to an 8k-window
# model overflows it and the gateway returns
# ``litellm.BadRequestError: ... Context size has been exceeded``.
#
# Resolve the real ceiling here: ``RLM_MODEL_CONTEXT_WINDOW`` >
# ``OLLAMA_NUM_CTX`` > fast-rlm ``max_prompt_tokens``. When the context window
# is smaller than max_prompt_tokens we also scale the reserve down so the
# effective per-chunk budget stays proportional (a 40k reserve makes no sense
# on an 8k window). See ``_resolve_codebase_chunk_budget``.
def _resolve_rlm_context_window() -> Optional[int]:
    """Resolve the model's actual context-window ceiling in tokens.

    Uses ``get_model_context_window(task="docgen")`` which checks explicit env
    vars, admin settings, live API metadata (/api/show or /v1/models), model name
    heuristics, and a safe default (8192).
    """
    try:
        from api.model_utils import get_model_context_window
        return get_model_context_window(task="docgen")
    except Exception as e:
        logger.debug("Could not resolve context window in artifact_docgen: %s", e)
        return 8192
# Token-counting is approximate by design: we need a budget estimate, not an
# exact count. tiktoken cl100k_base (the Ollama path in data_pipeline.count_tokens)
# is a good fit for the local models used here; if tiktoken is unavailable we
# fall back to a len//4 character ratio so chunking still works (just less
# precise). The estimate is intentionally conservative (chunks end up slightly
# smaller than the budget, which is the safe direction).
_TIKTOKEN_ENC = None


def _count_tokens(text: str) -> int:
    """Approximate token count for a chunk-budget estimate.

    Uses tiktoken ``cl100k_base`` (matches the Ollama path in
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
        from api.settings_store import get_model_for_task
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
# Prompt substitution (str.replace -- NOT str.format -- so Mermaid/JSON braces
# in refs/prompts/*.md stay unescaped; matches api/wiki_generator.py).
# ---------------------------------------------------------------------------
def _safe_replace(template: str, variables: Dict[str, Any]) -> str:
    """Substitute ``{var}`` placeholders in ``template`` using exact replacement.

    Unmatched placeholders are left intact (so they remain visible rather than
    silently disappearing), matching the behaviour of
    ``WikiGenerator._format_prompt``.
    """
    if not template:
        return ""
    out = template
    for key, value in variables.items():
        out = out.replace("{" + key + "}", "" if value is None else str(value))
    return out


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


# Regex for a leading line-number prefix on a code line: optional spaces, then
# digits, then an optional separator (spaces, '.', ':' or a tab), then the rest.
# Matches "1 import os", "  12. def f():", "3:  x = 1", "10\t# comment".
_LINE_NUM_PREFIX_RE = re.compile(r"^[ \t]*\d+[ \t]*[:.]?[ \t]+")
# A line that is ONLY a number (optionally with a separator/trailing spaces) —
# i.e. a line number for a blank code line, e.g. "2", "2  ", "3.". Stripped only
# inside a confirmed numbered block (see _strip_number_prefixes_from_block) so
# standalone numeric-literal blocks are not mangled.
_LINE_NUM_ONLY_RE = re.compile(r"^[ \t]*\d+[ \t]*[:.]?[ \t]*$")


def _strip_inline_line_numbers(text: Optional[str]) -> str:
    """Strip leading ``N``/``N.``/``N:`` prefixes from lines INSIDE fenced code
    blocks only.

    The UI's ``SyntaxHighlighter`` already renders line numbers via
    ``showLineNumbers``; when an LLM ALSO emits ``1 import os`` prefixes the
    numbers are duplicated/ugly. This post-processor removes them as a safety
    net on top of the prompt rule (which asks the model not to emit them).

    Only fenced code blocks (``` ... ```) are touched: prose, Mermaid diagrams
    (which are their own fenced lang) and already-clean code are left intact.
    A line is only de-numbered when its leading number equals the line's
    1-indexed position in the block (the signature of LLM-emitted line numbers
    that start at 1) and at least one sibling line shares that property, so
    legitimate code that happens to begin with a number (e.g. a numeric literal
    on the first line) is not mangled. Mermaid blocks are skipped explicitly
    (their content is not code).
    """
    if not text:
        return text or ""
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        # A fenced block opener: ```lang  (lang may be empty or ````mermaid````).
        if stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            is_mermaid = lang == "mermaid"
            out.append(line)
            i += 1
            # Collect the block body until the closing fence.
            block: List[str] = []
            while i < n and not lines[i].lstrip().startswith("```"):
                block.append(lines[i])
                i += 1
            if not is_mermaid and block:
                block = _strip_number_prefixes_from_block(block)
            out.extend(block)
            # The closing fence (if present) — append as-is.
            if i < n:
                out.append(lines[i])
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _strip_number_prefixes_from_block(block: List[str]) -> List[str]:
    """Remove leading line-number prefixes from a single code block's lines.

    A line is de-numbered only when its leading number equals its 1-indexed
    position in the block (``val == idx + 1``) -- the signature of LLM-emitted
    line numbers that start at 1. This single rule naturally protects:
      * numeric literals (``1000``/``2000``/``3000`` never equal their position),
      * out-of-order numbers (``5``/``3``/``1`` never equal ``1``/``2``/``3``),
      * large numbers (an excerpt numbered ``5``/``6``/``7`` is left intact -- a
        conservative trade-off since it is indistinguishable from numeric data).
    At least 2 position-matched lines are required before any stripping happens,
    so an isolated numbered line is left alone. Gaps are handled gracefully: each
    matched line is stripped independently, so an unnumbered blank line in the
    middle of a numbered block does not prevent the rest from being cleaned.
    """
    if not block:
        return block
    # Collect lines whose leading number equals their 1-indexed position.
    content_matches: List[int] = []  # idx of content-bearing matched lines
    bare_matches: List[int] = []     # idx of bare-number matched lines
    for idx, ln in enumerate(block):
        num_match = re.match(r"^[ \t]*(\d+)", ln)
        if not num_match:
            continue
        val = int(num_match.group(1))
        if val != idx + 1:
            continue
        if _LINE_NUM_ONLY_RE.match(ln):
            bare_matches.append(idx)
        elif _LINE_NUM_PREFIX_RE.match(ln):
            content_matches.append(idx)
    # Need at least 2 position-matched numbered lines to confirm a numbered block.
    if len(content_matches) + len(bare_matches) < 2:
        return block
    out = list(block)
    for idx in content_matches:
        out[idx] = _LINE_NUM_PREFIX_RE.sub("", out[idx], count=1)
    for idx in bare_matches:
        out[idx] = ""
    return out


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


def _cap(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (обрезано для контекста LLM)\n"


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
    ``api.settings_store.get_model_for_task("docgen")`` so docgen reaches the
    corporate AI gateway instead of falling back to a dead local env default.
    """

    def __init__(
        self,
        provider: str,
        model: Optional[str],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        import adalflow as adal
        from api.config import get_model_config

        if not provider:
            provider = os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local")
        if not model:
            model = os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b")
        generator_config = get_model_config(provider, model)
        model_client_class = generator_config["model_client"]
        # Thread admin base_url/api_key through to the client so docgen hits
        # the configured endpoint (e.g. corporate gateway) rather than the
        # dead env-default LM Studio :1234. Ollama takes host=; OpenAIClient
        # takes base_url=/api_key= (both already wire SSL verify via ssl_config).
        client_kwargs: Dict[str, Any] = {}
        if provider == "ollama":
            if base_url:
                client_kwargs["host"] = base_url
        elif provider in ("openai_local", "openai", "openai_compatible") or "openai" in str(provider):
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
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Resolve (provider, model, base_url, api_key) for the docgen task.

    Reads admin-configured ``models.docgen.*`` from the Config Abstraction Layer
    (with env fallbacks) so docgen hits the corporate AI gateway when configured.
    """
    try:
        from api.config_abstraction import get_task_config

        cfg = get_task_config("docgen") or {}
        provider = cfg.get("provider") or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local")
        resolved_model = model or cfg.get("model") or os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b")
        return provider, resolved_model, cfg.get("base_url"), cfg.get("api_key")
    except Exception as e:  # pragma: no cover - settings store is import-safe
        logger.debug("get_task_config(docgen) failed; using defaults: %s", e)
        return (
            os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local"),
            model or os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b"),
            None,
            None,
        )


def _safe_build_llm(
    provider: str,
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[_StandardLLM]:
    try:
        return _StandardLLM(provider, model, base_url=base_url, api_key=api_key)
    except Exception as e:  # pragma: no cover - depends on live config/Ollama
        logger.warning(
            "Could not initialise standard LLM (%s/%s): %s. "
            "Falling back to RLM/skeleton where possible.", provider, model, e,
        )
        return None


async def _llm_or_none(
    prompt: str,
    provider: str,
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Run the standard LLM on ``prompt``; return cleaned text or "" on failure."""
    if not prompt:
        return ""
    llm = _safe_build_llm(provider, model, base_url=base_url, api_key=api_key)
    if llm is None:
        return ""
    try:
        return _clean_llm_text(await llm.generate(prompt))
    except Exception as e:  # pragma: no cover - depends on live Ollama
        logger.warning("Standard LLM generation failed: %s", e)
        return ""


def _make_repair_llm(
    provider: str,
    model: Optional[str],
    existing: Optional[_StandardLLM] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional["object"]:
    """Build an async ``(prompt) -> str`` callable for the mermaid repair loop.

    Reuses an already-built ``_StandardLLM`` when available (so the codebase
    path doesn't construct a second client); otherwise builds one from the same
    provider/model/base_url/api_key. Returns None if no LLM could be built
    (repairs are then skipped and broken diagrams are surfaced with a marker).
    """
    if existing is not None:
        llm = existing
    else:
        llm = _safe_build_llm(provider, model, base_url=base_url, api_key=api_key)
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

    Mirrors the keys consumed by ``api.wiki_generator.create_wiki_section_context``
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

    Mirrors ``api.expert_agent._resolve_use_rlm`` but for the ``docgen`` task.
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
        from api.settings_store import get_rlm_mode
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
) -> Optional[str]:
    """Run one fast-rlm call. Returns cleaned text on success, ``None`` otherwise.

    Shared by the single-call section path and the per-chunk map step. Honors
    the per-run circuit breaker in ``rlm_state``: once ``RLM_MAX_FAILURES`` is
    reached this run, returns ``None`` immediately without invoking RLM (so the
    remaining sections/chunks skip RLM and use the standard LLM instead of
    wasting ~3 min/call on a model that consistently fails in the Pyodide REPL).
    On any non-success path a failure is recorded into ``rlm_state`` and the
    breaker trip is logged.
    """
    rlm_failures = (rlm_state or {}).get("failures", 0)
    if rlm_failures >= RLM_MAX_FAILURES:
        return None
    try:
        from api.model_utils import clamp_text_by_tokens
        ctx_win = _resolve_rlm_context_window() or 8192
        completion_res = max(1024, min(4096, ctx_win // 4))
        max_prompt_limit = max(1024, ctx_win - completion_res)
        safe_query_limit = max(2000, max_prompt_limit - 6000)
        safe_query = clamp_text_by_tokens(query, safe_query_limit)

        from api.rlm_runner import run_rlm_task  # lazy: fast_rlm is optional
        res = await asyncio.wait_for(
            run_rlm_task(safe_query, rlm_model), timeout=RLM_SECTION_TIMEOUT
        )
        if res.get("success") and res.get("results"):
            txt = _clean_llm_text(str(res["results"]))
            if txt:
                return txt
    except asyncio.TimeoutError:
        logger.warning("RLM timed out after %ss.", RLM_SECTION_TIMEOUT)
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
) -> str:
    """Generate a single section: RLM (long context) -> standard LLM fallback.

    ``codebase_chunks`` is the codebase split into token-budget-sized pieces
    (see ``_chunk_file_blocks``). A single chunk (or no chunk) takes the
    original single-call path; multiple chunks take a map-reduce path:

    - MAP: one RLM draft per chunk (RLM is the long-context engine). The shared
      ``rlm_state`` circuit breaker short-circuits RLM for the remaining chunks
      once ``RLM_MAX_FAILURES`` is hit this run.
    - REDUCE: the standard LLM synthesizes the per-chunk drafts into one
      coherent section. The drafts are small (a section's worth each), so the
      reduce input always fits the standard LLM context.

    When ``use_rlm`` is False (small codebase, or admin forced LLM) the caller
    passes a single chunk and the standard LLM is used directly, with the
    codebase blob appended only if it is small enough (unchanged behaviour).
    """
    chunks = [c for c in (codebase_chunks or []) if c]

    # Multi-chunk => map-reduce over RLM. Chunks are only produced for
    # large-codebase RLM runs, so ``use_rlm`` is True here.
    if use_rlm and len(chunks) > 1:
        return await _generate_section_mapreduce(
            section_prompt, chunks, llm, rlm_model, rlm_state
        )

    # Single-chunk (or no-chunk / use_rlm False) => original single-call path.
    codebase_blob = chunks[0] if chunks else ""
    rlm_failures = (rlm_state or {}).get("failures", 0)
    try_rlm = use_rlm and rlm_failures < RLM_MAX_FAILURES

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
            from api.model_utils import clamp_text_by_tokens
            ctx_win = _resolve_rlm_context_window() or 8192
            max_p_tokens = max(1024, ctx_win - 2048)

            # ALWAYS attach codebase_blob when available (clamped to fit model context)
            # so the standard LLM fallback has actual source code to generate the section from,
            # even if RLM failed or was skipped.
            prompt = section_prompt
            if codebase_blob:
                prompt = prompt + _CODEBASE_BLOCK_HEADER + codebase_blob
            
            prompt = clamp_text_by_tokens(prompt, max_p_tokens)
            txt = _clean_llm_text(await llm.generate(prompt))
            if txt:
                return txt
        except Exception as e:  # pragma: no cover - depends on live Ollama
            logger.warning("Standard LLM section generation failed: %s", e)

    return ""


async def _generate_section_mapreduce(
    section_prompt: str,
    chunks: List[str],
    llm: Optional[_StandardLLM],
    rlm_model: Optional[str],
    rlm_state: Optional[Dict[str, int]] = None,
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
        draft = await _attempt_rlm(query, rlm_model, rlm_state)
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
        except Exception as e:  # pragma: no cover - depends on live Ollama
            logger.warning("Agentic bottom-up section generation failed: %s", e)
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
    from api.model_utils import clamp_text_by_tokens
    prompt = _with_verification_guard(clamp_text_by_tokens(prompt, max_tokens))
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
        from api.model_utils import clamp_text_by_tokens
        prompt = clamp_text_by_tokens(section_prompt + _CODEBASE_BLOCK_HEADER + combined_blob, max_p_tokens)
        try:
            return _clean_llm_text(await llm.generate(prompt))
        except Exception:
            return ""

    # Phase 2: Synthesize the section from all file summaries
    return await _reduce_section_drafts(section_prompt, file_summaries, llm)


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
    """Fire-and-forget cognee indexing. Failures are logged, never fatal."""

    async def _run() -> None:
        try:
            from api.cognee_manager import add_and_index_document  # lazy: cognee optional
            await add_and_index_document(content_or_path, dataset_name=dataset_name)
        except Exception as e:  # pragma: no cover - depends on live cognee/DB
            logger.warning("Cognee indexing failed for %r: %s", dataset_name, e)

    try:
        asyncio.create_task(_run())
    except RuntimeError:
        # No running event loop -- best-effort skip.
        logger.warning(
            "No running event loop; skipping background cognee indexing for %r.",
            dataset_name,
        )


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


# ---------------------------------------------------------------------------
# Codebase documentation (RLM long-context + standard-LLM fallback)
# ---------------------------------------------------------------------------
async def generate_codebase_docs(
    artifact: Any,
    product: Any,
    provider: str = None,
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
    from api.data_pipeline import DatabaseManager, read_all_documents  # lazy

    repo_url = (getattr(artifact, "repo_url", "") or "").strip()
    if not repo_url:
        raise ValueError("Codebase artifact has no repo_url; cannot generate docs.")
    repo_type = getattr(artifact, "repo_type", None) or "github"
    token = getattr(artifact, "token", None)

    # Resolve the docgen LLM config from the admin store (models.docgen.*) so
    # codebase docgen reaches the corporate AI gateway when configured, instead
    # of the dead env-default LM Studio :1234. Per-request provider/model
    # overrides (when provided) win over the admin setting.
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key = _resolve_docgen_model(model)
    provider = provider or resolved_provider
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
            from api.expert_agent import _retrieve_product_knowledge
            p_knowledge = await _retrieve_product_knowledge(pid, "architecture functional API specifications")
            if p_knowledge and p_knowledge.strip():
                from api.model_utils import clamp_text_by_tokens
                clamped_kn = clamp_text_by_tokens(p_knowledge, 4000)
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
        provider=provider,
        model=model,
        language=language,
    )
    gen.set_context(context)

    llm = _safe_build_llm(provider, model, base_url=resolved_base_url, api_key=resolved_api_key)
    rlm_model = model  # run_rlm_task resolves local Ollama tags itself
    # Shared mutable state across sections: once RLM fails RLM_MAX_FAILURES
    # times within this run, remaining sections skip RLM and go straight to
    # the standard LLM.
    rlm_state: Dict[str, int] = {"failures": 0}

    _resolved_ctx_window = _resolve_rlm_context_window()
    logger.info(
        "Codebase docgen: repo=%s files=%d blob_chars=%d use_rlm=%s chunks=%d "
        "chunk_budget_tokens=%d provider=%s base_url=%s "
        "model_context_window=%s",
        repo_url, len(documents), len(codebase_blob), use_rlm, len(codebase_chunks),
        chunk_budget, provider, (resolved_base_url or "<env default>"),
        _resolved_ctx_window or "unset (no clamp; relies on max_prompt_tokens)",
    )

    sections: Dict[str, str] = {}
    repair_llm = _make_repair_llm(
        provider, model, llm, base_url=resolved_base_url, api_key=resolved_api_key
    )
    for section_type in gen.SECTION_ORDER:
        prompt = gen._format_prompt(section_type, context)
        content = await _generate_section_text(
            prompt, codebase_chunks, use_rlm, llm, rlm_model, rlm_state
        )
        if not content:
            content = "_(Содержимое раздела временно недоступно. Вы можете перезапустить генерацию.)_"
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

    repo_name = context.repo_name or _repo_name_from_url(repo_url)
    markdown = f"# Документация по кодовой базе: {repo_name}\n\n"
    for section_type in gen.SECTION_ORDER:
        sid = section_type.value
        title = get_section_title(sid, language)
        markdown += f"## {title}\n\n{sections[sid]}\n\n---\n\n"

    pages = _section_pages(sections, language)
    _persist_artifact(artifact, markdown, pages)
    # Index the cloned repo into cognee AFTER generation (non-blocking) into
    # the product-scoped dataset (item 1 cognee-first).
    _index_in_background(repo_dir, _cognee_dataset(product))
    return markdown


# ---------------------------------------------------------------------------
# Spec parsing (stdlib json/yaml) + structured renderers
# ---------------------------------------------------------------------------
def _parse_spec(content: str) -> Optional[dict]:
    text = (content or "").strip()
    if not text:
        return None
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    if yaml is not None:
        try:
            loaded = yaml.safe_load(text)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    return None


def _schema_field_table(schema: Any) -> List[str]:
    schema = schema or {}
    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    lines = [
        "| Поле | Тип | Обязательное | Описание |",
        "|------|-----|--------------|----------|",
    ]
    for field, fschema in props.items():
        fschema = fschema or {}
        ftype = fschema.get("type") or fschema.get("$ref", "")
        if isinstance(ftype, list):
            ftype = " | ".join(str(t) for t in ftype)
        desc = (fschema.get("description") or "").replace("\n", " ").strip()
        req = "да" if field in required else "нет"
        lines.append(f"| `{field}` | {ftype} | {req} | {desc} |")
    return lines


def _render_openapi_skeleton(spec: dict) -> str:
    md: List[str] = []
    info = spec.get("info", {}) or {}
    md.append(f"# {info.get('title', 'OpenAPI')}")
    if info.get("version"):
        md.append(f"**Версия:** `{info['version']}`")
    if info.get("description"):
        md.append(f"\n{info['description']}")

    servers = spec.get("servers", []) or []
    if servers:
        md.append("\n## Servers")
        for s in servers:
            s = s or {}
            md.append(f"- `{s.get('url', '')}` — {s.get('description', '')}")

    paths = spec.get("paths", {}) or {}
    if paths:
        md.append("\n## Endpoints")
        md.append("| Метод | Путь | Summary |")
        md.append("|-------|------|---------|")
        for path, methods in paths.items():
            for method, op in (methods or {}).items():
                if method.lower() not in ("get", "post", "put", "delete", "patch", "options", "head"):
                    continue
                op = op or {}
                md.append(f"| {method.upper()} | `{path}` | {op.get('summary', '')} |")

    components = spec.get("components", {}) or {}
    schemas = components.get("schemas", {}) or {}
    if schemas:
        md.append("\n## Schemas")
        for name, schema in schemas.items():
            md.append(f"\n### {name}")
            md.extend(_schema_field_table(schema))
    return "\n".join(md)


def _render_asyncapi_skeleton(spec: dict) -> str:
    md: List[str] = []
    info = spec.get("info", {}) or {}
    md.append(f"# {info.get('title', 'AsyncAPI')}")
    if spec.get("asyncapi"):
        md.append(f"**AsyncAPI version:** `{spec['asyncapi']}`")
    if info.get("version"):
        md.append(f"**Версия:** `{info['version']}`")
    if info.get("description"):
        md.append(f"\n{info['description']}")

    servers = spec.get("servers", {}) or {}
    if servers:
        md.append("\n## Servers")
        for name, srv in servers.items():
            srv = srv or {}
            md.append(
                f"- `{name}`: `{srv.get('url', '')}` ({srv.get('protocol', '')}) "
                f"— {srv.get('description', '')}"
            )

    channels = spec.get("channels", {}) or {}
    if channels:
        md.append("\n## Channels")
        md.append("| Канал | Операция | Message | Summary |")
        md.append("|-------|----------|---------|---------|")
        for name, ch in channels.items():
            ch = ch or {}
            for op in ("subscribe", "publish"):
                opdef = ch.get(op)
                if not opdef:
                    continue
                opdef = opdef or {}
                msg = opdef.get("message", "")
                if isinstance(msg, dict):
                    mname = msg.get("name") or msg.get("$ref", "")
                else:
                    mname = str(msg) if msg else ""
                md.append(f"| `{name}` | {op} | {mname} | {opdef.get('summary', '')} |")

    components = spec.get("components", {}) or {}
    schemas = components.get("schemas", {}) or {}
    if schemas:
        md.append("\n## Schemas")
        for name, schema in schemas.items():
            md.append(f"\n### {name}")
            md.extend(_schema_field_table(schema))
    return "\n".join(md)


def _render_raw_fallback(label: str, content: str, artifact: Any) -> str:
    name = getattr(artifact, "name", None) or label
    md = f"# {label}: {name}\n\n"
    md += "_(Не удалось разобрать спецификацию; показано исходное содержимое.)_\n\n"
    md += f"```yaml\n{_cap(content, 4000)}\n```\n"
    return md


def _render_testcase_skeleton(content: str, allure_url: str, artifact: Any) -> str:
    name = getattr(artifact, "name", None) or "Тест-кейсы"
    md = f"# Тест-кейсы: {name}\n\n"
    if allure_url:
        md += f"**Отчёт Allure:** [{allure_url}]({allure_url})\n\n"
    if content:
        md += content + "\n"
    else:
        md += "_(Содержимое тест-кейсов не предоставлено; см. ссылку Allure выше.)_\n"
    return md


# ---------------------------------------------------------------------------
# OpenAPI / AsyncAPI / Testcase documentation (standard LLM + stdlib render)
# ---------------------------------------------------------------------------
async def _generate_spec_doc(
    artifact: Any,
    product: Any,
    *,
    spec_kind: str,
    template_file: str,
    render_skeleton,
    page_id: str,
    page_title: str,
    provider: str,
    model: Optional[str],
    language: str,
) -> str:
    """Shared flow for openapi/asyncapi: parse -> structured render + LLM enrich."""
    content = (getattr(artifact, "content", "") or "").strip()
    if not content:
        raise ValueError(f"{spec_kind} artifact has empty content.")

    # Resolve admin docgen config (models.docgen.*) so the LLM enrichment hits
    # the configured gateway. Per-request provider/model overrides win.
    r_provider, r_model, r_base_url, r_api_key = _resolve_docgen_model(model)
    provider = provider or r_provider
    model = model or r_model

    from api.model_utils import get_model_context_window, clamp_text_by_tokens
    ctx_win = get_model_context_window(provider=provider, base_url=r_base_url, model_name=model, api_key=r_api_key, task="docgen")
    content_token_limit = max(1024, ctx_win - 2048)
    clamped_content = clamp_text_by_tokens(content, content_token_limit)

    spec = _parse_spec(content)
    skeleton = render_skeleton(spec) if spec else ""
    if not skeleton:
        skeleton = _render_raw_fallback(spec_kind, content, artifact)

    template = load_prompt_file(template_file, "")
    prompt = _with_verification_guard(_safe_replace(
        template,
        {
            "repo_name": _product_name(product, artifact),
            "artifact_name": getattr(artifact, "name", None) or spec_kind,
            "previous_content": "",
            "content": clamped_content,
        },
    ))
    llm_text = await _llm_or_none(
        prompt, provider, model, base_url=r_base_url, api_key=r_api_key
    )
    docs = llm_text or skeleton
    if not docs:
        docs = skeleton

    # Validate + repair any mermaid diagrams before persisting. The spec-doc
    # prompts instruct the LLM to emit architecture/flow diagrams; a broken one
    # would show as a render error. Non-fatal: returns docs unchanged if the
    # Node verifier is unavailable.
    try:
        docs, _mstats = await run_repair_loop(
            docs, _make_repair_llm(provider, model, base_url=r_base_url, api_key=r_api_key)
        )
    except Exception as e:  # pragma: no cover - verifier must never break gen
        logger.warning("Mermaid repair loop failed for %s doc: %s", spec_kind, e)

    pages = {
        page_id: {
            "id": page_id,
            "title": page_title,
            "content": docs,
            "filePaths": [],
            "importance": "high",
            "relatedPages": [],
        }
    }
    _persist_artifact(artifact, docs, pages)
    _index_in_background(content, _cognee_dataset(product))
    return docs


async def generate_openapi_docs(
    artifact: Any, product: Any, provider: str = None,
    model: Optional[str] = None, language: str = "ru",
) -> str:
    """Generate OpenAPI documentation (stdlib render + standard-LLM enrichment)."""
    return await _generate_spec_doc(
        artifact, product,
        spec_kind="OpenAPI",
        template_file="openapi_doc.md",
        render_skeleton=_render_openapi_skeleton,
        page_id="page_openapi",
        page_title="OpenAPI",
        provider=provider, model=model, language=language,
    )


async def generate_asyncapi_docs(
    artifact: Any, product: Any, provider: str = None,
    model: Optional[str] = None, language: str = "ru",
) -> str:
    """Generate AsyncAPI documentation (stdlib render + standard-LLM enrichment)."""
    return await _generate_spec_doc(
        artifact, product,
        spec_kind="AsyncAPI",
        template_file="asyncapi_doc.md",
        render_skeleton=_render_asyncapi_skeleton,
        page_id="page_asyncapi",
        page_title="AsyncAPI",
        provider=provider, model=model, language=language,
    )


async def generate_testcase_docs(
    artifact: Any, product: Any, provider: str = None,
    model: Optional[str] = None, language: str = "ru",
) -> str:
    """Generate test-case documentation. Allure URL is a LINK only (never fetched)."""
    content = (getattr(artifact, "content", "") or "").strip()
    allure_url = (getattr(artifact, "allure_url", "") or "").strip()
    if not content and not allure_url:
        raise ValueError("Test case artifact has no content and no Allure URL.")

    # Resolve admin docgen config (models.docgen.*) for the LLM enrichment.
    r_provider, r_model, r_base_url, r_api_key = _resolve_docgen_model(model)
    provider = provider or r_provider
    model = model or r_model

    from api.model_utils import get_model_context_window, clamp_text_by_tokens
    ctx_win = get_model_context_window(provider=provider, base_url=r_base_url, model_name=model, api_key=r_api_key, task="docgen")
    content_token_limit = max(1024, ctx_win - 2048)

    content_block = content or ""
    if allure_url:
        content_block += (
            "\n\n[Allure-отчёт]"
            f"({allure_url})"
            " (ссылка предоставлена вручную; данные Allure не загружаются автоматически)."
        )
    clamped_content_block = clamp_text_by_tokens(content_block, content_token_limit)

    template = load_prompt_file("testcase_doc.md", "")
    prompt = _with_verification_guard(_safe_replace(
        template,
        {
            "repo_name": _product_name(product, artifact),
            "artifact_name": getattr(artifact, "name", None) or "Тест-кейсы",
            "previous_content": "",
            "content": clamped_content_block,
        },
    ))
    llm_text = await _llm_or_none(
        prompt, provider, model, base_url=r_base_url, api_key=r_api_key
    )
    skeleton = _render_testcase_skeleton(content, allure_url, artifact)
    docs = llm_text or skeleton
    if not docs:
        docs = skeleton

    pages = {
        "page_testcase": {
            "id": "page_testcase",
            "title": "Тест-кейсы",
            "content": docs,
            "filePaths": [],
            "importance": "high",
            "relatedPages": [],
        }
    }
    _persist_artifact(artifact, docs, pages)
    _index_in_background(content_block, _cognee_dataset(product))
    return docs


# ---------------------------------------------------------------------------
# New artifact types: links / documentation / guides (item 2)
# ---------------------------------------------------------------------------
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

    from api.model_utils import get_model_context_window, clamp_text_by_tokens
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


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
async def generate_artifact_documentation(
    artifact: Any,
    product: Any,
    provider: str = None,
    model: Optional[str] = None,
    language: str = "ru",
) -> str:
    """Dispatch documentation generation by ``artifact.type`` (and ``kind``).

    Routes the new artifact-type enum (codebase|spec|links|documentation|guides)
    and maps legacy types (openapi/asyncapi/testcase) to the new (type, kind)
    pairs via ``LEGACY_ARTIFACT_TYPE_MAP`` so calls from clients still using the
    legacy vocabulary keep working:
    - ``codebase``         -> 7 wiki sections (RLM/standard LLM).
    - ``spec``             -> by ``kind``: ``asyncapi`` -> asyncapi render+LLM,
      otherwise (``openapi``) -> openapi render+LLM.
    - ``links``            -> links index page (no heavy gen).
    - ``documentation``    -> by ``kind``: ``testcase`` -> testcase render+LLM,
      otherwise manual/generated MD passthrough + optional LLM enrichment.
    - ``guides``           -> manual/generated MD passthrough.

    Returns the generated markdown and persists it onto ``artifact.generated_docs``
    + ``artifact.pages``. All generated content is indexed into the product-scoped
    cognee dataset ``prod_{product_id}`` (item 1 cognee-first). All backends
    degrade gracefully: cognee indexing is non-blocking, and RLM/LLM failures
    fall back to deterministic renders.
    """
    atype = (getattr(artifact, "type", "") or "").strip().lower()
    kind = (getattr(artifact, "kind", "") or "").strip().lower()
    # Route legacy types (openapi/asyncapi/testcase) to the new (type, kind).
    if atype in LEGACY_ARTIFACT_TYPE_MAP:
        atype, default_kind = LEGACY_ARTIFACT_TYPE_MAP[atype]
        kind = kind or default_kind

    if atype == "codebase":
        return await generate_codebase_docs(artifact, product, provider, model, language)
    if atype == "spec":
        if kind == "asyncapi":
            return await generate_asyncapi_docs(artifact, product, provider, model, language)
        return await generate_openapi_docs(artifact, product, provider, model, language)
    if atype == "links":
        return await generate_links_docs(artifact, product, provider, model, language)
    if atype == "documentation":
        if kind == "testcase":
            return await generate_testcase_docs(artifact, product, provider, model, language)
        return await generate_documentation_docs(artifact, product, provider, model, language)
    if atype == "guides":
        return await generate_guides_docs(artifact, product, provider, model, language)
    raise ValueError(f"Unsupported artifact type: {atype!r}")


__all__ = [
    "generate_artifact_documentation",
    "generate_codebase_docs",
    "generate_openapi_docs",
    "generate_asyncapi_docs",
    "generate_testcase_docs",
    "generate_links_docs",
    "generate_documentation_docs",
    "generate_guides_docs",
]
