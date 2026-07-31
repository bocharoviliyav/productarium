"""Product-scoped expert agent (item 3 — replaces Long-context tasks).

The expert agent answers questions about a single Product by grounding its
answers in the product's knowledge graph (cognee recall over the product-scoped
dataset ``prod_{product_id}``, which aggregates ALL artifacts: codebases, specs,
links, documentation, guides). When cognee has nothing indexed for the product,
it falls back to concatenating the artifacts' ``generated_docs`` / ``pages``.

Routing:
- Long context (combined prompt >= ``RLM_MIN_CHARS``) is routed through
  ``api.rlm_runner.run_rlm_task`` (fast-rlm) for deep, multi-step synthesis.
- Everything else uses the standard local LLM (adalflow OllamaClient /
  OpenAIClient via ``api.config.get_model_config``), with optional admin
  overrides from ``api.settings_store.get_model_for_task("expert")``.

Two entry points:
- ``run_expert_chat(product_id, query, messages, model=None, stream=True)``
    Conversational chat. When ``stream=True`` returns an async generator of
    text chunks; when ``stream=False`` returns a coroutine resolving to the
    full answer string.
- ``run_expert_doc(product_id, query, model=None) -> str``
    One-shot: returns a full self-contained Markdown document.

Prompt bodies live externally in ``refs/prompts/expert_agent_system.md`` and
``refs/prompts/expert_agent_doc.md`` and are loaded via
``api.prompts.load_prompt_file`` — never inlined here. Substitution uses
``str.replace`` (NOT ``str.format``) so literal Mermaid/JSON braces in the
prompt bodies stay unescaped (matches ``api/wiki_generator.py`` and
``api/artifact_docgen.py``).

Design rules (from AGENTS.md / the plan):
- No cloud API keys: LLM via local Ollama or local OpenAI-compatible server.
- All optional deps (adalflow, cognee, fast_rlm, the DB) are imported LAZILY
  inside the functions that need them and wrapped in try/except, so
  ``import api.expert_agent`` is cheap and every entry point degrades
  gracefully (returns "" / yields nothing) when a backend is unavailable.
- This module is file-disjoint from ``api/artifact_docgen.py``: the small
  ``_StandardLLM`` wrapper is replicated here as ``_ExpertLLM`` (with async
  streaming added) rather than imported, per the Wave 2 scope contract.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from api.logging_config import setup_logging
from api.prompts import load_prompt_file

setup_logging()
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
# RLM is for long context only (per the plan / api.artifact_docgen). Below this
# combined-prompt size we use the standard LLM directly.
RLM_MIN_CHARS = 20_000
# Cap the knowledge block injected into the prompt so very large products don't
# blow the LLM context window. RLM (when triggered) still receives the full
# prompt; this cap keeps the standard-LLM path manageable.
KNOWLEDGE_MAX_CHARS = 60_000
# Chunk size used when streaming a non-streaming source (RLM, or standard-LLM
# fallback) so the client still receives incremental SSE chunks.
STREAM_CHUNK_SIZE = 80

# Loaded once at import; the .md files are the source of truth.
EXPERT_SYSTEM_PROMPT: str = load_prompt_file("expert_agent_system.md", "")
EXPERT_DOC_PROMPT: str = load_prompt_file("expert_agent_doc.md", "")

# Per-section RLM timeout for the EXPERT path. fast-rlm can hang on a LOCAL
# model doing long-context synthesis (or on the first-run Deno/Pyodide
# bootstrap); without a cap the expert SSE stream stays open forever. Tunable
# via env (default 20 min). When the timeout fires we fall back to the
# standard LLM, which is the guaranteed-operation baseline.
RLM_EXPERT_TIMEOUT = float(os.environ.get("RLM_EXPERT_TIMEOUT", "1200"))

# Default language instruction substituted into {language_name}. The expert
# agent follows the user's language rather than a fixed one.
_DEFAULT_LANGUAGE_NAME = "the same language as the user's query (keep code identifiers, file paths, and API names in English)"


# --------------------------------------------------------------------------- #
# Prompt substitution (str.replace — NOT str.format — so Mermaid/JSON braces
# in refs/prompts/*.md stay unescaped; matches api/wiki_generator.py).
# --------------------------------------------------------------------------- #
def _safe_replace(template: str, variables: Dict[str, Any]) -> str:
    """Substitute ``{var}`` placeholders in ``template`` using exact replacement.

    Unmatched placeholders are left intact (matching
    ``WikiGenerator._format_prompt`` / ``api.artifact_docgen._safe_replace``).
    """
    if not template:
        return ""
    out = template
    for key, value in variables.items():
        out = out.replace("{" + key + "}", "" if value is None else str(value))
    return out


def _clean_llm_text(text: Optional[str]) -> str:
    """Strip surrounding whitespace, a wrapping ```markdown fence, <r> blocks,
    and any inline line-number prefixes the LLM emitted inside code blocks."""
    if not text:
        return ""
    t = text.strip()
    # Remove a leading ```lang fence and a trailing ``` if present.
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
        if t.endswith("```"):
            t = t[:-3]
    # Some local models emit  <r>... </r> blocks; strip them.
    t = re.sub(r" <r>.*?</r>", "", t, flags=re.DOTALL).strip()
    # Strip inline line-number prefixes inside code blocks (the UI renders its
    # own line numbers; duplicated prefixes are ugly). Shared implementation.
    try:
        from api.artifact_docgen import _strip_inline_line_numbers
        t = _strip_inline_line_numbers(t)
    except Exception:  # pragma: no cover - import-safe
        pass
    return t


def _cap(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (обрезано для контекста LLM)\n"


def _chunk_text(text: str, size: int = STREAM_CHUNK_SIZE) -> List[str]:
    """Split ``text`` into incremental pieces for chunked streaming delivery.

    Splits on newlines first (so line structure survives), then further splits
    over-long lines on word boundaries near ``size``.
    """
    if not text:
        return []
    pieces: List[str] = []
    for line in text.splitlines(keepends=True):
        if len(line) <= size:
            pieces.append(line)
            continue
        words = line.split(" ")
        buf = ""
        for w in words:
            candidate = w if not buf else buf + " " + w
            if len(candidate) > size and buf:
                pieces.append(buf)
                buf = w
            else:
                buf = candidate
        if buf:
            pieces.append(buf)
    return pieces


# --------------------------------------------------------------------------- #
# Standard (non-RLM) LLM wrapper — adalflow Generator over local Ollama /
# OpenAI-local. Replicated from api.artifact_docgen._StandardLLM (file-disjoint)
# and extended with async streaming + admin-configured base_url/api_key.
# --------------------------------------------------------------------------- #
class _ExpertLLM:
    """Thin LLM wrapper over the configured local model.

    Mirrors ``api.artifact_docgen._StandardLLM`` for non-streaming generation
    (adalflow ``Generator`` with ``template="{{input_str}}"``) and adds an async
    ``stream()`` that bypasses the Generator to call the model client's
    ``acall(stream=True)`` directly — the same pattern used by
    ``api/websocket_wiki.py``. Honors admin-configured ``base_url`` / ``api_key``
    from ``api.settings_store.get_model_for_task`` when provided.
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

        generator_config = get_model_config(provider, model)
        client_cls = generator_config["model_client"]
        self.model_kwargs = generator_config["model_kwargs"]
        self.model = self.model_kwargs.get("model", model)

        client_kwargs: Dict[str, Any] = {}
        if provider == "ollama":
            if base_url:
                client_kwargs["host"] = base_url
        elif provider in ("openai_local", "openai", "openai_compatible") or "openai" in str(provider):
            if base_url:
                client_kwargs["base_url"] = base_url
            if api_key:
                client_kwargs["api_key"] = api_key
        self.provider = provider
        self.model_client = client_cls(**client_kwargs)
        self.generator = adal.Generator(
            template="{{input_str}}",
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
        )

    async def generate(self, prompt: str) -> str:
        """Non-streaming generation. Returns the raw response text."""

        def _call() -> str:
            # adalflow 1.x Generator.call() takes ``prompt_kwargs`` (the dict
            # that fills the ``{{input_str}}`` template placeholder), NOT a
            # bare ``input_str=`` kwarg. Passing ``input_str=`` raises
            # ``TypeError: Generator.call() got an unexpected keyword argument
            # 'input_str'`` (see api/wiki_generator.py:283 for the pattern).
            result = self.generator(prompt_kwargs={"input_str": prompt})
            # On a model error (e.g. 401 / connection refused) adalflow returns
            # a GeneratorOutput with ``error`` set and ``data=None`` but stores
            # the full prompt on ``input`` for tracing. Returning
            # ``str(result)`` would leak the entire prompt (system prompt +
            # retrieved context + query) into the expert chat/doc. Treat any
            # error as "no generation" so the caller surfaces a graceful
            # failure message instead of the raw prompt.
            if getattr(result, "error", None):
                logger.warning("Expert LLM returned an error: %s", result.error)
                return ""
            for attr in ("data", "response", "answer", "raw_response", "output"):
                val = getattr(result, attr, None)
                if val:
                    return str(val)
            return ""

        return await asyncio.to_thread(_call)

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Async-stream text chunks from the local LLM.

        Attempts true token streaming via the model client's ``acall``; on any
        failure falls back to non-streaming ``generate`` + chunked delivery so
        the caller always receives incremental chunks.
        """
        try:
            from adalflow.core.types import ModelType

            if self.provider == "ollama":
                # Low temperature + seed for deterministic expert answers
                # (matches generator.json). Defaults 0.1/0.9.
                _exp_ollama_options = {
                    "temperature": self.model_kwargs.get("temperature", 0.1),
                    "top_p": self.model_kwargs.get("top_p", 0.9),
                    "num_ctx": self.model_kwargs.get("num_ctx", 32000),
                }
                if "seed" in self.model_kwargs:
                    _exp_ollama_options["seed"] = self.model_kwargs["seed"]
                mk = {
                    "model": self.model,
                    "stream": True,
                    "options": _exp_ollama_options,
                }
            else:
                mk = {
                    "model": self.model,
                    "stream": True,
                    "temperature": self.model_kwargs.get("temperature", 0.1),
                }
                if "top_p" in self.model_kwargs:
                    mk["top_p"] = self.model_kwargs["top_p"]
                if "seed" in self.model_kwargs:
                    mk["seed"] = self.model_kwargs["seed"]

            api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
                input=prompt, model_kwargs=mk, model_type=ModelType.LLM
            )
            response = await self.model_client.acall(
                api_kwargs=api_kwargs, model_type=ModelType.LLM
            )
            produced = False
            async for chunk in response:  # type: ignore[union-attr]
                text = _extract_chunk_text(chunk, self.provider)
                if text:
                    produced = True
                    yield text
            if produced:
                return
            logger.warning(
                "Expert LLM stream produced no chunks; falling back to generate."
            )
        except Exception as e:  # pragma: no cover - depends on live Ollama/HTTP
            logger.warning(
                "Expert LLM streaming failed (%s); falling back to chunked generate.",
                e,
            )

        # Fallback: non-streaming generate + chunked yield.
        try:
            text = _clean_llm_text(await self.generate(prompt))
        except Exception as e:  # pragma: no cover - depends on live Ollama
            logger.warning("Expert LLM fallback generate failed: %s", e)
            return
        for piece in _chunk_text(text):
            yield piece


def _extract_chunk_text(chunk: Any, provider: str) -> Optional[str]:
    """Extract a text delta from a streaming chunk (Ollama or OpenAI shape).

    Mirrors the extraction logic in ``api/websocket_wiki.py`` for both the
    Ollama native format (``message.content``) and the OpenAI streaming format
    (``choices[0].delta.content``).
    """
    # Ollama native format
    message = getattr(chunk, "message", None)
    if message is not None:
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            return content
    # dict-shaped Ollama chunk
    if isinstance(chunk, dict):
        msg = chunk.get("message")
        if isinstance(msg, dict) and msg.get("content"):
            return msg["content"]
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or choices[0].get("message") or {}
            if isinstance(delta, dict) and delta.get("content"):
                return delta["content"]
        if chunk.get("content"):
            return chunk["content"]
    # OpenAI ChatCompletion chunk format
    choices = getattr(chunk, "choices", None)
    if choices:
        try:
            delta = getattr(choices[0], "delta", None)
        except Exception:
            delta = None
        if delta is not None:
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                return content
    # adalflow GeneratorOutput-style: .response / .data
    for attr in ("response", "data", "text"):
        val = getattr(chunk, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


def _safe_build_llm(
    provider: str,
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[_ExpertLLM]:
    """Build an ``_ExpertLLM`` or return None on any failure (graceful degrade)."""
    try:
        return _ExpertLLM(provider, model, base_url=base_url, api_key=api_key)
    except Exception as e:  # pragma: no cover - depends on live config/Ollama
        logger.warning(
            "Could not initialise expert LLM (%s/%s): %s. "
            "Expert answers will be empty until a model is available.",
            provider,
            model,
            e,
        )
        return None


# --------------------------------------------------------------------------- #
# Model / knowledge / prompt assembly
# --------------------------------------------------------------------------- #
def _resolve_expert_model(model: Optional[str]) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Resolve (provider, model, base_url, api_key) for the expert task.

    Reads admin-configured ``models.expert.*`` from the Config Abstraction Layer.
    """
    try:
        from api.config_abstraction import get_task_config

        cfg = get_task_config("expert") or {}
        provider = cfg.get("provider") or os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local")
        resolved_model = model or cfg.get("model") or os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b")
        return provider, resolved_model, cfg.get("base_url"), cfg.get("api_key")
    except Exception as e:  # pragma: no cover - settings store is import-safe
        logger.debug("get_task_config(expert) failed; using defaults: %s", e)
        return os.environ.get("DEEPWIKI_DEFAULT_PROVIDER", "openai_local"), model or os.environ.get("DEEPWIKI_DEFAULT_MODEL", "qwen/qwen3.6-27b"), None, None


def _product_name_by_id(product_id: str) -> str:
    """Look up a product name from the DB; fall back to the id. Non-fatal."""
    try:
        from api.db import SessionLocal
        from api.models import ProductORM

        with SessionLocal() as db:
            p = db.get(ProductORM, product_id)
            if p is not None and getattr(p, "name", None):
                return p.name
    except Exception as e:
        logger.debug("product name lookup failed for %r: %s", product_id, e)
    return product_id


def _fallback_artifact_docs(product_id: str) -> str:
    """Concatenate artifact generated_docs + page content when cognee is empty.

    Opens its own short-lived session (non-fatal: returns "" on any error or
    when the product/artifacts are missing).
    """
    try:
        from sqlalchemy.orm import selectinload

        from api.db import SessionLocal
        from api.models import ProductORM

        with SessionLocal() as db:
            p = (
                db.query(ProductORM)
                .options(selectinload(ProductORM.artifacts))
                .filter(ProductORM.id == product_id)
                .first()
            )
            if p is None:
                return ""
            parts: List[str] = []
            for a in p.artifacts:
                name = getattr(a, "name", None) or getattr(a, "id", "artifact")
                docs = getattr(a, "generated_docs", None) or ""
                if docs:
                    parts.append(f"## Artifact: {name}\n{docs}")
                pages = getattr(a, "pages", None) or {}
                if isinstance(pages, dict):
                    for page_id, page in pages.items():
                        content = ""
                        if isinstance(page, dict):
                            content = page.get("content") or ""
                        elif isinstance(page, str):
                            content = page
                        if content:
                            parts.append(
                                f"## Artifact: {name} / page {page_id}\n{content}"
                            )
            return "\n\n".join(parts)
    except Exception as e:
        logger.warning(
            "Fallback artifact docs failed for product %r: %s", product_id, e
        )
        return ""


async def _retrieve_product_knowledge(product_id: str, query: str) -> str:
    """Retrieve product knowledge from cognee (prod_{product_id}); fall back to
    concatenated artifact docs or live Confluence. Never raises; returns "" if nothing available.
    """
    dataset = f"prod_{product_id}"
    try:
        from api.cognee_manager import query_cognee

        ctx = await query_cognee(query, dataset_name=dataset, top_k=20)
        if ctx:
            logger.info(
                "Expert: cognee recall for %r returned %d chars.", dataset, len(ctx)
            )
            return ctx
        logger.info("Expert: cognee recall empty for %r; using artifact fallback.", dataset)
    except Exception as e:
        logger.warning("Expert: cognee recall failed for %r: %s", dataset, e)

    fallback_docs = _fallback_artifact_docs(product_id)
    if fallback_docs:
        return fallback_docs

    # Fallback to live Confluence (direct or MCP) if configured
    try:
        from api.integrations.registry import get_connector
        c_connector = get_connector("confluence")
        if c_connector and c_connector.is_configured():
            spaces = c_connector.list_spaces()
            if spaces:
                sp_id = spaces[0].get("key") or spaces[0].get("id")
                if sp_id:
                    pulled = c_connector.pull(sp_id, opts={"recursive": False})
                    if pulled and pulled.get("markdown"):
                        return pulled["markdown"]
    except Exception as e:
        logger.debug("Expert live Confluence fallback skipped for %s: %s", product_id, e)

    return ""


def _format_history(messages: List[Dict[str, Any]]) -> str:
    """Render prior conversation turns (user/assistant pairs) as a history block."""
    if not messages:
        return ""
    lines: List[str] = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not role or not content:
            continue
        if role == "user":
            lines.append(f"<user>{content}</user>")
        elif role == "assistant":
            lines.append(f"<assistant>{content}</assistant>")
        else:
            lines.append(f"<{role}>{content}</{role}>")
    return "\n".join(lines)


def _build_prompt(
    template: str,
    product_name: str,
    knowledge: str,
    history: str,
    query: str,
    language_name: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Assemble the full expert prompt from a loaded template body.

    The template (from refs/prompts/*.md) carries role/guidelines and the
    ``{product_name}`` / ``{language_name}`` placeholders. Knowledge, history,
    and the query are appended as structured blocks (NOT substituted into the
    template) so the template body stays small and Mermaid/JSON-safe.
    
    Dynamically clamps knowledge and history to fit the target model's actual
    context window.
    """
    system = _safe_replace(
        template,
        {
            "product_name": product_name or "this product",
            "language_name": language_name or _DEFAULT_LANGUAGE_NAME,
        },
    )
    try:
        from api.prompts import VERIFICATION_GUARD as _guard
    except Exception:  # pragma: no cover - import-safe
        _guard = ""
    if _guard:
        system = system + "\n\n" + _guard

    try:
        from api.model_utils import get_model_context_window, clamp_text_by_tokens, _count_tokens
        ctx_win = get_model_context_window(provider=provider, base_url=base_url, model_name=model, api_key=api_key, task="expert")
    except Exception:
        ctx_win = 8192
        from api.model_utils import clamp_text_by_tokens, _count_tokens

    # Reserve 2048 tokens for system instructions, query, and LLM output completion
    avail_tokens = max(1024, ctx_win - 2048)
    
    # 1. Clamp history to at most 1/3 of available prompt budget
    clamped_history = ""
    if history:
        history_budget = max(512, avail_tokens // 3)
        clamped_history = clamp_text_by_tokens(history, history_budget, preserve_tail=True)

    # 2. Clamp knowledge to the remaining prompt budget
    history_tokens = _count_tokens(clamped_history)
    knowledge_budget = max(512, avail_tokens - history_tokens)
    clamped_knowledge = clamp_text_by_tokens(knowledge, knowledge_budget, preserve_tail=False)

    prompt = system + "\n\n"
    if clamped_history:
        prompt += f"<conversation_history>\n{clamped_history}\n</conversation_history>\n\n"
    if clamped_knowledge:
        prompt += f"<product_knowledge>\n{clamped_knowledge}\n</product_knowledge>\n\n"
    else:
        prompt += (
            "<note>No indexed product knowledge was available. Answer honestly: "
            "say the knowledge is missing and suggest indexing artifacts.</note>\n\n\n"
        )
    prompt += f"<query>\n{query}\n</query>\n\nAssistant: "
    return prompt


# --------------------------------------------------------------------------- #
# Answer generation (RLM long-context -> standard-LLM fallback)
# --------------------------------------------------------------------------- #
async def _rlm_generate(prompt: str, model: Optional[str]) -> str:
    """Run fast-rlm and return cleaned text, or "" on any failure/empty result.

    Capped by ``RLM_EXPERT_TIMEOUT`` so a hung fast-rlm (first-run bootstrap,
    dead Pyodide worker, a local model that never returns) cannot hold the
    expert SSE stream open indefinitely. On timeout we return "" and the caller
    falls back to the standard LLM.
    """
    try:
        from api.model_utils import get_model_context_window, clamp_text_by_tokens
        ctx_win = get_model_context_window(model_name=model, task="expert")
        completion_res = max(1024, min(4096, ctx_win // 4))
        max_prompt_limit = max(1024, ctx_win - completion_res)
        safe_prompt_limit = max(2000, max_prompt_limit - 6000)
        safe_prompt = clamp_text_by_tokens(prompt, safe_prompt_limit)

        from api.rlm_runner import run_rlm_task  # lazy: fast_rlm is optional

        res = await asyncio.wait_for(
            run_rlm_task(safe_prompt, model), timeout=RLM_EXPERT_TIMEOUT
        )
        if isinstance(res, dict) and res.get("success") and res.get("results"):
            return _clean_llm_text(str(res["results"]))
        logger.warning("RLM returned no usable result for expert task.")
    except asyncio.TimeoutError:
        logger.warning(
            "RLM expert generation timed out after %ss; falling back to standard LLM.",
            RLM_EXPERT_TIMEOUT,
        )
    except Exception as e:  # pragma: no cover - depends on live fast-rlm
        logger.warning("RLM expert generation failed: %s", e)
    return ""


async def _generate_answer(
    prompt: str,
    provider: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    use_rlm: bool,
) -> str:
    """Non-streaming answer: RLM (long context) -> standard LLM fallback -> ""."""
    if use_rlm:
        text = await _rlm_generate(prompt, model)
        if text:
            return text
        logger.warning("Expert RLM path empty; falling back to standard LLM.")
    llm = _safe_build_llm(provider, model, base_url=base_url, api_key=api_key)
    if llm is None:
        return ""
    try:
        return _clean_llm_text(await llm.generate(prompt))
    except Exception as e:  # pragma: no cover - depends on live Ollama
        logger.warning("Expert standard LLM generate failed: %s", e)
        return ""


async def _stream_answer(
    prompt: str,
    provider: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    use_rlm: bool,
) -> AsyncIterator[str]:
    """Streaming answer: RLM (chunked) -> standard LLM stream (with fallback)."""
    if use_rlm:
        text = await _rlm_generate(prompt, model)
        if text:
            for piece in _chunk_text(text):
                yield piece
            return
        logger.warning("Expert RLM stream empty; falling back to standard LLM.")
    llm = _safe_build_llm(provider, model, base_url=base_url, api_key=api_key)
    if llm is None:
        return
    async for piece in llm.stream(prompt):
        yield piece


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def _resolve_use_rlm(
    use_rlm: Optional[bool], task: str, prompt_len: int
) -> bool:
    """Resolve the final RLM flag from an optional override + the admin mode.

    Precedence:
    - ``use_rlm=True``  -> always RLM (caller forced it; LLM fallback still fires
      if RLM fails/times out).
    - ``use_rlm=False`` -> never RLM (caller forced the standard LLM).
    - ``use_rlm=None``  -> follow the admin ``rlm.<task>.mode`` setting:
        * ``llm``  -> False
        * ``rlm``  -> True
        * ``auto`` -> True only when the prompt is large (>= RLM_MIN_CHARS).
    This keeps "LLM is the guaranteed baseline; RLM is opt-in" regardless of
    how the mode is configured: ``get_rlm_mode`` already returns ``llm`` when
    fast-rlm is not installed, so ``None`` never accidentally enables RLM.
    """
    if use_rlm is True:
        return True
    if use_rlm is False:
        return False
    try:
        from api.settings_store import get_rlm_mode
        mode = get_rlm_mode(task)
    except Exception:  # pragma: no cover - settings store import-safe
        mode = "auto"
    if mode == "llm":
        return False
    if mode == "rlm":
        return True
    return prompt_len >= RLM_MIN_CHARS


async def _run_expert_chat_collect(
    product_id: str,
    query: str,
    messages: List[Dict[str, Any]],
    model: Optional[str],
    use_rlm: Optional[bool],
) -> str:
    """Non-streaming chat: returns the full answer string."""
    provider, resolved_model, base_url, api_key = _resolve_expert_model(model)
    knowledge = await _retrieve_product_knowledge(product_id, query)
    history = _format_history(messages)
    prompt = _build_prompt(
        EXPERT_SYSTEM_PROMPT, _product_name_by_id(product_id), knowledge, history, query,
        provider=provider, base_url=base_url, model=resolved_model, api_key=api_key,
    )
    use_rlm_resolved = _resolve_use_rlm(use_rlm, "expert", len(prompt))
    logger.info(
        "Expert chat (collect) product=%s prompt_chars=%d use_rlm=%s",
        product_id,
        len(prompt),
        use_rlm_resolved,
    )
    return await _generate_answer(
        prompt, provider, resolved_model, base_url, api_key, use_rlm_resolved
    )


async def _run_expert_chat_stream(
    product_id: str,
    query: str,
    messages: List[Dict[str, Any]],
    model: Optional[str],
    use_rlm: Optional[bool],
) -> AsyncIterator[str]:
    """Streaming chat: yields answer text chunks."""
    provider, resolved_model, base_url, api_key = _resolve_expert_model(model)
    knowledge = await _retrieve_product_knowledge(product_id, query)
    history = _format_history(messages)
    prompt = _build_prompt(
        EXPERT_SYSTEM_PROMPT, _product_name_by_id(product_id), knowledge, history, query,
        provider=provider, base_url=base_url, model=resolved_model, api_key=api_key,
    )
    use_rlm_resolved = _resolve_use_rlm(use_rlm, "expert", len(prompt))
    logger.info(
        "Expert chat (stream) product=%s prompt_chars=%d use_rlm=%s",
        product_id,
        len(prompt),
        use_rlm_resolved,
    )
    async for piece in _stream_answer(
        prompt, provider, resolved_model, base_url, api_key, use_rlm_resolved
    ):
        yield piece


def run_expert_chat(
    product_id: str,
    query: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    stream: bool = True,
    use_rlm: Optional[bool] = None,
):
    """Product-scoped expert chat.

    Args:
        product_id: The product to ground the answer in.
        query: The current user question.
        messages: Prior conversation history as ``[{"role": "user"|"assistant",
            "content": "..."}]`` (the current query is passed separately).
        model: Optional model override; otherwise resolved from the admin
            ``models.expert`` config (with env fallbacks).
        stream: When True, returns an async generator yielding text chunks.
            When False, returns a coroutine resolving to the full answer string.
        use_rlm: Optional explicit RLM override. ``True`` forces RLM (with LLM
            fallback), ``False`` forces the standard LLM, ``None`` (default)
            follows the admin ``rlm.expert.mode`` setting (auto/rlm/llm).

    Returns:
        An async generator (``stream=True``) or a coroutine (``stream=False``).
    """
    msgs = list(messages or [])
    if stream:
        return _run_expert_chat_stream(product_id, query, msgs, model, use_rlm)
    return _run_expert_chat_collect(product_id, query, msgs, model, use_rlm)


async def run_expert_doc(
    product_id: str,
    query: str,
    model: Optional[str] = None,
    use_rlm: Optional[bool] = None,
) -> str:
    """One-shot expert document: returns a full self-contained Markdown string.

    Uses the ``expert_agent_doc.md`` prompt variant, includes product knowledge
    (no conversation history), and never streams.
    """
    provider, resolved_model, base_url, api_key = _resolve_expert_model(model)
    knowledge = await _retrieve_product_knowledge(product_id, query)
    prompt = _build_prompt(
        EXPERT_DOC_PROMPT, _product_name_by_id(product_id), knowledge, "", query,
        provider=provider, base_url=base_url, model=resolved_model, api_key=api_key,
    )
    use_rlm_resolved = _resolve_use_rlm(use_rlm, "expert", len(prompt))
    logger.info(
        "Expert doc product=%s prompt_chars=%d use_rlm=%s",
        product_id,
        len(prompt),
        use_rlm_resolved,
    )
    doc = await _generate_answer(
        prompt, provider, resolved_model, base_url, api_key, use_rlm_resolved
    )
    if not doc:
        doc = (
            f"# Expert document for {product_id}\n\n"
            "_(No content was generated. Ensure the product has indexed "
            "knowledge (cognee) or generated artifact docs, and that a local "
            "LLM is available.)_\n"
        )
    return doc


__all__ = [
    "run_expert_chat",
    "run_expert_doc",
    "EXPERT_SYSTEM_PROMPT",
    "EXPERT_DOC_PROMPT",
]
