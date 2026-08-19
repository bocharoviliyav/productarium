import os
import logging
import asyncio

logger = logging.getLogger(__name__)

# fast_rlm is an optional, heavy dependency (it pulls Deno + Pyodide on first
# run). Import it defensively so this module always loads and every entrypoint
# can degrade gracefully when fast_rlm is not installed.
try:
    from fast_rlm import run, RLMConfig  # type: ignore
    _FAST_RLM_AVAILABLE = True
except Exception as _fast_rlm_import_err:  # pragma: no cover - optional dep
    run = None  # type: ignore
    RLMConfig = None  # type: ignore
    _FAST_RLM_AVAILABLE = False
    logger.warning("fast_rlm could not be imported; RLM features disabled: %s", _fast_rlm_import_err)


# --------------------------------------------------------------------------- #
# Per-scenario recursion budgets (token-based: local providers return no cost
# so max_money_spent is inert per the fast-rlm docs).
# Expert: deeper recursion for exhaustive cross-artifact search.
# Docgen: moderate recursion; the agent reads files on demand via tools.
# --------------------------------------------------------------------------- #
RLM_EXPERT_MAX_DEPTH = int(os.environ.get("RLM_EXPERT_MAX_DEPTH", "4"))
RLM_EXPERT_MAX_CALLS = int(os.environ.get("RLM_EXPERT_MAX_CALLS", "20"))
RLM_DOCGEN_MAX_DEPTH = int(os.environ.get("RLM_DOCGEN_MAX_DEPTH", "3"))
RLM_DOCGEN_MAX_CALLS = int(os.environ.get("RLM_DOCGEN_MAX_CALLS", "20"))


def get_rlm_session_dir(scope: str) -> str:
    """Resolve (and lazily create) a persistent on-disk session directory.

    fast-rlm persists the root agent's REPL state (built indexes, parsed
    corpora, helper functions, query→FINAL ledger) under ``session_dir`` so
    follow-up ``query()`` calls reuse it (measured ~2.6× cheaper in the docs).
    One directory per scope (``expert_<product_id>`` or
    ``docgen_<codebase_id>``); the ``session_id`` distinguishes conversations
    within a directory.

    Constraint (from the docs): one live query per session dir — no concurrent
    queries. Expert chat is serial per product (one SSE stream at a time per
    product), which is acceptable.
    """
    base = os.path.expanduser("~/.adalflow/rlm_sessions")
    session_dir = os.path.join(base, scope)
    try:
        os.makedirs(session_dir, exist_ok=True)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not create RLM session dir %r: %s", session_dir, e)
    return session_dir


def _resolve_rlm_config(model_name: str = None):
    """Resolve a configured ``RLMConfig`` + model id + base_url for fast-rlm.

    Shared by ``run_rlm_task_sync`` (flat-string path) and
    ``run_rlm_structured_sync`` (dict + tools + sessions path). Does all the
    admin-store model/base_url/api_key resolution, env export, and
    context-window clamping. Returns ``(config, resolved_model, base_url)``
    or raises if fast_rlm is unavailable.
    """
    # Resolve the LLM config from the admin store (models.docgen.*) so RLM
    # hits the corporate AI gateway when configured, instead of the dead
    # env-default LM Studio :1234. Falls back to env vars when the store / DB
    # is unavailable or the api_key can't decrypt. The admin base_url/api_key
    # are exported to RLM_MODEL_BASE_URL / RLM_MODEL_API_KEY because fast-rlm
    # reads them from the process environment (it runs in a Pyodide REPL that
    # cannot reach host Python).
    admin_base_url = None
    admin_api_key = None
    admin_model = None
    admin_max_prompt_tokens = None
    try:
        from api.config.settings import get_model_for_task
        cfg = get_model_for_task("docgen") or {}
        admin_base_url = cfg.get("base_url")
        admin_api_key = cfg.get("api_key")
        admin_model = cfg.get("model")
        admin_max_prompt_tokens = cfg.get("max_prompt_tokens")
    except Exception as e:  # pragma: no cover - settings store is import-safe
        logger.debug("get_model_for_task(docgen) failed in rlm_runner: %s", e)

    # Enforce standard RLM environment keys. Admin api_key wins over env.
    if admin_api_key and admin_api_key not in ("not-needed", "not_needed"):
        os.environ["RLM_MODEL_API_KEY"] = admin_api_key
        os.environ["OPENAI_API_KEY"] = admin_api_key
        os.environ["LOCAL_OPENAI_API_KEY"] = admin_api_key
    elif not os.environ.get("RLM_MODEL_API_KEY"):
        key_val = os.environ.get("LOCAL_OPENAI_API_KEY") or "not-needed"
        os.environ["RLM_MODEL_API_KEY"] = key_val
        if key_val not in ("not-needed", "not_needed"):
            os.environ["OPENAI_API_KEY"] = key_val

    # Resolve the OpenAI-compatible base URL for fast-rlm. Single path:
    #   1. admin models.docgen.base_url (corporate gateway etc.)
    #   2. LOCAL_OPENAI_BASE_URL (local OpenAI-compatible server)
    #   3. http://localhost:1234/v1 (default, LM Studio)
    base_url = (
        admin_base_url
        or os.environ.get("LOCAL_OPENAI_BASE_URL")
        or "http://localhost:1234/v1"
    )
    # Normalize to a clean ``/v1`` base: the fast-rlm Deno engine builds
    # ``new OpenAI({ baseURL })`` and the SDK POSTs to
    # ``${baseURL}/chat/completions``. If the admin pasted a bare host like
    # ``https://ai.corp.gateway`` (no ``/v1``), the SDK posts to
    # ``https://ai.corp.gateway/chat/completions`` — most gateways only expose
    # the OpenAI surface under ``/v1``, so a bare host 404s or closes the
    # connection, surfacing as "Connection error". Mirrors cognee's
    # ``_host_to_v1`` normalization (strips trailing slashes + a stray
    # ``/embeddings``/``/v1`` then appends ``/v1``).
    try:
        from api.cognee._runtime import _host_to_v1
        normalized = _host_to_v1(base_url)
        if normalized:
            base_url = normalized
    except Exception:  # pragma: no cover - import-safe
        _b = base_url.rstrip("/")
        if not _b.lower().endswith("/v1"):
            base_url = f"{_b}/v1"
    os.environ["RLM_MODEL_BASE_URL"] = base_url

    # Resolve model. Precedence:
    #   1. explicit model_name (callers pass the already-admin-resolved model;
    #      prewarm passes None so admin config wins)
    #   2. admin models.docgen.model (DB SettingORM, highest authority)
    #   3. RLM_MODEL_NAME env var
    #   4. "z-ai/glm-5" last-resort default
    # The endpoint is an OpenAI-compatible server (LM Studio, llama.cpp, vLLM,
    # corporate gateway); pass the model name through verbatim -- these servers
    # use their own model IDs.
    resolved_model = model_name or admin_model or os.environ.get("RLM_MODEL_NAME") or "z-ai/glm-5"
    # fast-rlm's Deno engine builds ``new OpenAI({ apiKey, baseURL })`` and calls
    # ``chat.completions.create({ model: resolvedModel, ... })``. The OpenAI SDK
    # POSTs to ``${baseURL}/chat/completions`` verbatim — there is NO litellm
    # routing layer, so the model name must be passed through UNTAMPERED (a
    # gateway alias like ``flash`` is what the gateway expects; prefixing with
    # ``openai/`` makes the SDK send ``model: "openai/flash"`` which the
    # gateway rejects as model-not-found -> "Connection error").
    # Strip a stray litellm provider prefix the admin might have pasted in the
    # settings UI (cognee normalizes its model WITH a prefix, so the admin
    # store sometimes carries ``openai/flash``). fast-rlm's native OpenAI SDK
    # must receive the bare alias the gateway actually knows.
    try:
        from api.cognee._runtime import _strip_provider_prefix
        resolved_model = _strip_provider_prefix(resolved_model)
    except Exception:  # pragma: no cover - import-safe
        if resolved_model.startswith("openai/"):
            resolved_model = resolved_model[len("openai/"):]

    config = RLMConfig.default()
    config.primary_agent = resolved_model
    config.sub_agent = resolved_model
    # Conservative defaults for the flat-string path; the structured path
    # applies larger per-scenario budgets (see run_rlm_structured_sync).
    config.max_depth = 2
    config.max_calls_per_subagent = 10

    # fast-rlm ships with a 30s per-API-call timeout (rlm_config.yaml:
    # api_timeout_ms: 30000). That is far too short for a LOCAL model doing
    # long-context generation (e.g. LM Studio running qwen3.6-27b can take
    # several minutes for a single long completion). Without raising this,
    # long tasks fail mid-run with "Request timed out." even though the model
    # is still happily generating. Resolved through the central timeout config
    # (admin > env > default) so the admin "Timeouts" panel overrides without a
    # restart. Default 3600000ms (1h), floor 30000ms.
    try:
        from api.config.timeout import resolve_rlm_api_timeout_ms
        config.api_timeout_ms = resolve_rlm_api_timeout_ms()
    except Exception:
        config.api_timeout_ms = 3600000
    # Likewise let the prompt/completion token budgets grow for long tasks
    # (fast-rlm defaults: max_prompt_tokens=200000, max_completion_tokens=50000).
    # Precedence for the prompt budget: admin models.docgen.max_prompt_tokens >
    # env RLM_MAX_PROMPT_TOKENS > fast-rlm default. The admin value is the
    # optional per-model override surfaced in the admin panel; when it is unset
    # we keep the default behavior exactly as before. max_completion_tokens
    # stays env-only (not the failure case being fixed).
    resolved_max_prompt_tokens = admin_max_prompt_tokens
    if resolved_max_prompt_tokens is None and os.environ.get("RLM_MAX_PROMPT_TOKENS"):
        try:
            resolved_max_prompt_tokens = int(os.environ["RLM_MAX_PROMPT_TOKENS"])
        except ValueError:
            resolved_max_prompt_tokens = None
    if resolved_max_prompt_tokens is not None:
        config.max_prompt_tokens = resolved_max_prompt_tokens
    if os.environ.get("RLM_MAX_COMPLETION_TOKENS"):
        try:
            config.max_completion_tokens = int(os.environ["RLM_MAX_COMPLETION_TOKENS"])
        except ValueError:
            pass

    # --- Clamp budgets to the MODEL'S REAL context window ----------------------
    # fast-rlm's ``max_prompt_tokens`` (default 200000) is a BUDGET fast-rlm
    # believes it can use, NOT the model's actual ``num_ctx``. A local
    # LM Studio / vLLM model's effective ``num_ctx`` defaults to e.g. 8192/32768
    # -- far below 200k. When fast-rlm sends a prompt larger than the gateway's
    # real window, the gateway returns HTTP 400
    # ``litellm.BadRequestError: ... Context size has been exceeded`` and every
    # section fails.
    #
    # Dynamically query/detect the model's actual context window and clamp both
    # ``max_prompt_tokens`` and ``max_completion_tokens`` so their SUM never
    # exceeds the window.
    _model_ctx = None
    try:
        from api.utils import get_model_context_window
        _model_ctx = get_model_context_window(
            base_url=base_url,
            model_name=resolved_model,
            api_key=admin_api_key,
            task="docgen",
        )
    except Exception as _ctx_err:
        logger.debug("Could not resolve model context window in rlm_runner: %s", _ctx_err)

    if _model_ctx and _model_ctx > 0:
        # Reserve room for the completion within the same window (e.g. up to 4096
        # tokens or a quarter of the window). Keep the rest for prompt tokens so
        # fast-rlm subagents have maximum prompt headroom.
        _completion_budget = int(getattr(config, "max_completion_tokens", 50000) or 50000)
        if _completion_budget >= _model_ctx:
            _completion_budget = max(1024, min(4096, _model_ctx // 4))
        config.max_completion_tokens = _completion_budget
        _prompt_cap = max(1024, _model_ctx - _completion_budget)
        config.max_prompt_tokens = _prompt_cap

    return config, resolved_model, base_url


def run_rlm_task_sync(query: str, model_name: str = None) -> dict:
    """
    Runs an iterative reasoning process with fast-rlm.
    This runs synchronously and should be wrapped in an executor/thread.
    """
    if not _FAST_RLM_AVAILABLE:
        return {
            "results": "fast_rlm not available",
            "usage": {},
            "success": False,
        }
    try:
        config, resolved_model, base_url = _resolve_rlm_config(model_name)
    except Exception as e:
        logger.error(f"RLM config resolution failed: {e}", exc_info=True)
        return {
            "results": f"Failed to configure RLM task: {str(e)}",
            "usage": {},
            "success": False,
        }
    logger.info(
        f"Triggering fast-rlm task reasoning. Model: {resolved_model}, "
        f"base_url: {base_url}, api_timeout_ms: {config.api_timeout_ms}, "
        f"max_prompt_tokens: {config.max_prompt_tokens}, "
        f"max_completion_tokens: {getattr(config, 'max_completion_tokens', '?')}"
    )
    try:
        result = run(query, config=config, verbose=True)
        return {
            "results": result.get("results", "No result returned"),
            "usage": result.get("usage", {}),
            "success": True
        }
    except Exception as e:
        # Surface a targeted hint for the common ``Context size has been
        # exceeded`` failure so the operator knows which env to set rather than
        # guessing from a bare litellm BadRequestError traceback.
        _emsg = str(e)
        if "Context size has been exceeded" in _emsg or "context length" in _emsg.lower():
            logger.error(
                "RLM failed: the model's context window was exceeded. Set "
                "RLM_MODEL_CONTEXT_WINDOW to the model's "
                "actual num_ctx so prompt/completion budgets are clamped. "
                "Error: %s",
                e,
            )
        else:
            logger.error(f"Error executing RLM reasoning: {e}", exc_info=True)
        return {
            "results": f"Failed to execute RLM task: {str(e)}",
            "usage": {},
            "success": False
        }


def run_rlm_structured_sync(
    query,
    model_name: str = None,
    output_schema=None,
    tools=None,
    env_variables=None,
    session_dir: str = None,
    session_id: str = None,
    max_depth: int = None,
    max_calls_per_subagent: int = None,
    task: str = "expert",
) -> dict:
    """Run fast-rlm with structured dict input, tools, env vars, and sessions.

    This is the enhanced entry point that unlocks fast-rlm's exhaustive
    recursive capabilities (the flat ``run_rlm_task`` uses only a string
    prompt with none of these):

    - ``query`` as a ``dict``: the agent indexes ``context["key"]`` directly
      instead of re-parsing a stringified blob every turn (fast-rlm best
      practice). Falls back to the flat-string path if a str is passed.
    - ``tools``: self-contained Python callables pre-loaded into the root REPL
      so the agent pulls exactly what it needs (a specific file, a knowledge
      slice) instead of having the whole corpus in the prompt. Tools run inside
      the Pyodide REPL and reach product data via HTTP callbacks.
    - ``env_variables``: credentials/ids injected into ``os.environ`` of every
      REPL (root + subagents), never in prompts.
    - ``session_dir`` / ``session_id``: persist the root agent's REPL state
      (built indexes, helper functions, query→FINAL ledger) between calls so
      follow-up questions and later docgen sections reuse it (~2.6× cheaper
      per the docs benchmark). One live query per session dir (no concurrency).
    - ``output_schema``: a Pydantic class / generic / JSON Schema dict that
      validates the agent's ``FINAL`` and retries on failure.
    - Per-scenario recursion budgets: expert (deeper) vs docgen (moderate).

    Defensive: returns the same ``{results, usage, success}`` shape as
    ``run_rlm_task_sync``. When fast_rlm is unavailable, when ``run()`` rejects
    the new kwargs (older versions), or on any error, it degrades gracefully —
    falling back to the flat-string ``run_rlm_task_sync`` where possible.
    """
    if not _FAST_RLM_AVAILABLE:
        return {
            "results": "fast_rlm not available",
            "usage": {},
            "success": False,
        }
    try:
        config, resolved_model, base_url = _resolve_rlm_config(model_name)
    except Exception as e:
        logger.error(f"RLM config resolution failed: {e}", exc_info=True)
        return {
            "results": f"Failed to configure RLM task: {str(e)}",
            "usage": {},
            "success": False,
        }

    # Apply per-scenario recursion budgets. The flat-string path keeps the
    # conservative max_depth=2/calls=10; the structured path uses deeper
    # recursion so the agent can exhaustively explore the corpus.
    if max_depth is not None:
        config.max_depth = max_depth
    elif task == "expert":
        config.max_depth = RLM_EXPERT_MAX_DEPTH
    elif task == "docgen":
        config.max_depth = RLM_DOCGEN_MAX_DEPTH
    if max_calls_per_subagent is not None:
        config.max_calls_per_subagent = max_calls_per_subagent
    elif task == "expert":
        config.max_calls_per_subagent = RLM_EXPERT_MAX_CALLS
    elif task == "docgen":
        config.max_calls_per_subagent = RLM_DOCGEN_MAX_CALLS
    # Sub-agents inherit the root agent's tools so they can also pull files /
    # knowledge on demand (default is no inheritance per the docs).
    try:
        config.inherit_tools = True
    except Exception:  # pragma: no cover - older fast-rlm without the field
        pass
    # add_session_code_to_context=True (the default) makes follow-up queries
    # ~2.6× cheaper by carrying the prior REPL state into context.
    try:
        config.add_session_code_to_context = True
    except Exception:  # pragma: no cover - older fast-rlm without the field
        pass

    logger.info(
        f"Triggering fast-rlm STRUCTURED reasoning. task={task} Model: {resolved_model}, "
        f"base_url: {base_url}, max_depth: {config.max_depth}, "
        f"max_calls_per_subagent: {config.max_calls_per_subagent}, "
        f"tools: {len(tools) if tools else 0}, session: {session_dir or '-'}/{session_id or '-'}, "
        f"output_schema: {'yes' if output_schema else 'no'}, "
        f"query_type: {'dict' if isinstance(query, dict) else 'str'}"
    )

    # Build the kwargs for run(). Only pass the kwargs fast-rlm supports so an
    # older version that doesn't know ``tools``/``env_variables``/``sessions``/
    # ``output_schema`` still works (we probe with a TypeError fallback).
    run_kwargs = {"config": config, "verbose": True}
    if output_schema is not None:
        run_kwargs["output_schema"] = output_schema
    if tools:
        run_kwargs["tools"] = tools
    if env_variables:
        run_kwargs["env_variables"] = env_variables
    if session_dir:
        run_kwargs["session_dir"] = session_dir
    if session_id:
        run_kwargs["session_id"] = session_id

    try:
        result = run(query, **run_kwargs)
        return {
            "results": result.get("results", "No result returned"),
            "usage": result.get("usage", {}),
            "success": True
        }
    except TypeError as e:
        # Older fast-rlm that rejects one of the new kwargs (tools/
        # env_variables/session_dir/output_schema). Fall back to the flat-string
        # path so the run still produces something useful instead of crashing.
        _emsg = str(e)
        if "unexpected keyword" in _emsg.lower():
            logger.warning(
                "fast-rlm rejected a structured kwarg (%s); falling back to "
                "flat-string run_rlm_task. Upgrade fast_rlm for tools/sessions.",
                _emsg,
            )
            flat_query = query if isinstance(query, str) else str(query)
            return run_rlm_task_sync(flat_query, model_name)
        logger.error(f"Error executing structured RLM reasoning: {e}", exc_info=True)
        return {
            "results": f"Failed to execute structured RLM task: {str(e)}",
            "usage": {},
            "success": False,
        }
    except Exception as e:
        _emsg = str(e)
        if "Context size has been exceeded" in _emsg or "context length" in _emsg.lower():
            logger.error(
                "RLM structured failed: context window exceeded. Set "
                "RLM_MODEL_CONTEXT_WINDOW. Error: %s",
                e,
            )
        else:
            logger.error(f"Error executing structured RLM reasoning: {e}", exc_info=True)
        return {
            "results": f"Failed to execute structured RLM task: {str(e)}",
            "usage": {},
            "success": False,
        }

async def run_rlm_task(query: str, model_name: str = None) -> dict:
    """
    Asynchronously runs the fast-rlm task in a separate thread.
    """
    return await asyncio.to_thread(run_rlm_task_sync, query, model_name)


async def run_rlm_structured(
    query,
    model_name: str = None,
    output_schema=None,
    tools=None,
    env_variables=None,
    session_dir: str = None,
    session_id: str = None,
    max_depth: int = None,
    max_calls_per_subagent: int = None,
    task: str = "expert",
) -> dict:
    """Async wrapper for ``run_rlm_structured_sync`` (runs in a worker thread)."""
    return await asyncio.to_thread(
        run_rlm_structured_sync,
        query,
        model_name,
        output_schema,
        tools,
        env_variables,
        session_dir,
        session_id,
        max_depth,
        max_calls_per_subagent,
        task,
    )


def prewarm_rlm_background() -> None:
    """Pre-warm fast-rlm in a daemon thread at startup.

    fast-rlm's first invocation downloads npm/jsr + Pyodide packages (a slow
    one-time bootstrap). Running a tiny no-op query at boot moves that cost out
    of the first generate request. Non-fatal and non-blocking: the caller never
    waits on it, and any failure is logged.
    """
    if not _FAST_RLM_AVAILABLE:
        logger.info("Skipping RLM prewarm: fast_rlm not available.")
        return
    import threading

    def _warm() -> None:
        try:
            # Pass model_name=None so _resolve_rlm_config resolves the
            # admin-configured models.docgen.model (synced at startup by
            # bootstrap_config -> sync_runtime_settings). Passing a hardcoded
            # model here would take FIRST precedence in _resolve_rlm_config
            # (``model_name or admin_model or ...``) and override the admin
            # setting — that was the root cause of startup requests hitting
            # qwen3.6 instead of the admin-configured qwen3.8.
            res = run_rlm_task_sync("Reply with the single word: ok")
            # run_rlm_task_sync catches its own exceptions and returns
            # {"success": False}; only a truthy success flag means RLM is
            # actually usable. Reporting "completed" on failure hid the real
            # state (e.g. jsr.io/@std/yaml fetch failing) and made the logs
            # misleading.
            if res.get("success"):
                logger.info("RLM prewarm completed (RLM is usable).")
            else:
                logger.warning(
                    "RLM prewarm ran but RLM is NOT usable (success=false). "
                    "Result: %s. Codebase generation will fall back to the "
                    "standard LLM until this is resolved.",
                    res.get("results"),
                )
        except Exception as e:  # pragma: no cover - depends on live fast-rlm
            logger.warning("RLM prewarm failed (non-fatal): %s", e)

    try:
        threading.Thread(target=_warm, daemon=True, name="rlm-prewarm").start()
        logger.info("RLM prewarm started in background thread.")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not start RLM prewarm thread: %s", e)
