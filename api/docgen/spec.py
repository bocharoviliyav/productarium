"""OpenAPI / AsyncAPI documentation (LangGraph flow: parse → enrich → guard).

Wave D rewrite. The flow is a LangGraph ``StateGraph`` with three nodes:

- **parse** (deterministic) — stdlib ``json``/``yaml`` parsing plus the
  structured skeleton renderers (or the raw-text fallback). No LLM.
- **enrich** (agent) — a ``create_react_agent`` agent with a read-only
  ``spec_lookup`` tool (dot-path resolution over the PARSED spec) documents
  the spec following the ``spec_enrich_task`` contract. When the agent is
  unavailable/fails, the node falls back to the Wave-A standard-LLM path
  (``_llm_or_none`` over the legacy ``refs/prompts/<kind>_doc.md`` template),
  and finally to the deterministic skeleton.
- **guard** (verification) — mermaid repair loop, secret masking
  (``api.docgen.verification``), an optional LLM judge comparing the draft
  against the spec evidence (flags, never blocks), write-back onto the
  spec's ``content`` and background indexing.

If langgraph is unavailable the same three node functions run straight-line,
so the module has no hard langgraph dependency. Shared LLM/persistence
helpers live in ``api.docgen._common``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from api.utils import setup_logging
from api.utils.llm_helpers import (  # noqa: E402
    cap as _cap,
    safe_replace as _safe_replace,
)
from api.formats.mermaid import run_repair_loop
from api.prompts import load_prompt_file
from api.docgen._common import (
    _clean_llm_text,
    _close_owned_llm,
    _with_verification_guard,
    _resolve_docgen_model,
    _llm_or_none,
    _make_repair_llm,
    _product_dataset,
    _index_in_background,
    _product_name,
    emit_progress,
)
from api.docgen.fact_fold import fold, rank_split
from api.docgen.verification import judge_enabled, judge_section, mask_secrets

# Optional YAML support. PyYAML is declared explicitly in pyproject.toml; if
# it is ever missing we degrade gracefully to JSON-only parsing + a raw-text
# fallback.
try:  # pragma: no cover - import guard
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional at runtime
    yaml = None  # type: ignore

setup_logging()
logger = logging.getLogger(__name__)

# Env kill-switch for the LLM judge stage — read at CALL time via
# ``api.docgen.verification.judge_enabled`` (same flag as the codebase flow;
# the judge is non-fatal either way).


# --------------------------------------------------------------------------- #
# Spec parsing (stdlib json/yaml) + structured renderers
# --------------------------------------------------------------------------- #
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


# How many schemas stay visible in the Schemas section; the rest go under a
# <details> disclosure (fact fold) — a big spec can carry hundreds of models.
_SCHEMAS_VISIBLE = 10


def _render_schemas_folded(schemas: dict) -> List[str]:
    """Schemas section body: top-_SCHEMAS_VISIBLE by field count visible,
    the rest under ONE disclosure («не резать, а прятать")."""
    def _field_count(item: tuple) -> int:
        schema = item[1] or {}
        try:
            return -len((schema.get("properties") or {}))
        except Exception:
            return 0

    visible, hidden = rank_split(
        list((schemas or {}).items()), key=_field_count, keep=_SCHEMAS_VISIBLE
    )
    md: List[str] = []
    for name, schema in visible:
        md.append(f"\n### {name}")
        md.extend(_schema_field_table(schema))
    if hidden:
        hidden_lines: List[str] = []
        for name, schema in hidden:
            hidden_lines.append(f"### {name}")
            hidden_lines.extend(_schema_field_table(schema))
            hidden_lines.append("")
        md.append("")
        md.extend(fold("Остальные схемы", hidden_lines, count=len(hidden)))
    return md


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
        md.extend(_render_schemas_folded(schemas))
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
        md.extend(_render_schemas_folded(schemas))
    return "\n".join(md)


def _render_raw_fallback(label: str, content: str, spec: Any) -> str:
    name = getattr(spec, "name", None) or label
    md = f"# {label}: {name}\n\n"
    md += "_(Не удалось разобрать спецификацию; показано исходное содержимое.)_\n\n"
    md += f"```yaml\n{_cap(content, 4000)}\n```"
    return md


# --------------------------------------------------------------------------- #
# spec_lookup tool (dot-path resolution over the parsed spec)
# --------------------------------------------------------------------------- #
class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<missing>"


_MISSING = _Missing()


def _lookup_path(obj: Any, dotted: str) -> Any:
    """Resolve a dotted path against the parsed spec.

    Keys may themselves contain dots (e.g. ``paths./users/{id}.get``), so at
    every level the longest join of the remaining parts that matches a key
    wins before falling back to a single-part match. List indices are
    supported (``channels.0``).
    """
    cur = obj
    parts = [p for p in (dotted or "").split(".") if p]
    while parts:
        if isinstance(cur, dict):
            if parts[0] in cur:
                cur = cur[parts[0]]
                parts = parts[1:]
                continue
            for i in range(len(parts), 1, -1):
                cand = ".".join(parts[:i])
                if cand in cur:
                    cur = cur[cand]
                    parts = parts[i:]
                    break
            else:
                return _MISSING
            continue
        if isinstance(cur, list):
            try:
                idx = int(parts[0])
            except ValueError:
                return _MISSING
            if idx < 0 or idx >= len(cur):
                return _MISSING
            cur = cur[idx]
            parts = parts[1:]
            continue
        return _MISSING
    return cur


def make_spec_lookup_tool(parsed: dict) -> Any:
    """Build the read-only ``spec_lookup`` tool over the parsed spec dict."""
    from langchain_core.tools import tool

    @tool
    def spec_lookup(path: str) -> str:
        """Look up an exact value from the parsed API spec by dotted path.

        Examples: ``info``, ``paths./users.get``, ``components.schemas.User``.
        Returns the JSON fragment (pretty-printed, capped), or an ERROR line
        when the path does not exist.
        """
        value = _lookup_path(parsed, path or "")
        if value is _MISSING:
            return f"ERROR: path not found in the spec: {path}"
        try:
            return _cap(json.dumps(value, ensure_ascii=False, indent=2, default=str), 8_000)
        except Exception as e:  # pragma: no cover - defensive
            return f"ERROR: could not serialize value at {path}: {e}"

    return spec_lookup


# --------------------------------------------------------------------------- #
# Enrichment agent (react agent over spec_lookup)
# --------------------------------------------------------------------------- #
_SPEC_AGENT_SYSTEM_FALLBACK = (
    "You are a technical writer agent documenting an API specification with "
    "the spec_lookup tool. Verify every detail (paths, methods, schemas, "
    "fields, required flags) with spec_lookup before writing it down; never "
    "invent endpoints or fields. Cite values exactly as they appear in the "
    "spec. Write the documentation in {language_name} (technical terms in "
    "English). Finish with ONLY the finished documentation Markdown as your "
    "final message."
)
_SPEC_ENRICH_TASK_FALLBACK = (
    "Document the {spec_kind} specification `{artifact_name}`.\n\n"
    "<skeleton>\n{skeleton}\n</skeleton>\n\n"
    "<source_spec>\n{content}\n</source_spec>\n\n"
    "Produce complete, well-structured Markdown documentation. Your final "
    "message must contain ONLY the finished document."
)


def _resolve_language_name(language: str) -> str:
    from api.prompts import LANGUAGE_NAMES

    return LANGUAGE_NAMES.get(language, language)


def _build_spec_agent_system_prompt(spec_kind: str, language: str) -> str:
    template = load_prompt_file("spec_agent_system.md", _SPEC_AGENT_SYSTEM_FALLBACK)
    return _with_verification_guard(
        template.replace("{spec_kind}", spec_kind)
        .replace("{language_name}", _resolve_language_name(language))
    )


def build_spec_enrich_task(
    *,
    spec_kind: str,
    artifact_name: str,
    skeleton: str,
    content: str,
) -> str:
    template = load_prompt_file("spec_enrich_task.md", _SPEC_ENRICH_TASK_FALLBACK)
    for var, value in (
        ("spec_kind", spec_kind),
        ("artifact_name", artifact_name),
        ("skeleton", skeleton or "(unavailable)"),
        ("content", _cap(content, 50_000)),
    ):
        template = template.replace("{" + var + "}", str(value))
    return template


def _build_spec_agent(
    parsed: Optional[dict],
    *,
    spec_kind: str,
    language: str = "ru",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Build the react enrichment agent.

    Returns ``(agent, chat_model)``; ``(None, None)`` when the agent path is
    unavailable (langgraph/llm factory import failure, model build error, or
    an unparseable spec — the lookup tool needs the parsed dict).
    """
    if not isinstance(parsed, dict) or not parsed:
        return None, None
    try:
        from langgraph.prebuilt import create_react_agent

        from api.llm.client import build_chat_model
    except Exception as e:  # pragma: no cover - import guard
        logger.warning("spec react agent unavailable (%s); standard-LLM path.", e)
        return None, None
    try:
        chat = build_chat_model(model=model, base_url=base_url, api_key=api_key)
        agent = create_react_agent(
            model=chat,
            tools=[make_spec_lookup_tool(parsed)],
            prompt=_build_spec_agent_system_prompt(spec_kind, language),
        )
        return agent, chat
    except Exception as e:  # pragma: no cover - depends on config validity
        logger.warning("spec react agent could not be built (%s).", e)
        return None, None


async def _run_spec_agent(agent: Any, task_prompt: str) -> str:
    """Run the react agent once; '' on any failure (never raises)."""
    from langchain_core.messages import HumanMessage

    from api.docgen.codebase import _final_agent_text

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=task_prompt)]},
            config={"recursion_limit": 40},
        )
    except Exception as e:  # pragma: no cover - depends on live model
        logger.warning("spec react agent run failed: %s", e)
        return ""
    return _clean_llm_text(_final_agent_text(result)) or ""


# --------------------------------------------------------------------------- #
# LangGraph flow: parse (deterministic) → enrich (agent) → guard (verification)
# --------------------------------------------------------------------------- #
from typing import TypedDict  # noqa: E402  (kept next to the state it defines)


class SpecDocState(TypedDict, total=False):
    spec: Any
    product: Any
    spec_kind: str
    template_file: str
    render_skeleton: Any
    artifact_name: str
    content: str
    language: str
    model: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    # node outputs
    parsed: Optional[dict]
    skeleton: str
    docs: str
    enrich_source: str


def _parse_node(state: SpecDocState) -> Dict[str, Any]:
    """Deterministic node: stdlib parse + skeleton render (no LLM)."""
    content = state.get("content", "")
    parsed = _parse_spec(content)
    render_skeleton = state.get("render_skeleton")
    skeleton = render_skeleton(parsed) if (parsed and render_skeleton) else ""
    if not skeleton:
        skeleton = _render_raw_fallback(state.get("spec_kind", "Spec"), content, state.get("spec"))
    return {"parsed": parsed, "skeleton": skeleton}


async def _enrich_node(state: SpecDocState) -> Dict[str, Any]:
    """Agent node: react agent over ``spec_lookup`` → standard LLM → skeleton."""
    spec_kind = state.get("spec_kind", "Spec")
    skeleton = state.get("skeleton", "")
    content = state.get("content", "")
    model = state.get("model")
    base_url = state.get("base_url")
    api_key = state.get("api_key")

    # 1) react agent (spec_lookup tool over the parsed spec)
    agent, chat = _build_spec_agent(
        state.get("parsed"),
        spec_kind=spec_kind,
        language=state.get("language", "ru"),
        model=model, base_url=base_url, api_key=api_key,
    )
    if agent is not None:
        try:
            task = build_spec_enrich_task(
                spec_kind=spec_kind,
                artifact_name=state.get("artifact_name", spec_kind),
                skeleton=skeleton,
                content=content,
            )
            text = await _run_spec_agent(agent, task)
        finally:
            from api.docgen.codebase import _close_chat_client

            await _close_chat_client(chat)
        if text:
            return {"docs": text, "enrich_source": "agent"}

    # 2) Wave-A standard-LLM path (legacy prompt contract, guard-wrapped)
    template = load_prompt_file(state.get("template_file", ""), "")
    prompt = _with_verification_guard(_safe_replace(
        template,
        {
            "repo_name": _product_name(state.get("product"), state.get("spec")),
            "artifact_name": state.get("artifact_name", spec_kind),
            "previous_content": "",
            "content": content,
        },
    ))
    llm_text = await _llm_or_none(prompt, model, base_url=base_url, api_key=api_key)
    if llm_text:
        return {"docs": llm_text, "enrich_source": "standard-llm"}

    # 3) deterministic skeleton
    return {"docs": skeleton, "enrich_source": "skeleton"}


async def _guard_node(state: SpecDocState) -> Dict[str, Any]:
    """Verification node: mermaid repair → secret masking → judge → persist."""
    docs = state.get("docs", "")
    model = state.get("model")
    base_url = state.get("base_url")
    api_key = state.get("api_key")
    spec = state.get("spec")
    product = state.get("product")

    # Validate + repair any mermaid diagrams before persisting. Non-fatal by
    # contract: a failed repair loop returns/leaves the docs unchanged. The
    # repair client is BUILT here (no shared fallback LLM in this flow), so it
    # is closed here too.
    repair_llm = None
    try:
        repair_llm = _make_repair_llm(model, base_url=base_url, api_key=api_key)
        docs, _mstats = await run_repair_loop(docs, repair_llm)
    except Exception as e:  # pragma: no cover - verifier must never break gen
        logger.warning("Mermaid repair loop failed for %s doc: %s", state.get("spec_kind"), e)
    finally:
        await _close_owned_llm(repair_llm)

    # Secret masking (deterministic guard; the masked text is persisted).
    masked, findings = mask_secrets(docs or "")
    if findings:
        logger.warning(
            "spec docgen guard: masked %d secret-like value(s) (%s)",
            len(findings), ", ".join(sorted(set(findings))),
        )
    docs = masked

    # LLM judge (flags, never blocks). Only model-generated docs are judged —
    # the deterministic skeleton is grounded in the spec by construction.
    if judge_enabled() and state.get("enrich_source") in ("agent", "standard-llm"):
        evidence = (state.get("skeleton") or "") + "\n\n" + _cap(state.get("content", ""), 20_000)
        try:
            verdict = await judge_section(
                state.get("spec_kind", "spec"), docs, evidence, model=model
            )
            if verdict.verdict != "consistent" and verdict.issues:
                logger.warning(
                    "spec docgen judge [%s]: %s — %s",
                    state.get("spec_kind"), verdict.verdict,
                    "; ".join(verdict.issues[:3]),
                )
        except Exception as e:  # pragma: no cover - judge must never break gen
            logger.warning("spec docgen judge failed: %s", e)

    # Specs carry a single ``content`` field: write the enriched doc back.
    try:
        spec.content = docs
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not write enriched docs onto spec: %s", e)

    _index_in_background(
        docs, _product_dataset(product),
        source_type="spec", source_id=getattr(spec, "id", None),
    )
    return {"docs": docs}


_SPEC_GRAPH: Optional[Any] = None


def _get_spec_graph() -> Optional[Any]:
    """Compile (once) the parse → enrich → guard StateGraph; None w/o langgraph."""
    global _SPEC_GRAPH
    if _SPEC_GRAPH is not None:
        return _SPEC_GRAPH
    try:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(SpecDocState)
        graph.add_node("parse", _parse_node)
        graph.add_node("enrich", _enrich_node)
        graph.add_node("guard", _guard_node)
        graph.add_edge(START, "parse")
        graph.add_edge("parse", "enrich")
        graph.add_edge("enrich", "guard")
        graph.add_edge("guard", END)
        _SPEC_GRAPH = graph.compile()
    except Exception as e:  # pragma: no cover - import guard
        logger.warning("langgraph unavailable (%s); spec flow runs straight-line.", e)
        _SPEC_GRAPH = None
    return _SPEC_GRAPH


# --------------------------------------------------------------------------- #
# OpenAPI / AsyncAPI documentation (LangGraph flow)
# --------------------------------------------------------------------------- #
async def _generate_spec_doc(
    spec: Any,
    product: Any,
    *,
    spec_kind: str,
    template_file: str,
    render_skeleton,
    model: Optional[str],
    language: str,
    progress: Optional[Any] = None,
) -> str:
    """Shared flow for openapi/asyncapi: parse → enrich (agent) → guard.

    The enriched markdown is written back onto the spec's ``content`` (specs
    carry a single content field, no pages/generated_docs) and indexed into
    the active memory backend.
    """
    content = (getattr(spec, "content", "") or "").strip()
    if not content:
        raise ValueError(f"{spec_kind} spec has empty content.")

    # Resolve admin docgen config (models.docgen.*) so the LLM enrichment hits
    # the configured gateway. Per-request model override wins.
    r_model, r_base_url, r_api_key = _resolve_docgen_model(model)
    model = model or r_model

    state: SpecDocState = {
        "spec": spec,
        "product": product,
        "spec_kind": spec_kind,
        "template_file": template_file,
        "render_skeleton": render_skeleton,
        "artifact_name": getattr(spec, "name", None) or spec_kind,
        "content": content,
        "language": language,
        "model": model,
        "base_url": r_base_url,
        "api_key": r_api_key,
    }

    emit_progress(progress, phase="planning")
    graph = _get_spec_graph()
    if graph is not None:
        try:
            emit_progress(
                progress, phase="sections", sections_total=1,
                current_section="enrichment",
            )
            t_enrich = time.monotonic()
            result = await graph.ainvoke(state)
            emit_progress(
                progress, section_done="enrichment",
                section_seconds=time.monotonic() - t_enrich,
            )
            emit_progress(progress, phase="verifying")
            emit_progress(progress, phase="indexing")
            return result.get("docs", "")
        except Exception as e:  # pragma: no cover - graph failure safety net
            logger.warning(
                "spec LangGraph flow failed (%s); running nodes straight-line.", e
            )

    # Straight-line fallback: the SAME node functions, no graph runtime.
    emit_progress(
        progress, phase="sections", sections_total=1,
        current_section="enrichment",
    )
    t_enrich = time.monotonic()
    state.update(_parse_node(state))
    state.update(await _enrich_node(state))
    state.update(await _guard_node(state))
    emit_progress(
        progress, section_done="enrichment",
        section_seconds=time.monotonic() - t_enrich,
    )
    emit_progress(progress, phase="verifying")
    emit_progress(progress, phase="indexing")
    return state.get("docs", "")


async def generate_openapi_docs(
    spec: Any, product: Any,
    model: Optional[str] = None, language: str = "ru",
    progress: Optional[Any] = None,
) -> str:
    """Generate OpenAPI documentation (LangGraph parse → enrich → guard)."""
    return await _generate_spec_doc(
        spec, product,
        spec_kind="OpenAPI",
        template_file="openapi_doc.md",
        render_skeleton=_render_openapi_skeleton,
        model=model, language=language,
        progress=progress,
    )


async def generate_asyncapi_docs(
    spec: Any, product: Any,
    model: Optional[str] = None, language: str = "ru",
    progress: Optional[Any] = None,
) -> str:
    """Generate AsyncAPI documentation (LangGraph parse → enrich → guard)."""
    return await _generate_spec_doc(
        spec, product,
        spec_kind="AsyncAPI",
        template_file="asyncapi_doc.md",
        render_skeleton=_render_asyncapi_skeleton,
        model=model, language=language,
        progress=progress,
    )
