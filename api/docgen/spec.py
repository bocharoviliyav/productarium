"""OpenAPI / AsyncAPI documentation (standard LLM + stdlib render).

Spec artifacts are parsed with stdlib json/yaml into a structured skeleton,
then enriched via the standard LLM (``refs/prompts/<kind>_doc.md``). The
enriched markdown is written back onto the spec's ``content`` and indexed into
cognee in the background.

Shared LLM/persistence helpers live in ``api.docgen._common``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from api.utils import setup_logging
from api.utils.llm_helpers import (  # noqa: E402
    cap as _cap,
    safe_replace as _safe_replace,
)
from api.formats.mermaid import run_repair_loop
from api.prompts import load_prompt_file
from api.docgen._common import (
    _with_verification_guard,
    _resolve_docgen_model,
    _llm_or_none,
    _make_repair_llm,
    _cognee_dataset,
    _index_in_background,
    _product_name,
)

# Optional YAML support. PyYAML is a transitive dep of adalflow/cognee and is
# declared explicitly in pyproject.toml; if it is ever missing we degrade
# gracefully to JSON-only parsing + a raw-text fallback.
try:  # pragma: no cover - import guard
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional at runtime
    yaml = None  # type: ignore

setup_logging()
logger = logging.getLogger(__name__)


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


def _render_raw_fallback(label: str, content: str, spec: Any) -> str:
    name = getattr(spec, "name", None) or label
    md = f"# {label}: {name}\n\n"
    md += "_(Не удалось разобрать спецификацию; показано исходное содержимое.)_\n\n"
    md += f"```yaml\n{_cap(content, 4000)}\n```"
    return md


# --------------------------------------------------------------------------- #
# OpenAPI / AsyncAPI documentation (standard LLM + stdlib render)
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
) -> str:
    """Shared flow for openapi/asyncapi: parse -> structured render + LLM enrich.

    The enriched markdown is written back onto the spec's ``content`` (specs
    carry a single content field, no pages/generated_docs) and indexed into cognee.
    """
    content = (getattr(spec, "content", "") or "").strip()
    if not content:
        raise ValueError(f"{spec_kind} spec has empty content.")

    # Resolve admin docgen config (models.docgen.*) so the LLM enrichment hits
    # the configured gateway. Per-request model override wins.
    r_model, r_base_url, r_api_key = _resolve_docgen_model(model)
    model = model or r_model

    parsed = _parse_spec(content)
    skeleton = render_skeleton(parsed) if parsed else ""
    if not skeleton:
        skeleton = _render_raw_fallback(spec_kind, content, spec)

    template = load_prompt_file(template_file, "")
    prompt = _with_verification_guard(_safe_replace(
        template,
        {
            "repo_name": _product_name(product, spec),
            "artifact_name": getattr(spec, "name", None) or spec_kind,
            "previous_content": "",
            "content": content,
        },
    ))
    llm_text = await _llm_or_none(
        prompt, model, base_url=r_base_url, api_key=r_api_key
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
            docs, _make_repair_llm(model, base_url=r_base_url, api_key=r_api_key)
        )
    except Exception as e:  # pragma: no cover - verifier must never break gen
        logger.warning("Mermaid repair loop failed for %s doc: %s", spec_kind, e)

    # Specs carry a single ``content`` field: write the enriched doc back onto it.
    try:
        spec.content = docs
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not write enriched docs onto spec: %s", e)

    _index_in_background(docs, _cognee_dataset(product))
    return docs


async def generate_openapi_docs(
    spec: Any, product: Any,
    model: Optional[str] = None, language: str = "ru",
) -> str:
    """Generate OpenAPI documentation (stdlib render + standard-LLM enrichment)."""
    return await _generate_spec_doc(
        spec, product,
        spec_kind="OpenAPI",
        template_file="openapi_doc.md",
        render_skeleton=_render_openapi_skeleton,
        model=model, language=language,
    )


async def generate_asyncapi_docs(
    spec: Any, product: Any,
    model: Optional[str] = None, language: str = "ru",
) -> str:
    """Generate AsyncAPI documentation (stdlib render + standard-LLM enrichment)."""
    return await _generate_spec_doc(
        spec, product,
        spec_kind="AsyncAPI",
        template_file="asyncapi_doc.md",
        render_skeleton=_render_asyncapi_skeleton,
        model=model, language=language,
    )
