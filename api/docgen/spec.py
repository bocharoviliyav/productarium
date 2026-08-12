"""OpenAPI / AsyncAPI / Testcase documentation (standard LLM + stdlib render).

Split out of the former ``api/artifact_docgen.py`` (Step 4). Spec artifacts are
parsed with stdlib json/yaml into a structured skeleton, then enriched via the
standard LLM (``refs/prompts/<kind>_doc.md``). Test-case artifacts render an
Allure URL as a LINK only (never fetched). All paths index into cognee and
persist ``generated_docs`` + ``pages``.

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
    _persist_artifact,
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

    from api.utils import get_model_context_window, clamp_text_by_tokens
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

    from api.utils import get_model_context_window, clamp_text_by_tokens
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
