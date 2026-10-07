"""HLD (high-level design) generation — one pseudo-entity per product.

Builds the product-wide HLD from the ALREADY-GENERATED documentation of every
nested entity (codebases, databases, specs, knowledge pages, links) via
``collect_entity_parts`` — no direct repo/DB access. Six fixed sections, one
page each (``hld_{section}``), sharing the verification pipeline of the other
flows: mermaid repair, secret masking, LLM judge, provenance split.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from api.docgen._common import (
    _carry_page_verify_flags,
    _check_cancel,
    _checkpoint_partial_docs,
    _clean_llm_text,
    _index_in_background,
    _make_repair_llm,
    _persist_artifact,
    _product_dataset,
    _resolve_docgen_model,
    _safe_aclose,
    _safe_build_llm,
    _split_provenance_block,
    _with_verification_guard,
    collect_entity_parts,
    emit_progress,
)
from api.docgen.verification import judge_enabled, judge_section, mask_secrets
from api.formats.mermaid import run_repair_loop
from api.prompts import LANGUAGE_NAMES, load_prompt_file
from api.utils.llm_helpers import wrap_untrusted

logger = logging.getLogger(__name__)

_HLD_PROMPT_FALLBACK = (
    "Ты — системный архитектор. Напиши раздел HLD «{section_title}» продукта "
    "«{product_name}» на языке {language_name} ТОЛЬКО на основе источника "
    "фактов. Не выдумывай компоненты; заверши документ блоком "
    "`### Провенанс и проверка`.\n\n"
    "## Инструкция раздела\n{section_instructions}\n\n"
    "## Источник фактов\n<context>\n{context}\n</context>\n\n"
    "Ответ: только готовый Markdown без H1."
)

# Flat 6-section contract: page id = hld_{id}.
HLD_SECTIONS: List[Dict[str, str]] = [
    {
        "id": "overview",
        "title_ru": "Обзор продукта",
        "title_en": "Product Overview",
        "instructions": (
            "Назначение продукта: какую задачу он решает и для кого; контекст "
            "в ландшафте ИТ-систем; ключевые компоненты и их ответственность "
            "(короткий абзац или список на каждый)."
        ),
    },
    {
        "id": "architecture",
        "title_ru": "Архитектура",
        "title_en": "Architecture",
        "instructions": (
            "Компоненты продукта и внешние системы, связи и потоки управления "
            "и данных между ними. ОБЯЗАТЕЛЬНО включи одну Mermaid-диаграмму "
            "компонентов (```mermaid flowchart```); при достаточных данных — "
            "также диаграмму развёртывания."
        ),
    },
    {
        "id": "capabilities",
        "title_ru": "Capability Map",
        "title_en": "Capability Map",
        "instructions": (
            "Матрица возможностей продукта: по одному подразделу `###` на "
            "capability — название, какие сущности-источники её реализуют, "
            "зависимые возможности."
        ),
    },
    {
        "id": "integrations",
        "title_ru": "Интеграции",
        "title_en": "Integrations",
        "instructions": (
            "Внешние системы и смежные продукты: для каждой интеграции — "
            "протокол/канал, API-контракт (если есть в спецификациях), "
            "направление обмена (inbound/outbound) и передаваемые данные."
        ),
    },
    {
        "id": "data",
        "title_ru": "Данные",
        "title_en": "Data",
        "instructions": (
            "Хранилища и их роль, ключевые схемы и таблицы БД (из "
            "RE-документации), потоки данных между хранилищами и "
            "компонентами; при наличии данных — Mermaid ER-диаграмма."
        ),
    },
    {
        "id": "api",
        "title_ru": "API",
        "title_en": "API",
        "instructions": (
            "Обзор контрактов продукта — эндпоинты, каналы и события из "
            "OpenAPI/AsyncAPI-спецификаций; группируй по доменам, отмечай "
            "потребителей каждого контракта."
        ),
    },
]


def _section_title(section: Dict[str, str], language: str) -> str:
    return section["title_ru"] if language == "ru" else section["title_en"]


async def generate_hld_docs(
    entity: Any,
    product: Any,
    *,
    model: Optional[str] = None,
    language: str = "ru",
    progress: Optional[Callable[..., None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> str:
    """Generate the product HLD onto ``entity`` (HldORM); returns the markdown.

    Raises ValueError when the product has no source documentation or no LLM
    could be built (fast-fail; jobs maps it onto the failed job status).
    """
    from api.models import HldORM

    emit_progress(progress, phase="planning")

    r_model, r_base_url, r_api_key = _resolve_docgen_model(model)
    try:
        from api.utils import get_model_context_window_async

        ctx = await get_model_context_window_async(
            base_url=r_base_url, model_name=r_model, api_key=r_api_key,
            task="docgen",
        )
    except Exception:
        ctx = 8192
    budget = max(8_000, min(400_000, int(ctx * 4 * 0.55)))

    parts = collect_entity_parts(
        product.codebases,
        product.databases,
        product.specs,
        getattr(product, "knowledge_nodes", None) or [],
        product.links,
        budget_chars=budget,
    )
    # Entity docs are member-editable data: frame them so embedded prompt
    # injections stay inert (P0-8) — here and in the judge prompt below.
    context = wrap_untrusted("\n\n".join(parts))
    if not context.strip():
        raise ValueError(
            "У продукта нет исходных данных для HLD: не найдено ни одной "
            "сущности с документацией (кодовая база, БД, спецификация, база "
            "знаний, ссылки). Сначала сгенерируйте документацию вложенных "
            "сущностей."
        )

    llm = _safe_build_llm(r_model, base_url=r_base_url, api_key=r_api_key)
    if llm is None:
        raise ValueError(
            "Не удалось инициализировать LLM для задачи docgen — проверьте "
            "настройку модели (Admin → Models)."
        )

    product_name = getattr(product, "name", "") or "product"
    language_name = LANGUAGE_NAMES.get(language, language)
    old_pages = entity.pages if isinstance(entity.pages, dict) else {}
    template = load_prompt_file("hld_page.md", _HLD_PROMPT_FALLBACK, language)

    pages: Dict[str, Any] = {}
    done_sections: List[Any] = []
    docs = ""
    try:
        for i, section in enumerate(HLD_SECTIONS):
            _check_cancel(should_cancel)
            sid = section["id"]
            title = _section_title(section, language)
            started = time.monotonic()
            emit_progress(
                progress,
                phase="sections",
                sections_total=len(HLD_SECTIONS),
                sections_done=i,
                current_section=sid,
            )

            prompt = (
                template
                .replace("{product_name}", product_name)
                .replace("{section_title}", title)
                .replace("{section_instructions}", section["instructions"])
                .replace("{language_name}", language_name)
                .replace("{context}", context)
            )
            content = _clean_llm_text(await llm.generate(_with_verification_guard(prompt)))
            content, report = _split_provenance_block(content)

            emit_progress(progress, phase="verifying")
            mermaid_stats: Optional[Dict[str, int]] = None
            if "```mermaid" in content:
                repair_llm = _make_repair_llm(
                    r_model, existing=llm,
                    base_url=r_base_url, api_key=r_api_key,
                )
                try:
                    content, mermaid_stats = await run_repair_loop(content, repair_llm)
                except Exception as e:  # pragma: no cover - non-fatal guard
                    logger.warning("HLD mermaid repair failed (%s): %s", sid, e)
            content, findings = mask_secrets(content)

            judge: Optional[Dict[str, Any]] = None
            if judge_enabled():
                try:
                    verdict = await judge_section(sid, content, context, model=r_model)
                    judge = {"verdict": verdict.verdict, "issues": verdict.issues}
                except Exception as e:  # pragma: no cover - judge never blocks
                    logger.warning("HLD judge failed (%s): %s", sid, e)

            prov: Dict[str, Any] = {
                "section_id": sid,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": r_model,
                "prompt_file": "hld_page.md",
                "secrets_masked": len(findings),
                "regen": "generated",
            }
            if mermaid_stats is not None:
                prov["mermaid"] = mermaid_stats
            if judge is not None:
                prov["judge"] = judge
            if report:
                prov["report"] = mask_secrets(report)[0]

            pages[f"hld_{sid}"] = {
                "id": f"hld_{sid}",
                "title": title,
                "content": content,
                "parent": None,
                "importance": "high" if sid == "overview" else "medium",
                "relatedPages": [],
                "filePaths": [],
                "provenance": prov,
            }
            done_sections.append((title, content))
            docs = "\n\n".join(f"## {t}\n\n{c}" for t, c in done_sections)
            _checkpoint_partial_docs(entity.id, HldORM, docs, dict(pages))
            emit_progress(
                progress,
                section_done=sid,
                section_seconds=round(time.monotonic() - started, 1),
                sections_done=i + 1,
            )
    finally:
        await _safe_aclose(llm)

    _check_cancel(should_cancel)  # pre-persist: nothing written after Stop
    _carry_page_verify_flags(pages, old_pages)
    _persist_artifact(entity, docs, pages)
    _index_in_background(
        docs,
        _product_dataset(product),
        source_type="hld",
        source_id=getattr(entity, "id", None),
    )
    return docs


__all__ = ["HLD_SECTIONS", "generate_hld_docs"]
