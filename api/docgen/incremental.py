"""SPEC-010 incremental documentation updates — change detection.

``detect_changes`` compares the artifact's stored ``source_state`` baseline
against the LIVE source — the git remote tip (codebase, fetch without a
re-clone or working-tree reset), a fresh cache-bypassing catalog
introspection (database) or the spec content hash — and maps the changed
files/objects onto documentation pages through provenance. One conservative
LLM call may ADD pages to the deterministic impact set (never remove); a
failed or unparseable call keeps the deterministic set.

``changed`` semantics: ``True`` (source moved), ``False`` (up to date) or
``None`` (cannot verify — no baseline/clone/remote). The impact summary is
consumed by the detect job result and the incremental update UI; the update
itself reuses the existing ``force_pages`` generation path.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from api.utils import setup_logging
from api.utils.llm_helpers import cap as _cap
from api.docgen._common import (
    _llm_or_none,
    _resolve_docgen_model,
    extract_json,
    utc_now_iso,
)

setup_logging()
logger = logging.getLogger(__name__)

# Prompt shape guards for the single LLM impact-expansion call.
_IMPACT_ITEMS_LISTED = 200
_IMPACT_PAGES_LISTED = 300
_IMPACT_PROMPT_CAP = 12_000


def _entity_pages(entity: Any) -> Dict[str, Dict[str, Any]]:
    pages = getattr(entity, "pages", None)
    if not isinstance(pages, dict):
        return {}
    return {pid: p for pid, p in pages.items() if isinstance(p, dict)}


def _content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Per-type source checks
# --------------------------------------------------------------------------- #
async def _detect_codebase(
    entity: Any,
) -> Tuple[Optional[bool], str, Optional[List[str]]]:
    """(changed, reason, changed_files); ``changed_files`` None = unknown."""
    from api.repositories.documents import managed_repo_dir

    repo_url = (getattr(entity, "repo_url", "") or "").strip()
    if not repo_url:
        return None, "У артефакта не задан repo_url.", None
    repo_type = getattr(entity, "repo_type", None) or "github"
    repo_dir = managed_repo_dir(repo_url, repo_type)
    if not repo_dir or not os.path.isdir(repo_dir):
        return (
            None,
            "Локальный клон репозитория не найден; выполните полную генерацию.",
            None,
        )
    baseline = ((getattr(entity, "source_state", None) or {}).get("baseline")) or ""
    if not baseline:
        return (
            None,
            "Базовый коммит неизвестен (документация создана до инкрементальных "
            "обновлений); выполните полную генерацию.",
            None,
        )

    from api.repositories.product_repo import get_codebase_token

    token = get_codebase_token(entity)
    try:
        from api.config.settings import resolve_git_token

        resolved = resolve_git_token(repo_url, repo_type)
        if resolved:
            token = resolved
    except Exception as e:  # pragma: no cover - import-safe
        logger.debug("resolve_git_token failed for %s: %s", repo_url, e)

    from api.clients.git import diff_commit_names, fetch_remote_tip

    try:
        tip = await asyncio.to_thread(
            fetch_remote_tip, repo_dir, repo_url, repo_type, token
        )
    except ValueError as e:
        return None, f"Не удалось получить состояние удалённого репозитория: {e}", None
    if tip == baseline:
        return False, f"Удалённый HEAD не изменился ({tip[:10]}).", []
    try:
        files = await asyncio.to_thread(diff_commit_names, repo_dir, baseline, tip)
    except ValueError as e:
        logger.info(
            "codebase diff %s..%s unavailable (%s); impact unknown.",
            baseline[:8], tip[:8], e,
        )
        return (
            True,
            f"Коммит изменился ({baseline[:10]} → {tip[:10]}), но список изменённых "
            "файлов недоступен — потребуется полная регенерация.",
            None,
        )
    return True, f"Коммит изменился ({baseline[:10]} → {tip[:10]}).", files


async def _detect_database(
    entity: Any, product: Any
) -> Tuple[bool, str, List[str], List[Dict[str, Any]]]:
    """(changed, reason, changed_objects, impacted units).

    Re-introspects the live catalog (bypassing the disk cache, which holds
    the state the current docs were generated FROM) and fingerprints every
    documented object against the stored page provenance. FK-dependent
    objects are caught automatically: touching edges ride in the evidence
    fingerprint, so a new FK marks BOTH sides changed. Raises ValueError on
    an unusable MCP surface (an honest detect-job failure, same as generate).
    """
    from api.docgen.database import (
        _subpage_units,
        database_source_baseline,
        fresh_database_info,
        source_fingerprint,
    )

    product_id = str(
        getattr(product, "id", None) or getattr(product, "product_id", None) or ""
    )
    info = await fresh_database_info(entity, product_id)
    pages = _entity_pages(entity)
    if ((getattr(entity, "source_state", None) or {}).get("baseline") or "") == (
        database_source_baseline(info)
    ):
        return False, "Каталог базы данных не изменился.", [], []

    changed_objects: List[str] = []
    impacted: List[Dict[str, Any]] = []
    for cat, full, pid in _subpage_units(info):
        fingerprint = source_fingerprint(info, cat, full)
        page = pages.get(pid)
        stored = ((page or {}).get("provenance") or {}).get("object_fingerprint")
        if stored == fingerprint:
            continue
        changed_objects.append(full)
        impacted.append({
            "id": pid,
            "title": (page or {}).get("title") or full,
            "new": page is None,
        })
    reason = f"Каталог изменился: затронуто объектов — {len(changed_objects)}."
    return True, reason, changed_objects, impacted


def _detect_spec(entity: Any) -> Tuple[Optional[bool], str]:
    baseline = (getattr(entity, "source_state", None) or {}).get("baseline") or ""
    if not baseline:
        return (
            None,
            "Базовое состояние контента неизвестно (документация создана до "
            "инкрементальных обновлений); выполните полную генерацию.",
        )
    if _content_hash(getattr(entity, "content", "") or "") == baseline:
        return False, "Контент спецификации не изменился."
    return (
        True,
        "Контент спецификации изменился — потребуется полная регенерация.",
    )


# --------------------------------------------------------------------------- #
# LLM impact expansion (one call, add-only)
# --------------------------------------------------------------------------- #
def _parse_id_array(text: str) -> List[str]:
    """Id strings from an LLM answer (thin wrapper over ``_common.extract_json``;
    the prompt demands a bare array, non-list answers contribute no ids)."""
    data = extract_json(text)
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, str)]


async def _expand_impact(
    changed_items: List[str],
    items_label: str,
    candidates: List[Tuple[str, str, str]],
    model: Optional[str],
) -> Set[str]:
    """One LLM call: pages ADDITIONALLY impacted beyond provenance matches.

    ``candidates`` = (page_id, title, sources) of pages outside the
    deterministic core. The LLM can only add valid candidate ids; any
    failure/empty answer keeps the deterministic set.
    """
    if not changed_items or not candidates:
        return set()
    tail = (
        "Какие из перечисленных страниц ДОПОЛНИТЕЛЬНО затронуты этими "
        "изменениями (обзорные или связанные страницы, зависимые объекты)?\n"
        "Ответь ТОЛЬКО JSON-массивом id строк (можно пустым): [\"page_x\", ...]"
    )
    # Each list is capped on its OWN: a whole-prompt cap would slice the
    # tail — the answer-format instruction — and the expansion would degrade
    # to prose parsing on exactly the largest documentations.
    items_block = _cap(
        "\n".join(f"- {item}" for item in changed_items[:_IMPACT_ITEMS_LISTED]),
        _IMPACT_PROMPT_CAP // 3,
    )
    pages_block = _cap(
        "\n".join(
            f"- {pid} | {title} | {sources}"
            for pid, title, sources in candidates[:_IMPACT_PAGES_LISTED]
        ),
        _IMPACT_PROMPT_CAP // 3,
    )
    prompt = "\n".join([
        "Документация проекта обновляется инкрементально.",
        f"Изменённые {items_label}:",
        items_block,
        "",
        "Страницы документации (id | заголовок | исходные объекты):",
        pages_block,
        "",
        tail,
    ])
    r_model, r_base_url, r_api_key = _resolve_docgen_model(model)
    text = await _llm_or_none(
        prompt, r_model, base_url=r_base_url, api_key=r_api_key
    )
    if not text:
        return set()
    valid = {pid for pid, _, _ in candidates}
    added = {pid for pid in _parse_id_array(text) if pid in valid}
    if added:
        logger.info(
            "impact expansion: LLM added %d page(s): %s",
            len(added), ", ".join(sorted(added)),
        )
    return added


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
async def detect_changes(
    entity: Any,
    product: Any,
    entity_type: str,
    *,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect source changes for one artifact and plan the impacted pages.

    Returns a plain dict (detect job result / UI summary). Updates the
    entity's ``source_state["checked_at"]`` in memory; the caller (job
    worker) persists it. Baselines are only rewritten by a successful
    generation, never here.
    """
    entity_type = (entity_type or "").lower()
    checked_at = utc_now_iso()
    changed: Optional[bool] = None
    reason = ""
    changed_files: Optional[List[str]] = None
    changed_objects: Optional[List[str]] = None
    impacted: List[Dict[str, Any]] = []
    candidates: List[Tuple[str, str, str]] = []
    items_label = ""

    if entity_type == "codebase":
        changed, reason, changed_files = await _detect_codebase(entity)
        if changed and changed_files:
            pages = _entity_pages(entity)
            changed_set = set(changed_files)
            core: Set[str] = set()
            for pid, page in sorted(pages.items()):
                sources = set((page.get("provenance") or {}).get("source_files") or [])
                if sources & changed_set:
                    core.add(pid)
                    impacted.append({
                        "id": pid,
                        "title": page.get("title") or pid,
                        "new": False,
                    })
            candidates = [
                (
                    pid,
                    page.get("title") or pid,
                    ", ".join(
                        (page.get("provenance") or {}).get("source_files") or []
                    )[:200],
                )
                for pid, page in sorted(pages.items())
                if pid not in core
            ]
            items_label = "файлы репозитория"
    elif entity_type == "database":
        changed, reason, changed_objects, impacted = await _detect_database(
            entity, product
        )
        if changed and changed_objects:
            pages = _entity_pages(entity)
            core = {p["id"] for p in impacted}
            # Root pages (overview / tables / category roots / schema groups)
            # have no per-object provenance — only the LLM pass can flag them.
            candidates = [
                (pid, page.get("title") or pid, "")
                for pid, page in sorted(pages.items())
                if pid not in core
            ]
            items_label = "объекты базы данных"
    elif entity_type == "spec":
        changed, reason = _detect_spec(entity)
    else:
        raise ValueError(f"Unsupported entity type for change detection: {entity_type}")

    if changed and candidates:
        added = await _expand_impact(
            changed_objects or changed_files or [], items_label, candidates, model
        )
        pages = _entity_pages(entity)
        for pid in sorted(added):
            impacted.append({
                "id": pid,
                "title": pages.get(pid, {}).get("title") or pid,
                "new": False,
            })
        impacted.sort(key=lambda p: p["id"])

    state = getattr(entity, "source_state", None)
    if isinstance(state, dict) and state.get("baseline"):
        try:
            entity.source_state = {**state, "checked_at": checked_at}
        except Exception:  # pragma: no cover - defensive over attribute setting
            logger.debug("could not update source_state.checked_at")

    result: Dict[str, Any] = {
        "entity_type": entity_type,
        "entity_id": str(getattr(entity, "id", "") or ""),
        "changed": changed,
        "reason": reason,
        "checked_at": checked_at,
        "impacted_pages": impacted,
    }
    if changed_files is not None:
        result["changed_files"] = changed_files
    if changed_objects is not None:
        result["changed_objects"] = changed_objects
    return result
