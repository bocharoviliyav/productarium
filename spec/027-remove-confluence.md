# SPEC-027: Полное удаление Confluence

Статус: planned · Волна 14 · Приоритет: высокий

## Проблема

Confluence-интеграция не нужна: убрать полностью — из админки, логики кода и тестов. Импорт страниц в Базу знаний в перспективе возродится как MCP-сервер со scope «база знаний» (SPEC-028), сейчас удаляем всё.

## Решение (полная инвентаризация)

Backend:
- Удалить целиком `api/integrations/confluence.py` (коннектор Cloud v2/Server v1: list_spaces/pull/attachments) и `tests/unit/test_integrations_confluence.py`.
- `api/integrations/__init__.py` + `api/integrations/base.py`: убрать импорт/авто-регистрацию и docstring-упоминания confluence.
- `api/config/settings.py`: удалить `get_confluence_creds()` (~543-555) + строку docstring (~14). Хранящийся секрет `confluence.token` в settings store не трогаем (YAGNI).
- `api/routers/admin.py`: убрать "confluence" из VALID_GROUPS (~95), ветку GET-группы (~650-651), `_redact_confluence_creds` (~229), диспетч теста (~936-937), `_test_confluence` (~1099-1104), упоминания в docstring (~4, 14, 629, 925-927).
- `api/routers/public.py`: `_default_push_target` (~315-335) — оставить только git-логику; убрать import `get_confluence_creds` (~40).
- `api/expert/knowledge.py` (~124-143): удалить live-Confluence fallback.
- `api/expert/prompt.py` (~210-215): убрать предложение про Confluence из промпта.
- `api/docgen/codebase.py` (~2680-2689): удалить confluence-ветку сбора product-knowledge контекста.
- `api/routers/integrations.py` (~53): описание поля connector — без примера confluence.
- Комментарии/docstring: `api/main.py` (~12), `api/schemas.py` (~14), `api/models.py` (~14, 569), `api/docgen/database.py` (~47, 3751), `api/config/ssl.py` (~206), `api/routers/knowledge.py` (~3, 286 — «Confluence-like» переформулировать нейтрально), `api/agents/tools.py` (~692 — аналог.), `api/formats/markitdown.py` (~3).

Frontend:
- `src/app/admin/page.tsx`: удалить ConfluenceSection (~748-830+), пункт навигации, типы (AdminTab/ConfluenceGroupResponse).
- `src/components/knowledge/KnowledgeTree.tsx`: удалить режим импорта страницы по URL из Confluence (~261-262, 305-322, 358-393, 623-677 — выбор коннектора, поле URL, submit `connector: "confluence"`).
- `src/lib/types.ts` (~395): тип-упоминание.
- i18n: ключи confluence* в `src/messages/ru.json` + `en.json`.

Тесты: обновить `tests/unit/test_routers_admin.py`, `test_routers_public.py`, `test_routers_integrations.py`, `test_settings_store.py`, `test_wave_f_security.py`, `test_expert_knowledge.py`, `test_docgen_codebase.py`, `test_products_repo.py` и интеграционные `test_foundation.py`, `test_admin_public.py`, `test_integrations.py` — убрать confluence-кейсы.

## Критерии приёмки

- `grep -ri confluence` по `api/ tests/ src/` (без `__pycache__`, логов, README) — 0 попаданий.
- GET/PUT /api/admin/confluence и тест-роут больше не существуют (404), навигации и секции в UI нет.
- Полный hermetic-сьют + `bun run lint` / `bun run build` зелёные.

## Затрагиваемые файлы

См. инвентаризацию выше (~20 файлов + тесты).
