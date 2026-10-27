# SPEC-028: Области применения MCP-серверов (scopes)

Статус: planned · Волна 14 · Приоритет: высокий

## Проблема

Любой привязанный к продукту MCP-сервер сегодня доступен всюду (эксперт-чат, Deep Research, fallback интроспекции БД). Нужен множественный признак «область применения» — кодовая база | база данных | база знаний | эксперт: сервер доступен только в выбранных областях. Зашитые пресеты БД (dbhub, oracle-mcp-server) должны быть помечены «база данных»; новый сервер со scope «база данных» можно выбрать при добавлении БД-артефакта; «кодовая база» — инструменты доступны агентам кодогенерации; «эксперт» — в чате с экспертом.

## Решение

- Модель: `McpServerORM.scopes` (JSON-список строк) + шим `("mcp_servers", "scopes", "JSON")` в `api/db.py::_SCHEMA_COLUMN_SHIMS`; константа `MCP_SCOPES = ("codebase", "database", "knowledge", "expert")` в `api/models.py`.
- Бэкфилл (идемпотентно при старте, рядом с `_ensure_schema_columns`, `WHERE scopes IS NULL`): строки с `preset_key IS NOT NULL` → `["database"]`; прочие существующие → `["expert", "database"]` (сохраняет сегодняшнее поведение: эксперт-чат + fallback интроспекции БД).
- Потребители (фильтрация по scope):
  - `api/mcp/manager.py::get_tools_for_product(..., scope=None)` — опциональный фильтр по `server.scopes`; `gather_mcp_agent_tools(product_id)` вызывает с `scope="expert"` (эксперт-чат и Deep Research).
  - `api/docgen/database.py::_resolve_mcp_tools` (~281) fallback-ветка — только `scope="database"`; `_validate_mcp_server_pin` (`api/routers/databases.py` ~146) — 400, если у сервера нет scope "database".
  - НОВОЕ: юнит-агенты codebase-докгена получают инструменты codebase-scoped серверов: `api/docgen/codebase.py` (~1894, `tools=repo_tools`) — добавить `plain_agent_tools(gather(..., scope="codebase"))` (пустой список — ничего не меняется); один сбойный вызов инструмента не роняет юнит (уже обеспечивается `api/docgen/_common.py::plain_agent_tools`).
  - «knowledge» — хранится и редактируется, потребитель появится позже (импорт страниц БЗ через MCP вместо удалённого Confluence, SPEC-027).
- `api/routers/mcp_admin.py`: CRUD — поле `scopes` обязательно (непустой подсет `MCP_SCOPES`, дедуп), возвращается в GET; зашитые preset-строки остаются immutable (как сейчас).
- Админ-UI (`src/app/admin/page.tsx` McpSection ~1634): мультивыбор scopes чекбокс-чипами при создании/редактировании; в списке серверов — теги. i18n: «Кодовая база» / «База данных» / «База знаний» / «Эксперт» (ru+en).
- Модалка добавления БД (`src/app/products/[productId]/page.tsx` ~1044-1129): при наличии у продукта привязанных enabled серверов со scope "database" — переключатель «Встроенный (DSN) | MCP-сервер»; при выборе сервера DSN не требуется, POST шлёт `mcp_server_id` (бэкенд уже поддерживает: `databases.py` ~186-187). Список серверов — из bindings API + scopes из admin registry GET.
- `src/components/mcp/McpServersPanel.tsx`: теги scopes у серверов.

## Критерии приёмки

- Пресеты dbhub/oracle — только «database»; эксперт-чат не получает серверы без scope expert; пин и fallback интроспекции БД — только database-scoped (пин другого — 400); codebase-агенты получают codebase-scoped инструменты.
- Создать/отредактировать сервер без scopes нельзя (400 / UI-валидация); scopes видны тегами в админке и панели продукта.
- Полный hermetic-сьют + lint/build зелёные; тесты: бэкфилл, фильтры get_tools_for_product, пин-валидация, валидация CRUD, инъекция инструментов в codebase-агента.

## Затрагиваемые файлы

`api/models.py`, `api/db.py`, `api/mcp/manager.py`, `api/docgen/database.py`, `api/docgen/codebase.py`, `api/routers/databases.py`, `api/routers/mcp_admin.py`, `src/app/admin/page.tsx`, `src/components/mcp/McpServersPanel.tsx`, `src/app/products/[productId]/page.tsx`, `src/lib/types.ts`, `src/messages/ru.json`+`en.json`, тесты.
