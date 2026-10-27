# Productarium — спеки доработок

Каждая спека = отдельная задача. Волны выполняются последовательно, запуск волны — по отбивке владельца. Внутри волны спеки можно параллелить отдельными сабагентами, если они не трогают одни файлы.

## Волны

- **В1 — быстрые фиксы**: SPEC-005 (контекст 200k), SPEC-006 (чистка RLM), SPEC-002 (редактирование продукта), SPEC-003 (failed to fetch)
- **В2 — качество документации**: SPEC-004 (гейт качества страниц, `<tool_call>`)
- **В3 — ролевая модель**: SPEC-001 (роли + маппинг)
- **В4 — автоматизация**: SPEC-010 (инкрементальные обновления) → SPEC-009 (крон-генерация; зависит от 010)
- **В5 — база знаний**: SPEC-008 (графы знаний + MCP)
- **В6 — ревью**: SPEC-007 (три прохода ревью всей кодовой базы)
- **В7 — критичный DB-обход + UI**: SPEC-016 (полнота обхода больших схем, приоритет 1), SPEC-011 (раздельный скролл дерева)
- **В8 — версии + интеграция**: SPEC-012 (дельта-версии) + SPEC-013 (удаление версии) — один агент (смежные файлы), SPEC-014 (REA) — параллельно вторым
- **В9 — детерминизм**: SPEC-015 (Output contract + watchdog)
- **В10 — релиз**: SPEC-017 (предрелизный харднинг)
- **В11 — RBAC-фиксы**: SPEC-018 (единые 4 роли в UI для всех провайдеров) — параллельно SPEC-019 (404 /api/admin/grants, порядок маршрутов)
- **В12 — админ-UX**: SPEC-020 (маппинг: combobox с поиском + перенос на «Пользователи») + SPEC-022 (textarea промптов) одним агентом, SPEC-021 (планировщик: шестерёнка + модалка) — параллельно вторым
- **В13 — настройки продукта + stability**: SPEC-023 (права edit/configure на бэке) — параллельно SPEC-024 (модалки продукта) и SPEC-025 (краш чата: content_and_artifact MCP-инструментов)
- **В14 — диагностика LLM + чистка + MCP-scopes**: SPEC-026 (честная проба LLM и причины ошибок генерации) — параллельно SPEC-027 (удаление Confluence) и SPEC-028 (области применения MCP); admin/page.tsx и ru/en.json — не более одного агента (компоновка при запуске)

## Статус

- SPEC-001 — done (В3): двухуровневые права docs/manage (require_product_permission), гранты продукта /api/products/{id}/grants, Keycloak Admin API + /admin/keycloak/groups, админ-экран маппинга + панель пользователей; 3146 тест
- SPEC-002 — done (В1): PATCH /api/products/{id} + inline-UI правки названия/описания/владельца, 12 тестов
- SPEC-003 — done (В1): общий fetch-клиент src/lib/api.ts; дашборд и страница продукта переведены (ретраи GET, abort-молчание)
- SPEC-004 — done (В2): санитайз артефактов (`<tool_call>`/im_start) во всём docgen, валидатор `validate_page` + гейт с авто-ретраем (2 доп.) в codebase/database, degraded-флаг в provenance; полный сюит 3111
- SPEC-005 — done (В1): каталог моделей + fallback 200k + overflow-халвинг (halve_budget_on_overflow) + контекст в админ-Models; 371 тест
- SPEC-006 — done (В1): RLM-упоминания вычищены (код/README/тесты), legacy env только с deprecation-варнингом, README.ru.md — стаб
- SPEC-007 — done (В6): три прохода ревью (43 находки: 7 high / 15 medium / 21 low; отчёты spec/findings-wave6*.md), фиксы 5 группами: RBAC expert/product_mcp/knowledge-нод (SPEC-001 доведён до конца), git-detect таймауты, busy/cancel detect-джоб, write-lock rebuild графа, verified-only MCP-экспорт + rate limit bearer, дедуп фронта (useIncrementalDocs), find_entity/extract_json; 3259 тест
- SPEC-008 — done (В5): граф знаний JSONB per-product (knowledge_graphs; term:/page: ключи, LRU-капы, версии), инкрементальная LLM-экстракция терминов после индексации, graph-boost в pgvector-поиске (byte-identical fallback), inbound MCP tools knowledge_search/knowledge_graph (product_id обязателен), админ-статистика + graph/reindex; 3235 тест
- SPEC-009 — done (В4): APScheduler in-process (60s tick, UTC), docgen_schedules, busy→skipped_busy, инкремент при baseline, admin-REST /schedules + секция «Автогенерация»; 3202 тест
- SPEC-010 — done (В4): source_state на артефактах, джоба detect (git diff / диф фингерпринтов / hash спеки), impact-план provenance+LLM, инкремент через force_pages, page-гранулярная память, UI «Проверить изменения»; 3181 тест
- SPEC-011 — done (В7): aside дерева получил lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto — независимый скролл на lg+, мобильный layout не тронут; lint/build зелёные
- SPEC-012 — done (В8): DocVersion.base_version + override-дельты для edit/force_pages/incremental/rollback, материализация при чтении (set_committed_value), разглаживание при глубине >50, scope в листе/детали; реген 1 страницы не копирует остальные
- SPEC-013 — done (В8): DELETE .../versions/{v} (manage, 204/404/409 current/single/base) удаляет только строку снапшота; корзина артефакта убрана из панели версий, per-version delete с confirm-модалкой; 3284 тест
- SPEC-014 — done (В8): STDIO_TOOL_PRESETS с REA-пресетом, полиси-карваут только node по абсолютному пути (npx/nodejs/bare node остаются заблокированы, env-полиси не тронуты), README §10 — подключение за один шаг; 111 тест
- SPEC-015 — done (В9): единый контракт-блок _unit_output_contract (композиция реестром в 7 юнит-промптов ru+en, trip-wire тест на новые файлы без контракта), превентивная структурная валидация в гейте (обрезанность → retry, не деградация), wall-clock бюджеты docgen_unit/db_unit + watchdog (wait_for) для codebase-юнитов и deep-units; 3294 тест
- SPEC-016 — done (В7): chop-detection маркера «…[truncated» + halving страницы с повтором окна (836 таблиц собираются полностью), COUNT(*)-сверка + per-query walk_stats в coverage, гейт сэмплинга первых строк (DB_DEEP_SAMPLE_ROWS=5), чекпоинты прогресса deep-units; 3264 тест
- SPEC-017 — done (В10): удалено ~145 строк мёртвого кода (9 прод-файлов + embedder.json.bak + dead-тесты, греп-верификация), flaky-тест docgen_async починен по корню (StaticPool→per-test file SQLite), ИБ-ревью новых фич (2 LOW-фикса: блоклист docker/kubectl/ssh-трамплинов, кэп focus_term; summary в docgen rate-бакете), README актуализирован под волны 7-9; S-9/P-5 — осознанный wont-fix; 3292 тест
- SPEC-018 — done (В11): UsersSection — Select всех 4 ролей при создании и per-user роль-Select в таблице для любых провайдеров (замена Promote/Demote, guard no-op, i18n roleManager/roleViewerGlobal, мёртвые promote/demote удалены), docstring PUT /api/admin/users актуализирован
- SPEC-019 — done (В11): блок GET/PUT/DELETE /api/admin/grants перенесён выше динамических /{group}-маршрутов (404 «Unknown admin group: grants» устранён), hermetic-регресс-тест порядка маршрутов; 3296 тест
- SPEC-020 — done (В12): минимальный Combobox в ui.tsx (поиск по вводу, Enter/Esc/blur, пустая опция «все»); фильтры матрицы grants — два Combobox (продукт/пользователь, единый список local+keycloak); секция mapping убрана из навигации, MappingSection (матрица + KC groups→roles) встроена в UsersSection
- SPEC-021 — done (В12): секция автогенерации убрана из потока страницы продукта; для admin — IconButton-шестерёнка в хедере рядом с карандашом + Modal xl (ленивое монтирование панели, проп embedded убирает SectionHeader)
- SPEC-022 — done (В12): редактор промптов — Card flex flex-col, контейнер flex-1 gap-3, Textarea flex-1 (rows=20 как минимальная высота) — заполняет карточку по высоте; 3296 тест, lint/build зелёные
- SPEC-023 — done (В13): ProductAccess.can_edit/can_configure + require_product_permission("edit"/"configure"): PATCH продукта — owner+admin; schedules и grants продукта — owner+manager+admin (менеджер глобально); rw-грант — только CRUD/доки; итог: can_edit (owner+admin) на PATCH, can_configure (owner+manager+admin) на schedules/grants, rw-грант — по-прежнему CRUD/доки; матрица tests/unit/test_product_rbac_edit_configure.py; сьют 3302 passed / 11 failed (падения — зона параллельного агента MCP-scopes)
- SPEC-024 — done (В13): редактирование продукта (название/описание/владелец) — модалка по карандашу (canEditProduct: owner+admin); модалка шестерёнки = расписание + пользователи продукта + MCP-серверы (canConfigureProduct: owner+manager+admin); зеркальные хелперы в lib/types.ts (productAccess живёт там)
- SPEC-025 — done (В13): _wrap_bounded_tool — model_copy(response_format="content") + try/except в _bounded: ValueError «content_and_artifact … list» от MCP-адаптеров больше не роняет граф чата (ошибка → ERROR-строка, tuple-путь проходит content); регресс-тесты в test_mcp_manager.py
- SPEC-026 — done (В14): честная chat-проба в /admin/models/test (4xx/5xx → success=false с sanitized-причиной из JSON error.message, 400 → retry max_completion_tokens, предупреждение о модели вне /v1/models, exception → имя класса); SummaryLLMError → 503 саммари с причиной; первая LLM-причина (log-capture по потоку джобы) дописана в error/indexing_message docgen-джоб; 3250 тест
- SPEC-027 — done (В14): Confluence удалён полностью (коннектор, админ-группа+тест, fallback эксперта, ветки контекста docgen, импорт по URL в БЗ, UI/i18n/тесты); grep -ri confluence api/ tests/ = 0 (README вне acceptance; test_docgen_codebase.py — зона агента SPEC-028); 3250 тест
- SPEC-028 — done (В14): McpServerORM.scopes (MCP_SCOPES, обязательный непустой подсет в CRUD) + идемпотентный бэкфилл (пресеты→database, прочие→expert+database); фильтры: эксперт-чат=expert, интроспекция БД=database с пин-валидацией 400, codebase-агенты получают codebase-scoped инструменты (инъекция в юнит/параллел-агенты); scopes в ответах привязок; мультивыбор-теги в админке, выбор MCP-сервера при добавлении БД; knowledge — резерв; тесты: 483 targeted + новый test_db_mcp_scopes_backfill.py

## Правила для исполнителей (сабагентов)

1. Минимальные диффы, без лишних абстракций (YAGNI/DRY), код самодокументируется.
2. Hermetic-тесты обязательны: `poetry -C api run sh -c 'cd "$(git rev-parse --show-toplevel)" && python -m pytest -q'`; фронт: `bun run lint`, `bun run build`.
3. Секреты: не читать `.env*`; DSN маскируются; файловые чтения через `open_read_nofollow`.
4. При архитектурных изменениях — обновить Knowledge Graph в `AGENTS.md`.
5. Коммиты — Conventional Commits, только по явному запросу владельца; пуш запрещён.
6. По завершении спеки — отметить статус в этом файле (`done` + одна строка итога).

## Журнал решений grilling (2026-10-19)

1. Менеджер видит все продукты: RW на свои/выданные, RO на остальные; админку не видит.
2. Гранты остаются ro/rw, но любой доступ (ro+) даёт docs-права: правка документации, verify, регенерация. RW = управление артефактами/продуктом/пользователями. `viewer_global` — чистый RO.
3. Маппинг продукт↔пользователи — в нашей БД (страница маппинга); группы Keycloak — только глобальные роли, справочник групп через KC Admin API.
4. Редактирование продукта: название + описание + смена владельца.
5. «Failed to fetch» чаще на дашборде и странице продукта.
6. Битые страницы `<tool_call>`: codebase + database RE.
7. Качество: валидационный гейт + авто-ретрай unit-агента (2–3 попытки), санитайз — последний рубеж.
8. Контекст: каталог известных моделей + fallback 200k; overflow → авто-халвинг бюджета + ретрай.
9. База знаний: inbound MCP (`knowledge.graph` / `knowledge.search`) со scope по продукту; граф-дизамбигуация поверх одного векторного namespace на продукт.
10. Крон: in-process шедулер в FastAPI; конфликт → skip с записью в историю; cron per codebase/db; настраивает только админ.
11. Инкремент: git diff / диф каталог-фингерпринтов / hash спеки; impacted-страницы через provenance + LLM-донастройка; скрытые dirty-флаги.
12. Порядок волн: В1 → В2 → В3 → В4 → В5 → В6.
