# Wave 6 (SPEC-007) — единый триаж-отчёт ревью

Три независимых read-only прохода (2026-10-26): Simple (`findings-wave6-simple.md`, 19), QA (`findings-wave6-qa.md`, 16), Pentest (`findings-wave6-pentest.md`, 8). Детали — в линз-файлах; здесь сводка и решение владельца.

Итог: **43 находки — 0 critical / 7 high / 15 medium / 21 low**. Пересечения: P-3 ⊂ Q-1; S-14 = Q-13; S-12 = Q-14; S-1 + S-2 + Q-8 — один фронтовый фронт работ.

Рекомендация оркестратора: **fix** — все high и medium, большая часть low; **wont-fix** — S-9, S-19, P-8 (обоснования в таблице), из P-5 — только минимальная часть (verified-only фоллбек), колонка scope на токены отложена.

**Фаза 3 завершена (2026-10-26): все находки с решением fix закрыты (включая Q-6, выпавший из промптов групп и закрытый группой E); wont-fix — S-9, S-19, P-8, scope-колонка P-5. Полный сьют 3259 passed / 0 failed; lint/build зелёные.**

## Триаж

| ID | Sev | Файл | Суть | Решение |
| :--- | :--- | :--- | :--- | :--- |
| P-1 | high | api/routers/expert.py:341 | `/ask`+stream/cancel/attachments без product-access: извлечение знаний любого продукта | **fix** |
| P-2 | high | api/routers/product_mcp.py:114 | CRUD MCP-биндингов без product-access: тамперинг чужого продукта | **fix** |
| Q-1 | high | api/routers/knowledge.py:156,410 | `_is_owner` вместо SPEC-001: retention прав после revoke, 403/404 leak, rw не может verify; миграция нод на docs-уровень | **fix** |
| Q-2 | high | api/docgen/jobs.py:406 + scheduler.py:99 | detect-джобы невидимы busy-семантике (error вместо skipped_busy) и cancel | **fix** |
| Q-3 | high | api/clients/git.py:265-298 | git fetch/diff без timeout под entity-lock: зависший remote = вечный 409 без cancel | **fix** |
| Q-4 | high | api/memory/knowledge_graph.py:751 | rebuild графа мимо write-lock: lost update против конкурентной индексации | **fix** |
| S-1 | high | SpecEditor.tsx:381-516 | ~135 строк detect/incremental-хендлеров ×2 (вьюер + редактор): общий хук useIncrementalDocs | **fix** |
| S-2 | med | фронт | три HTTP-паттерна; новые хендлеры мимо src/lib/api.ts; 401-редирект ×3; useAdminApi мигрировать | **fix** (useAdminApi — опц.) |
| S-3 | med | pgvector_backend.py:828 | `_cosine_rows_with_source` дублирует `_cosine_rows`: один метод с source_id | **fix** |
| S-4 | med | jobs.py:703 | cancel-ветка инлайнит `_load_docgen_entity` | **fix** |
| S-5 | med | системно | entity_type→коллекция ×8+ (волна +3): единый хелпер в product_repo | **fix** |
| S-6 | med | mcp/inbound.py:238,246 | два байт-идентичных MCP-тула: оставить `knowledge_search`, старый удалить (внешних клиентов нет) | **fix** |
| Q-5 | med | incremental.py:211 | cap режет финальную LLM-инструкцию: expansion тихо пустой на больших доках | **fix** |
| Q-6 | med | routers/admin.py:320 | reindex: unbounded gather (N×concurrency LLM) + синхронный долгий HTTP | **fix** (мин.: последовательно; all-products → 202+фон) |
| Q-7 | med | mcp/inbound.py:256 | knowledge_graph tool: fresh=True каждый вызов + строковая обрезка JSON | **fix** |
| Q-8 | med | артефакт-вьюер | поллинг raw-fetch: один сбой = «Update failed» при живой джобе; → getJson/postJson (в S-1) | **fix** |
| Q-9 | med | products.py:307, databases.py:457 | очистка страницы до пуста не чистит её чанки: recall цитирует удалённый текст | **fix** |
| Q-10 | med | knowledge.py:101 | GET tree тащит content_md всех нод: метаданные в дереве, тело — существующим lazy-эндпоинтом (+фронт) | **fix** |
| P-3 | med | knowledge.py:414 | (= Q-1) verify_node retention/leak | **fix** (в Q-1) |
| P-4 | med | mcp/inbound.py:274 | ask_expert без rate limit (cost/DoS): bucket на bearer-токен | **fix** |
| P-5 | med | mcp/inbound.py:157 | токены без product-scope/TTL; get_product_knowledge экспортирует не-verified черновики | **fix-min** (фоллбек убрать); scope-колонка — отложить |
| P-6 | med | products.py:596 | GET grants грузит ВСЕХ юзеров платформы: перечислять только грантованных | **fix** |
| S-7 | low | schedules.py:17 | неиспользуемый импорт List | **fix** |
| S-8 | low | schedules.py:155 | DELETE игнорирует query entity_type | **fix** |
| S-9 | low | новые файлы | Optional[...]-стиль вместо `A \| B` (репо-широкий стиль) | **wont-fix** (точечно при правках) |
| S-10 | low | knowledge_graph.py:88 | `_now_iso` дублирует `utc_now_iso` | **fix** |
| S-11 | low | incremental/database/knowledge_graph | 4-й вариант tolerant-JSON: один extract_json | **fix** |
| S-12 | low | products.py:282, databases.py:439 | `_reindex` ×2 (~30 строк) | **fix** (= Q-14) |
| S-13 | low | docgen.py:345,390 | 202-блок ×2: хелпер _submit_202 | **fix** |
| S-14 | low | products.py:243 | устаревший докстринг «until SPEC-001» + guard на require_product_permission("manage") | **fix** (= Q-13) |
| S-15 | low | app/page.tsx | DEGRADED_LABEL: устаревшая отсрочка i18n | **fix** |
| S-16 | low | панели фронтa | мелкие дубли fail/busy-guard, users-нормализация, непоследовательный eslint-disable | **fix** |
| S-17 | low | knowledge_graph.py:351 | subgraph: alias-скан при прямом попадании | **fix** |
| S-18 | low | documents.py:346 | DatabaseManager() ради приватного метода | **fix** |
| S-19 | low | jobs.py | boilerplate финализации статуса | **wont-fix** (объём сопоставим, сам ревьюер признал пограничным) |
| Q-11 | low | types.ts:337 | viewer_global+rw: UI-зеркало расходится с бэкендом (порядок проверок) | **fix** |
| Q-12 | low | scheduler.py:150 | синхронная БД в async tick: asyncio.to_thread | **fix** |
| Q-13 | low | products.py:236 | (= S-14) | **fix** (в S-14) |
| Q-14 | low | products/databases | (= S-12) | **fix** (в S-12) |
| Q-15 | low | ProductUsersPanel.tsx:113 | user id очищается при неудачном grant | **fix** |
| Q-16 | low | schedules.py:123 | гонка PUT→IntegrityError→500: upsert-обработка | **fix** |
| P-7 | low | knowledge_graph.py:709, admin.py | сырой текст исключений клиенту: generic + лог | **fix** |
| P-8 | low | repo_url.py:59 | SSRF deny-list приватных диапазонов для git | **wont-fix** (local-first; env-флаг при необходимости) |

## Предлагаемые группы фиксов (фаза 3)

- **A. RBAC-роутеры**: P-1 (expert.py), P-2 (product_mcp.py), Q-1+P-3 (knowledge.py, вкл. Q-10 light-tree), P-6 (products.py grants), S-14/Q-13 (guard+докстринг)
- **B. Джобы/шедулер/git**: Q-2 (busy/cancel detect), Q-3 (git timeouts), Q-16+S-8+S-7 (schedules), S-4, S-13, Q-12
- **C. Memory/knowledge graph**: Q-4 (write-lock), Q-7+S-6+S-17+S-10 (inbound/graph), S-3 (cosine dedup), Q-6 (reindex), Q-9+S-12/Q-14 (_reindex+prune), S-18, P-7, S-11
- **D. Фронтенд**: S-1+S-2+Q-8 (useIncrementalDocs + api.ts), Q-11, Q-15, S-15, S-16
- **E. Системные хелперы**: S-5 (entity_type helper)

Границы по файлам не пересекаются; A/B/C/D/E можно параллелить воркдридами или делать последовательно в общем чекауте.
