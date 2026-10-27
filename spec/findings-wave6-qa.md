# Wave 6 — Adversarial QA (эффективность и корректность) — findings

Скоуп: дифф В1–В5 против HEAD (SPEC-001…010: RBAC/grants, PATCH, fetch-клиент, page-gate, context-window, knowledge graph, scheduler, incremental) + остальной репозиторий. Приоритет — непроревьюянный код. Известный шум (21 exhaustive-deps, ru/en дубликаты messages) пропущен.

Базлайн-прогон: `pytest tests/unit/test_schedules.py tests/unit/test_incremental_jobs.py` → 38 passed.

---

## Находки

**Q-1 [high] api/routers/knowledge.py:156-163, 410-433 — verify_node живёт вне SPEC-001 (известный кандидат с В3); плюс CRUD/summary нод на легаси-rw.**
Проблемы: (a) `_is_owner` даёт право verify автору ноды (`created_by == user.id`) даже после полного лишения доступа к продукту — privilege retention после revoke гранта; (b) на эндпоинте нет `require_product_access`/`require_product_permission` → посторонний получает 403 вместо 404 и тем самым узнаёт о существовании продукта/ноды (SPEC-001: invisible → 404, insufficient → 403); (c) рассинхрон: verify для codebase/spec/database/hld уже переведён на docs-уровень (`routers/products.py:_verify_entity`, `routers/docgen.py`), а knowledge — нет.
Фикс — миграция на SPEC-001 и удаление `_is_owner`:

```diff
 def verify_node(
     product_id: str,
     node_id: str,
     db: Session = Depends(get_db),
-    user: UserORM = Depends(get_current_user),
+    _product: ProductORM = Depends(require_product_permission("docs")),
+    user: UserORM = Depends(get_current_user),
 ) -> KnowledgeNode:
-    product = _load_product(db, product_id)
     node = _load_node(db, product_id, node_id)
-    if not _is_owner(product, node, user):
-        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, ...)
     node.verified = True
     node.verified_by = user.id
```

Удалить `_is_owner`; `_load_product` — если не останется других потребителей.
Также: create_node:194, update_node:289, delete_node:335, upload:357, summary:443 стоят на `require_product_access("rw")` — по SPEC-001 knowledge-ноды это docs-контент (аналог страниц артефактов): ro-грант (can_docs) может редактировать страницы codebase, но не может создать/отредактировать knowledge-ноду; `POST /summary` — генерация, тоже docs. Предлагается перевести контентные операции (create/update/upload/verify/summary) на `require_product_permission("docs")`; delete поддерева — docs достаточно (нода — не артефакт, строгий тайлинг «artifact CRUD = manage» тут не обязателен).

**Q-2 [high] api/docgen/jobs.py:406-427 + api/docgen/scheduler.py:99-123 + api/routers/docgen.py:204-222 — detect-джобы невидимы для busy-семантики шедулера и для cancel.**
Detect-джоба регистрируется под ключом `detect:{entity_type}` и держит реальный `lock_for_entity` минуты (git fetch / полный обход каталога БД). Тик шедулера при baseline зовёт `submit_incremental_job` → `create_or_get_job` по реальному ключу не находит активную джобу (ключи разные) → is_new=True → воркер ждёт lock 5 c (`ENTITY_LOCK_TIMEOUT`, jobs.py:230) → `EntityBusyError` → джоба failed → reconcile пишет `last_result="error"`, хотя контракт SPEC-009 заявляет «busy → skipped_busy, running job untouched». Симметрично: `request_cancel_for_entity` матчится только по реальному ключу → долгую detect-джобу нельзя остановить «Stop»-ом (cancelled: false).
Фикс: в `scheduler._fire` (или в jobs) проверять занятость сущности по ЛЮБОМУ активному job с данным entity_id (`entity_busy(entity_type, entity_id)` — скан registry без учёта `detect:` префикса) → записывать skipped_busy; в `request_cancel_for_entity` добавить матч `detect:{entity_type}`. Добавить тест «busy detect job → skipped_busy» (сейчас `test_busy_entity_records_skipped_busy` покрывает только generation-джобу).

**Q-3 [high] api/clients/git.py:265-281, 284-298 (вызов из api/docgen/incremental.py:94-102) — git-сабпроцессы detect-пути без таймаута, под entity-lock.**
`subprocess.run([… "fetch" …])` и `diff --name-only` без `timeout=`: зависший remote (медленная сеть, зависший git) навсегда блокирует detect-джобу; она держит `lock_for_entity` → все API-записи по сущности 409, incremental/generate не стартуют, cancel недоступен (Q-2). Нарушение AGENTS §6 «каждый внешний вызов подчиняется api/config/timeout.py» (для git-clone это pre-existing, но новый detect-путь делает багу эксплуатируемой по кнопке «Проверить изменения»).
Фикс: `timeout=resolve_git_timeout()` (новая запись в `api/config/timeout.py`, например 120 s) в `fetch_remote_tip`/`diff_commit_names`; `subprocess.TimeoutExpired` → `ValueError` → detect корректно вернёт `changed=None`.

**Q-4 [high] api/memory/knowledge_graph.py:751-753 — rebuild_product_graph пишет граф без write-lock → потеря апдейтов.**
Финал rebuild'а — прямые `_load_graph_sync`/`_save_graph_sync` мимо `_graph_write_lock`/`mutate_product_graph`. Rebuild идёт минуты (LLM-извлечение по всем страницам); за это время пост-индексация генерации (например, по расписанию SPEC-009) закоммитит `update_graph_from_pages` → финальный save rebuild'а перезапишет его своим снимком (stale `prev_version`, lost update; `row.version = prev_version + 1` без проверки версии не детектит конфликт).
Фикс: сохранять результат rebuild через `mutate_product_graph` с мутатором, заменяющим содержимое, либо обернуть финальные load+save в `async with _graph_write_lock()`.

**Q-5 [medium] api/docgen/incremental.py:211-226 + api/utils/llm_helpers.py:39-44 — cap обрезает инструкцию LLM в конце промпта экспансии.**
Промпт `_expand_impact` заканчивается строкой «Ответь ТОЛЬКО JSON-массивом…», а `cap()` режет хвост. При >12 000 символов (200 файлов + 300 страниц со списками источников — легко превышается) инструкция исчезает, модель отвечает прозой, `_parse_id_array` возвращает [] → LLM-расширение impact-плана тихо деградирует ровно на больших документациях, где оно нужнее всего.
Фикс: собирать промпт из покомпонентно капнутых секций так, чтобы финальная инструкция гарантированно попадала в лимит (перенести инструкцию перед списками и/или капнуть каждую секцию отдельно).

**Q-6 [medium] api/routers/admin.py:320-348 — POST /api/admin/memory/graph/reindex: unbounded gather + синхронный долгий запрос.**
(a) `asyncio.gather(*[rebuild_product_graph(p) for p in pids])` — все продукты параллельно, каждый со своим семафором `docgen_llm_concurrency` → процесс-широкий лимит не соблюдается: N×concurrency параллельных LLM-вызовов → риск 429 локального сервера (AGENTS §6). (b) Ручка выполняется синхронно: на всех продуктах это минуты-часы LLM-работы в одном HTTP-запросе — прокси/клиент отвалится по таймауту (контраст: docgen — 202+poll).
Фикс: последовательный `for` (или общий `Semaphore(1)`); для «всех продуктов» — 202 + фоновая задача (готового job-реестра хватает).

**Q-7 [medium] api/mcp/inbound.py:256-272 — knowledge_graph tool: fresh=True на каждый вызов + строковая обрезка ломает JSON.**
`load_product_graph(product_id, fresh=True)` обходит TTL-кэш, введённый именно потому, что «recall path reads the graph per query»: каждый вызов инструмента тянет из БД и парсит потенциально мегабайтный JSON (граф до 5000 нод). При превышении лимита ответ усекается как строка `out[:_MAX] + "…(truncated)"` → невалидный JSON для MCP-клиента.
Фикс: убрать `fresh=True` (30-сек TTL достаточен для справочного тула); при переполнении уменьшать `max_nodes` у `subgraph` и сериализовать заново, а не резать строку.

**Q-8 [medium] src/app/products/[productId]/artifacts/[artifactId]/page.tsx (handleDetectChanges / handleIncrementalUpdate) — поллинг мимо src/lib/api.ts: один сетевой сбой убивает отслеживание джобы.**
Оба новых цикла поллинга используют сырой `fetch`: у статусных GET нет ретраев SPEC-003 (api.ts ретраит только сам себя) и нет пер-запросного таймаута. Один transient 5xx/обрыв за 30-минутное окно incremental → throw → юзер видит «Update failed», хотя джоба на сервере продолжает работать (и версию создаст); повесившийся status-запрос блокирует цикл до системного таймаута.
Фикс: поллинг через `getJson` (GET-ретраи уже встроены), старт — `postJson`; этого достаточно, внутренний цикл оставить.

**Q-9 [medium] api/routers/products.py:307 (+ api/routers/databases.py:457) — очистка страницы до пустого текста не чистит её чанки в памяти.**
`merge_stored_page_bodies` поддерживает явную очистку (`content: ""`), но `_reindex` при `page_id` и пустом `indexed_text` не делает ничего → чанки `{entity_id}::page::{pid}` остаются в recall навсегда (wholesale-ветка чистит через `prune_stale_sources`, одностраничная — нет). Эксперт/Ask продолжает цитировать удалённый текст.
Фикс: в ветке `page_id` при пустом тексте вызвать `prune_stale_sources(product_id, source_type, entity_id, keep=[])` (или прямой delete source) перед выходом.

**Q-10 [medium] api/routers/knowledge.py:101-120, 169-181 — GET knowledge/tree отдаёт content_md всех нод сразу.**
`_node_dict` кладёт полное тело каждой ноды; ноды наполняются в т.ч. 50-МБ markitdown-конверсиями. Отдельный ленивый `GET /knowledge/nodes/{id}` для полного тела уже существует (страница ноды фетчит его отдельно). Дерево на продукте с сотнями документов = мегабайты на каждый заход.
Фикс: в tree-ответе отдавать метаданные без `content_md` (или поддержать `?light=1` по аналогии с продуктами); тело — через существующий get_node.

**Q-11 [low] src/lib/types.ts:337-348 vs api/auth/deps.py:111-125 — viewer_global + rw-грант: UI-зеркало расходится с бэкендом.**
Бэкенд: viewer_global → всегда `ro`, грант не действует, can_docs=False. Фронт: `grant?.level === "rw"` проверяется раньше роли → `"manage"` → UI показывает delete/rollback, которые вернут 403.
Фикс: во фронтовом `productAccess` проверять `viewer_global` до гранта.

**Q-12 [low] api/docgen/scheduler.py:150-179 — синхронная БД в async tick.**
`tick()` крутится в главном event loop, но `db.query(...)/commit()` синхронные — на Postgres каждый тик блокирует loop на время запросов (интервал 60 c, объём мал, но под нагрузкой/медленным диском заметно). Паттерн `asyncio.to_thread` уже есть в knowledge_graph.py.
Фикс: обернуть тело тика в `await asyncio.to_thread(_tick_sync)`.

**Q-13 [low] api/routers/products.py:236-271 — patch_product: устаревший докстринг про «SPEC-001 ещё не влит».**
Комментарий «the manage/rw permission split lands with SPEC-001 — until then reuses plain 'rw'» неверен: SPEC-001 влит в этом же диффе. Поведение корректно (`require_product_access("rw")` ≡ can_manage ≡ manage), но текст вводит в заблуждение.
Фикс: заменить на `require_product_permission("manage")` и убрать абзац.

**Q-14 [low] api/routers/products.py:282-331 ↔ api/routers/databases.py:436-470 — дублированный `_reindex` с новой page-логикой.**
Две копии идентичных ~30 строк, включая новую page_id/pages ветвь (и баг Q-9 задвоен). Обоснование «independently testable» слабое: helper в `_common.py` тестируется напрямую, роутеры мокаются уровнем выше.
Фикс: один общий `reindex_entity(...)` в `api/docgen/_common.py`, вызов из обоих роутеров.

**Q-15 [low] src/components/ProductUsersPanel.tsx:113-119 — поле user id очищается даже при неудачном grant.**
`grant()` глотает ошибки внутри (`fail(e)`), поэтому `setUserId("")` в `add` выполняется всегда: при 400/404 пользователь теряет введённый id и должен вспоминать его заново.
Фикс: вернуть из `grant` успех и чистить поле только при нём.

**Q-16 [low] api/routers/schedules.py:123-146 — гонка двух одновременных PUT на новый schedule → 500.**
Query-then-insert без `IntegrityError`-обработки: два параллельных PUT на один новый (product, entity) → один получает IntegrityError → 500 вместо 200-upsert. Паттерн обработки уже есть в `_start_generate` (docgen.py:113-122).
Фикс: `except IntegrityError: db.rollback(); перечитать строку и обновить`.

---

## Герметичность тестов (сводно, без отдельных ID)

- Новые сьюиты герметичны: `test_schedules`/`test_incremental_jobs` чистят `_docgen_jobs`/`_pending` autouse-фикстурой; `test_knowledge_graph` сбрасывает `_graph_cache`/`_graph_lock`; моки — через monkeypatch (авто-undo), executor не диспатчится (`_patch_submits`). Существенных mock-ликов не найдено.
- Пробелы покрытия ошибочных веток: (1) busy-detect vs scheduler — Q-2, теста нет; (2) таймаут-ветка `fetch_remote_tip`/`diff_commit_names` — сейчас не тестируема, таймаута нет (Q-3); (3) конкурентный rebuild vs update графа — Q-4; (4) очистка страницы до пуста + recall — Q-9.

## Сводка по severity

- critical: 0
- high: 4 (Q-1, Q-2, Q-3, Q-4)
- medium: 6 (Q-5 … Q-10)
- low: 6 (Q-11 … Q-16)
- Итого: 16

## Топ-5 must-fix

1. Q-1 — миграция knowledge.py на `require_product_permission("docs")`: retention прав у автора ноды после revoke + 403/404 leak + рассинхрон с SPEC-001.
2. Q-3 — таймауты на git fetch/diff в detect-пути: сейчас зависший remote блокирует сущность (409 на все записи) без возможности cancel.
3. Q-2 — busy-семантика шедулера и cancel не видят detect-джобы: `error` вместо `skipped_busy`, «Stop» не работает.
4. Q-4 — rebuild графа без write-lock теряет конкурентные апдейты (lost update).
5. Q-5 — cap режет инструкцию `_expand_impact`: LLM-расширение impact тихо не работает на больших документациях.
