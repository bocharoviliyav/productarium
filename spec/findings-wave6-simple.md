# Wave 6 — Simple Review (минимализм / DRY / читабельность)

Скоуп: дифф В1–В5 против HEAD (~70 файлов, +5170/−1568) + репо целиком. Линза: лишние абстракции, дублирование, мёртвый код, неиспользуемые импорты, Google Python Style / PEP 8, типизация 3.11+ (`A | B`), Next.js 15 конвенции, strict TS. Валидация: `bunx tsc --noEmit` — чисто; ruff в poetry-окружении отсутствует.

## Находки

### S-1 [high] src/components/SpecEditor.tsx:381-516 — ~135 строк detect/incremental-хендлеров скопированы из артефакт-вьюера
`handleDetectChanges` + `handleIncrementalUpdate` — почти дословная копия тех же хендлеров из `src/app/products/[productId]/artifacts/[artifactId]/page.tsx:725-865`: одинаковые raw-fetch + poll-циклы (setTimeout 2000, границы 10/30 мин), одинаковые notify-блоки и проводка `DetectChangesModal`; отличаются только базовый URL, секция i18n и побочные эффекты успеха. Две копии polling-логики = двойная цена фиксов (тайминги, обработка 401, интервалы уже могли разъехаться).
Фикс: вынести общий хук `useIncrementalDocs({ detectUrl, incrementalUrl, onApplied })` в `src/lib/` или hooks-модуль; обе точки вызывают его.

### S-2 [medium] фронтенд: три параллельных HTTP-паттерна; новые хендлеры — raw fetch при живом src/lib/api.ts
Новые detect/incremental-хендлеры (S-1) используют сырой `fetch` (credentials/no-store/ручной разбор JSON и ошибок), тогда как эта же волна мигрирует страницы на `src/lib/api.ts` (`postJson`/`getJson`) с заявленным в докстринге контрактом «Every non-SSE API call goes through here» (SPEC-003). Параллельно: паттерн `e instanceof ApiError && e.status === 401 → router.replace(login)` продублирован 3+ раза по страницам; админка держит собственную обёртку `useAdminApi` на raw fetch (`src/app/admin/page.tsx:101`). Итого три клиента с разной обработкой ошибок/401/таймаутов.
Фикс: detect/incremental-хендлеры перевести на `postJson`/`getJson`; 401-редирект centralized (перехват в api.ts или хелпер `onApiError(e, router)`); `useAdminApi` мигрировать на api.ts (можно отдельным PR).

### S-3 [medium] api/memory/pgvector_backend.py:828 — `_cosine_rows_with_source` дублирует `_cosine_rows` (:874)
~35 строк одинаковой формы: тот же SQL-скелет, `vec_literal`, timeout, `wait_for`; отличие только в выборе `source_id`+distance. Любой фикс SQL/таймаутов придётся делать дважды в бэкенде памяти.
Фикс: один приватный метод, возвращающий `(content, source_id, distance)`; публичный `_cosine_rows` — тонкая обёртка, собирающая текст.

### S-4 [medium] api/docgen/jobs.py:703-720 — обработчик JobCancelledError инлайнит reload+entity-find вместо `_load_docgen_entity` (:503)
Хендлер заново строит то, что волна только что вынесла в хелпер (reload продукта, словарь коллекций, поиск сущности). Незавершённый рефакторинг: два пути загрузки сущности разъедутся.
Фикс: в cancel-ветке вызвать `_load_docgen_entity(...)` в try/except ValueError (сущность удалена → тихий rollback-путь как сейчас).

### S-5 [medium] системно: маппинг entity_type→коллекция продублирован 8+ раз, волна добавила ещё 3 копии
Новые копии: `api/routers/docgen.py:337` (`_require_detectable_entity`), `api/routers/schedules.py:76`, `api/docgen/scheduler.py:82` (`_load_entity`) — рядом с существующими docgen.py:125/252/295, doc_versions.py:80, jobs.py:521/715. Один и тот же if/elif-словарь `codebases|specs|databases|hlds`; добавление типа сущности (как когда-то hld) требует правок во всех точках — уже источник пропусков.
Фикс: один хелпер в `api/repositories/product_repo.py` (рядом с `_find_child`, product_repo.py:1078), возвращающий ORM-класс/коллекцию; все точки вызывают его.

### S-6 [medium] api/mcp/inbound.py:238,246 — два MCP-инструмента с байт-идентичными телами
`search_knowledge` (:238) и `knowledge_search` (:246) — оба `return await _search_cited(...)`. Графовое осознание теперь живёт в бэкенд-запросе (`PgVectorMemoryBackend.query`), так что различие контрактов исчезло; один инструмент лишний (YAGNI) и размывает MCP-поверхность.
Фикс: оставить один инструмент (каноничное имя `knowledge_search`), второй удалить или задепрекейтить с «alias to …» в описании на переходный период.

### S-7 [low] api/routers/schedules.py:17 — неиспользуемый импорт `List`
Единственное вхождение в файле — строка импорта. Мёртвый код.
Фикс: удалить из строки импорта typing.

### S-8 [low] api/routers/schedules.py:155-176 — DELETE-endpoint принимает `entity_type`, но фильтр его игнорирует
Query-параметр объявлен, в фильтрации участвуют только product_id+entity_id: API обещает семантику, которую не реализует (пользователь просит удалить «расписание для codebase», удаляется любое).
Фикс: либо добавить `entity_type` в WHERE-условие, либо убрать параметр из сигнатуры.

### S-9 [low] типизация на новых строках нарушает правило проекта «Python 3.11+, union `A | B`»
Конкретные новые точки: `api/clients/git.py:249` (`repo_type: str = None`, `access_token: str = None` — implicit Optional), `:284` `diff_commit_names(...) -> list` (голый list); `api/repositories/documents.py:346` `managed_repo_dir(repo_url, repo_type: str = None)`; `api/docgen/jobs.py:476` `active_jobs_for_product(...) -> list`. Консолидированно: новые файлы incremental.py / scheduler.py / schedules.py / knowledge_graph.py написаны в `Optional[...]`/`Dict[...]` — стиле; сам стиль репо-широкий, но новые файлы — точка, где стоило стартовать в `A | B` (AGENTS.md п.1).
Фикс: в перечисленных сигнатурах — `str | None`, `list[...]`; новые файлы перевести на `X | None` точечно при следующих правках (не блокер).

### S-10 [low] api/memory/knowledge_graph.py:88-91 — `_now_iso()` байт-дублирует `utc_now_iso()` из api/docgen/_common.py:275
Модуль уже импортирует из `_common` другое — противопоказаний к импорту нет. Второй источник правды для формата времени.
Фикс: `from api.docgen._common import utc_now_iso`, удалить локальный хелпер.

### S-11 [low] четвёртый вариант tolerant-JSON-экстрактора
`api/docgen/incremental.py:183-194` `_parse_id_array` повторяет паттерн среза JSON-массива из `api/docgen/database.py:~1685` (версия в database.py к тому же срезает code-fences — тихое расхождение поведений); `api/memory/knowledge_graph.py:521` `_parse_extraction` — четвёртый вариант. Один и тот же риск «LLM-ответ с мусором вокруг JSON» решается по-разному.
Фикс: один tolerant `extract_json(text)` в `api/docgen/_common.py` (или utils), специализированные парсеры — тонкие обёртки.

### S-12 [low] `_reindex` трижды: api/routers/products.py:282-323 и api/routers/databases.py:439-480 — почти идентичные ~30 строк
Одинаковое 3-ветвление page/pages/blob; doc_versions.py:180 импортирует версию из products. Комментарий «kept local for independent testability» слаб, когда логика утроена.
Фикс: вынести в один модуль (например `api/routers/_indexing.py` или в docgen/_common), три точки вызывают.

### S-13 [low] api/routers/docgen.py:345-375, 390-425 — два новых эндпоинта делят байт-идентичный 10-строчный блок JSONResponse-202 + одинаковый преамбул
`detect_docgen_changes` и `incremental_docgen_update` повторяют rate-limit + `_require_detectable_entity` + возврат 202 с полем job_id.
Фикс: маленький хелпер `_submit_202(job_id)` (или общий преамбул-декоратор), −~15 строк.

### S-14 [low] устаревшие комментарии «until SPEC-001»
`api/routers/products.py:243-248` (docstring patch_product: «The manage/rw permission split lands with SPEC-001 — until then this reuses the plain 'rw' guard») — SPEC-001 приземлён в этой же волне; аналогичная формулировка в `src/app/products/[productId]/page.tsx` (~1314 в диффе, комментарий edit-состояния). Комментарий обещает временное поведение, которое уже стало постоянным.
Фикс: либо переключить guard на `require_product_permission("manage")` (если это и был план), либо переписать комментарий в безусловную форму.

### S-15 [low] src/app/page.tsx — `DEGRADED_LABEL`: заметка об отсрочке («src/messages/** is owned by a parallel change; move to catalogs later») устарела
Каталоги messages в этой же волне менялись — причина отсрочки исчезла, инлайн-двуязычная константа осталась.
Фикс: перенести строки в ru/en каталоги, удалить константу и комментарий.

### S-16 [low] мелкие дубли фронтенда
- Хелпер `fail` (notify) + busy-guard try/catch продублированы между `ProductSchedulesPanel.tsx:77` и `ProductUsersPanel.tsx:46` — кандидаты на общий мелкий хук `usePanelActions` или хотя бы одинаковый стиль.
- Fetch+нормализация `/api/admin/users` (`Array.isArray(data) ? data : data?.users ?? []`) в двух местах: `ProductUsersPanel.tsx:70-80` и products-страница `openProductEdit`.
- `ProductUsersPanel.tsx:61` добавляет `// eslint-disable-next-line react-hooks/exhaustive-deps`, тогда как соседний `ProductSchedulesPanel` обходит проблему инлайном значений — непоследовательно в рамках одной волны (известные 21 варнинг — вне скоупа, речь о новой строке).

### S-17 [low] api/memory/knowledge_graph.py:351-362 — `subgraph()` считает `alias_hit` (полный скан нод) даже при прямом попадании
Прямое попадание уже даёт ответ; скан алиасов выполняется всегда. Мелкая трата + лишняя ветка чтения.
Фикс: ранний return по прямому попаданию, alias-скан только при промахе.

### S-18 [low] api/repositories/documents.py:346 — `managed_repo_dir` создаёт `DatabaseManager()` ради вызова приватного `_extract_repo_name_from_url`
Доступ к приватному методу чужого класса + лишняя инстанциация ради одной строки.
Фикс: вынести извлечение имени репо в модульную функцию (или сделать метод публичным статическим), вызывать её.

### S-19 [low] api/docgen/jobs.py — boilerplate финализации статуса повторён 6+ раз; новый `_run_detect_job_async` копирует error-пути `_run_docgen_job_async`
Присвоения `job["status"]/["indexing_status"]/["indexing_message"]/["error"]/["finished_at"]` размножены по функциям; detect-воркер (~30 строк) дублирует обработку исключений docgen-воркера.
Фикс: маленький `_finish(job, status, message)` хелпер; в перспективе — общий раннер-каркас для асинхронных воркеров jobs.py. Пограничный (код останется сопоставим по объёму), потому low.

## Сводка

- critical: 0
- high: 1 (S-1)
- medium: 5 (S-2 … S-6)
- low: 13 (S-7 … S-19)
- Всего: 19

Не находки (проверено): `typing.Dict/List`-стиль как таковой (репо-широкий), router-level auth-dependency и `__all__` (конвенция), `setup_logging()` на уровне модуля, перенос `_is_context_overflow`/`halve_budget_on_overflow` в `_common.py` (хороший рефакторинг), `Messages = Record<string, any>` (pre-existing, объясняет слабую i18n-типизацию), рефакторинг llm_tokens.py и экстракции в database.py — чистые.

## Топ-5 must-fix

1. **S-1** — дедуплицировать detect/incremental-хендлеры SpecEditor ↔ артефакт-вьюер (~135 строк × 2).
2. **S-5** — единый хелпер entity_type→коллекция в product_repo (сейчас 8+ копий, волна добавила 3).
3. **S-2** — перевести новые хендлеры на `src/lib/api.ts`, centralize 401-обработку (три HTTP-паттерна → один).
4. **S-3** — слить `_cosine_rows_with_source` с `_cosine_rows` в один метод с source_id.
5. **S-6** — удалить дублирующий MCP-инструмент (`search_knowledge` ↔ `knowledge_search`).

Быстрые победы на 5 минут: S-7 (unused import), S-8 (лишний query-параметр), S-10 (`_now_iso`), S-14 (устаревшие комментарии).
