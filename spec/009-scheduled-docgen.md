# SPEC-009: Крон-генерация документации по расписанию

Статус: planned · Волна 4 (выполнять после SPEC-010) · Приоритет: средний

Решения grilling: in-process шедулер в FastAPI; при уже идущей генерации — skip с записью в историю; cron-выражение per codebase/database; настраивает только админ.

## Решение

1. **Модель данных**: таблица `docgen_schedules` (`id`, `product_id` FK, `entity_type` (codebase|database), `entity_id`, `cron_expr`, `enabled`, `last_run_at`, `last_result` (ok|skipped_busy|error|…), `created_by`, `updated_at`; unique `(product_id, entity_id)`).
2. **Движок**: APScheduler `AsyncIOScheduler` в lifespan (`api/api.py`), проверка каждую минуту:
   - триггер → если по сущности нет активной джобы (`api/docgen/jobs.py`) → `submit_job` с `source="schedule"`; режим инкрементальный, если доступен baseline SPEC-010, иначе полная генерация;
   - занято → запись в историю «skipped_busy» (не трогаем текущий прогон);
   - лок-семантика и бюджеты существующие (`lock_for_entity`, concurrency, timeouts) — без изменений.
3. **REST (админ)**: `GET/PUT/DELETE /api/products/{product_id}/schedules` (+ при необходимости сквозной `/admin/schedules`); валидация cron-выражения и существования сущности; ответ содержит `next_run_at` (расчёт шедулером).
4. **UI**: на странице продукта — админ-секция «Автогенерация»: по каждому codebase/database — вкл/выкл, пресеты (ежедневно 02:00, еженедельно вс 03:00) + поле cron, следующий запуск, итог последнего прогона. Часовой пояс — серверный (UTC), показывать в UI явно.
5. **История**: джобы уже пишутся — добавить метку source=schedule и skipped-записи; на странице продукта видна история автогенераций.

## Не-цели

- Очередь отложенных прогонов (пропускаем, не копим), календарь праздников, расписания для spec/hld (можно включить позже тем же механизмом), email-уведомления.

## Критерии приёмки

- Юнит: парсинг cron, расчёт next_run, уникальность расписания на сущность.
- Интеграционный (герметичный): событие расписания при активной джобе → skipped_busy в истории; при свободной — джоба создана с source=schedule.
- Ручной прогон с коротким cron (например `*/1 * * * *`) на тестовом продукте.

## Затрагиваемые файлы

`api/models.py`, `api/api.py` (lifespan), новый `api/docgen/scheduler.py`, `api/docgen/jobs.py` (source-метка), `api/routers/products.py` (или новый подроутер schedules), `api/pyproject.toml` (apscheduler), `src/app/products/[productId]/page.tsx`, `src/lib/types.ts`, `tests/unit/test_schedules*.py`, `AGENTS.md` (Knowledge Graph).
