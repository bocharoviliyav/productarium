# SPEC-023: Права на настройки продукта — edit (owner+admin) и configure (owner+manager+admin)

Статус: planned · Волна 13 · Приоритет: высокий

## Проблема

Права на «настройки продукта» сейчас размазаны и не совпадают с целевой моделью (ответы grilling 2026-10-27):

- `PATCH /api/products/{id}` (название/описание/владелец) — `require_product_permission("manage")`: rw-грант может переименовать продукт и сменить владельца. Цель: только владелец и админ.
- `GET/PUT/DELETE /api/products/{id}/schedules` — `require_admin` (api/routers/schedules.py:87,104,179): владелец и менеджер не могут настроить расписание своих продуктов. Цель: owner+manager+admin.
- `GET/PUT/DELETE /api/products/{id}/grants` (пользователи продукта) — `manage`: rw-грант управляет доступом. Цель: owner+manager+admin.
- Менеджер — глобальная роль (как в Grafana): настройки видны на ВСЕХ продуктах, а не только на своих.

Итоговая матрица: **edit** (название/описание/владелец) = owner || admin; **configure** (расписание, пользователи продукта) = owner || manager || admin; rw-грант — по-прежнему CRUD артефактов + правки дока (can_manage/can_docs не меняются); user без грантов и viewer_global — ничего из перечисленного.

## Решение

- `api/auth/deps.py`: в `ProductAccess` добавить `can_edit` (product.owner_id == user.id или role == "admin") и `can_configure` (owner, manager или admin — НЕ через гранты); расширить `require_product_permission` значениями `"edit"` / `"configure"` (+ `_PERMISSION_DETAILS`), 404-невидимость сохранить как есть.
- `api/routers/products.py`: PATCH продукта (и transfer owner_id) → `require_product_permission("edit")`; grants REST → `"configure"`.
- `api/routers/schedules.py`: три ручки с `require_admin` → `require_product_permission("configure")` (product_id уже в пути).
- Обновить затронутые тесты (RBAC/schedules/grants) + добавить новые кейсы: rw-грант-юзер — 403 на PATCH/grants/schedules; manager-не-владелец — 403 на PATCH, 200 на schedules/grants; owner — 200 везде; viewer_global — 404/403 по невидимости.

## Критерии приёмки

- Матрица edit/configure соблюдается на бэке; полный hermetic-сьют зелёный.

## Затрагиваемые файлы

`api/auth/deps.py`, `api/routers/products.py`, `api/routers/schedules.py`, `tests/` (RBAC-тесты).
