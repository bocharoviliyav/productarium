# SPEC-018: Единый каркас ролей в админке — 4 роли для всех провайдеров

Статус: planned · Волна 11 · Приоритет: высокий

## Проблема

Бэкенд давно поддерживает все 4 роли (`USER_ROLES` = admin/manager/user/viewer_global): `PUT /api/admin/users` с `{user_id, role}` валидирует роль по `USER_ROLES` (api/routers/admin.py:716-730) и меняет её юзеру любого провайдера. Но UI `UsersSection` (src/app/admin/page.tsx) ограничен парой user/admin:

- форма создания — Select только user/admin (~3298);
- таблица — Tag только admin/user (~3378);
- действия — только Promote/Demote admin↔user (~3400-3420).

Полный список `ROLES` (строка ~3660) используется только в Keycloak groups→roles маппинге — отсюда ощущение «4 роли доступны только для клока». Локальным пользователям назначить manager/viewer_global из UI нельзя.

## Решение

Единый UI в духе Grafana — роль задаётся одинаково для локальных и внешних пользователей:

- Форма создания: Select со всеми 4 ролями (label-ключи i18n для manager/viewer_global добавить).
- Таблица: колонка «Роль» — per-user Select со всеми 4 ролями для ВСЕХ провайдеров; выбор вызывает существующий `setRole` (PUT /api/admin/users); busy-строка блокирует Select. Кнопки Promote/Demote удалить, i18n-ключи `promote`/`demote` (ru/en) удалить как мёртвые.
- Tag-тон роли: admin — blue, остальные нейтральные (без новых сущностей).
- Обновить устаревший docstring `PUT /api/admin/users` в api/routers/admin.py (~613): роль — любая из USER_ROLES, не «(user, admin)».
- Нюанс (не менять): при настроенном `auth.keycloak.role_mapping` роль внешнего юзера пересинхронизируется с групп Keycloak при следующем логине — ожидаемое поведение провайдер-синка, ручная роль остаётся до следующего логина.

## Критерии приёмки

- Локальному и keycloak-пользователю можно выставить любую из 4 ролей из UI (создание + редактирование).
- Promote/Demote и их i18n-ключи удалены; `bun run lint` / `bun run build` зелёные.

## Затрагиваемые файлы

`src/app/admin/page.tsx` (UsersSection), `src/…/ru.json`+`en.json` (admin.users.*), api/routers/admin.py (только docstring).
