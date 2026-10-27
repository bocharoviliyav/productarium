# SPEC-001: Ролевая модель и маппинг доступа

Статус: planned · Волна 3 · Приоритет: высокий

## Проблема

Роли `admin/manager/user/viewer_global` и гранты `ro/rw` уже есть (`api/models.py:USER_ROLES`, `ProductGrantORM`, `api/auth/deps.py:resolve_product_access`), Keycloak мапит только глобальные роли (`api/auth/router.py:_keycloak_role_mapping`, setting `auth.keycloak.role_mapping`). Не хватает:

- прав «читает продукт, но правит документацию» без права управления продуктом;
- страницы маппинга продукт↔пользователи/менеджеры (гранты сейчас только через `/api/admin/grants`, без UI-матрицы);
- менеджмента грантов владельцем-менеджером продукта;
- выделенной админ-группы и справочника групп Keycloak в UI.

## Решения

1. **Менеджер**: видит ВСЕ продукты; `rw` на свои и выданные, `ro` на остальные; создаёт продукты; админку (`/admin`) не видит.
2. **Двухуровневые гранты + docs-права**: любой доступ к продукту (`ro` и выше: владелец, грант, менеджер, админ) даёт **docs-права**: правка страниц документации, verify (страниц/артефактов), запуск регенерации. **rw/manage** добавляет: CRUD артефактов, редактирование продукта (SPEC-002), управление пользователями продукта, удаление продукта. `viewer_global` — чистый RO без docs-прав.
3. **Маппинг**: матрица продукт×пользователь живёт в нашей БД (`product_grants` + `owner_id`); группы Keycloak — только глобальные роли; справочник групп подтягивается через Keycloak Admin API (service account, client credentials) для настройки маппинга роль↔группа.
4. **Админ-группа отделена**: роль `admin` назначается только через группу Keycloak (или локально админом через существующий users API); в UI маппинга — отдельная секция, недоступная менеджерам.

## Скоуп

### Backend
- `api/auth/deps.py`: `resolve_product_access` возвращает уровень + производные флаги (`can_docs`, `can_manage`); новый dependency `require_product_permission("docs"|"manage")`.
- Переклассифицировать эндпоинты: docs-уровень — правка документации (`PUT .../codebases/{id}`, страницы), `verify*`, `generate`, `pages/{page_id}/regenerate`; manage-уровень — CRUD артефактов, `DELETE product`, `PATCH product`.
- Гранты продукта: `GET/PUT/DELETE /api/products/{product_id}/grants` — право manage на продукте (админ всегда; менеджер-владелец или с грантом rw). Существующие `/api/admin/grants` не трогать.
- Keycloak Admin API-клиент в `api/auth/keycloak.py` + `GET /api/admin/keycloak/groups` (только админ) — список групп для настройки маппинга.
- `/api/auth/me` — отдавать роль как сейчас (продуктовые права фронт выводит из owner_id + грантов в payload продукта).

### Frontend
- Админ: экран «Маппинг доступа» — матрица продукты×пользователи (владелец, гранты ro/rw), поиск/фильтры; секция глобальных ролей↔групп Keycloak (со справочником групп).
- Страница продукта: секция «Пользователи продукта» (видима при manage) — назначение/снятие грантов.
- Гейтинг кнопок: docs-действия — при любом доступе; manage-действия — при manage. Скрыть админку для не-админов (проверить существующее поведение).

### Keycloak
- `keycloak/realm-export.json`: отдельные группы `productarium-admins`, `productarium-managers`, `productarium-users`; обновить `.env.example`/README по маппингу.

### DB
- Миграций не требуется (`ProductGrantORM` достаточно).

## Не-цели

- Row-level security в Postgres, аудит-лог действий, SSO кроме Keycloak, кросс-продуктовые роли.

## Критерии приёмки

- `user` с грантом `ro`: читает продукт, правит доки, verify, регенерация — OK; добавление артефакта — 403; не видит чужие продукты без гранта.
- `manager`: создаёт продукт, управляет его грантами, видит все продукты; `/admin` — 403/скрыт.
- `viewer_global`: только чтение, docs-прав нет.
- Герметичные тесты: матрица доступов (`tests/unit/`), эндпоинты грантов, кейклок-маппинг групп→ролей (`tests/unit/test_auth_keycloak.py`).

## Затрагиваемые файлы

`api/auth/deps.py`, `api/auth/keycloak.py`, `api/auth/router.py`, `api/routers/products.py`, `databases.py`, `docgen.py`, `hld.py`, `admin.py`, `api/models.py` (комментарии), `src/app/admin/page.tsx`, `src/app/products/[productId]/page.tsx`, `src/lib/types.ts`, `keycloak/realm-export.json`, `tests/unit/test_rbac_*.py`, `tests/integration/test_auth_router.py`.
