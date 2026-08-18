# Productarium — Техническая спецификация

## Обзор проекта

**Productarium** — продакто-центричная платформа документации, генерирующая структурированную, навигируемую документацию из кодовых баз, спецификаций и внешних источников знаний с использованием **полностью локальных LLM** (любой OpenAI-совместимый сервер: LM Studio, llama.cpp, vLLM). Облачные API-ключи не требуются.

**Продукт** (микросервис, монолит или databus-сервис) владеет **Артефактами** (codebase, spec, links, documentation, guides) и деревом **узлов знаний** (Knowledge Nodes). Каждый артефакт документируется, индексируется в графе знаний **cognee** (Postgres + pgvector) и становится доступным для запросов через экспертного чат-агента и RAG.

### Ключевые характеристики

- **Продукто-центричная модель**: Продукты → Артефакты (codebase, spec, links, documentation, guides) + дерево знаний
- **Локальные LLM**: локальный OpenAI-совместимый API (LM Studio, llama.cpp, vLLM) — без облачных ключей
- **Граф знаний**: cognee + pgvector индексируют артефакты и узлы знаний для RAG
- **Экспертный агент**: стриминг-чат (SSE) + генерация самодостаточного Markdown-документа
- **fast-rlm**: Recursive Language Models для длинноконтекстной генерации (≥20k символов) и Deep Research
- **Аутентификация**: локальная (passlib bcrypt + JWT) и/или Keycloak OIDC
- **Админ-панель**: модели, git-credentials, Confluence, интеграции (MCP + web), пользователи, API-токены
- **Интеграции**: GitHub, GitLab, Confluence, MCP (stdio/http) — авто-обнаружение через реестр
- **Верифицированные знания**: отметка знаний как верифицированных; экспорт через публичный API

---

## Архитектура системы

### Общая схема

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Фронтенд      │     │   Бэкенд        │     │   OpenAI-совм. /│
│   (Next.js 15)  │◄───►│   (FastAPI)     │◄───►│   Локальный LLM │
│   Порт: 3000    │     │   Порт: 8001    │     │   Порт: 1234    │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                        ┌────────▼────────┐
                        │   Postgres +    │
                        │   pgvector      │
                        │   (cognee +     │
                        │    продукты)    │
                        └─────────────────┘
```

### Поток данных (Продукт → Codebase/Spec → Документация)

1. Создаётся **Продукт** и добавляются типизированные сущности **Codebase** (через URL репозитория), **Spec** (содержимое спецификации) или **Links** (JSON-массив ссылок).
2. **Генерация** доков (per-type эндпоинты):
   - **codebase** (`POST /api/products/{id}/codebases/{id}/generate`): бэкенд клонирует репозиторий (shallow, `--depth=1`), читает файлы, собирает длинный контекст → **fast-rlm** (если ≥20k символов) или стандартный LLM генерирует 7 секций wiki из `refs/prompts/*.md` → `generated_docs` + `pages` сохраняются → репозиторий индексируется в cognee (в фоне, async 202+poll).
   - **spec** (`POST /api/products/{id}/specs/{id}/generate`): спецификация парсится (stdlib json/yaml) → markdown-скелет + LLM-обогащение → индексация в cognee.
   - **links**: генерации нет (только хранение содержимого).
3. Фронтенд-вьюер рендерит `codebase.pages` (дерево навигации) + markdown/Mermaid; **Экспертный агент** и панель **Ask** используют RAG (FAISS, top_k=20) с дополнением через cognee.

---

## Backend (API)

### Технологический стек

- **Python 3.11+**, **FastAPI**, **Uvicorn** (hot-reload в dev)
- **Pydantic v2** — валидация данных
- **SQLAlchemy 2.0** — ORM (Postgres)
- **adalflow** — RAG-фреймворк
- **FAISS** — векторная база данных
- **cognee** — граф знаний (Postgres + pgvector)
- **tiktoken** — токенизация
- **passlib** (bcrypt) + **authlib** (Keycloak OIDC) — аутентификация
- **cryptography** (Fernet) — шифрование секретов настроек

### Основные модули

#### `api/main.py`

Точка входа. Загружает `.env`, настраивает логирование, запускает uvicorn на порту `PORT` (8001).

#### `api/api.py`

Главное FastAPI-приложение. Подключает все роутеры через `include_all_routers(app)`. `startup_event()` вызывает `init_db()`, затем `init_cognee()` (оба non-fatal).

#### `api/routers/` — Авто-обнаруживаемые роутеры

Роутеры авто-обнаруживаются: добавьте `api/routers/<name>.py` с модулем-уровня `router = APIRouter(...)` — он подключится автоматически.

- **`admin.py`** — CRUD под защитой админа + тесты связности для `models`, `git`, `confluence`, `integrations`, `rlm`, `ssl`, `cognee`, `timeouts`, `users`, `apitokens`, `prompts`. Секреты шифруются при сохранении, редактируются при чтении.
- **`auth/`** (в `api/auth/`) — Локальный login/me/logout, настройка при первом запуске, смена/сброс пароля, Keycloak OIDC login/callback.
- **`docgen.py`** — Per-type генерация: `POST .../codebases/{id}/generate` + status, `POST .../specs/{id}/generate` + status. Links не генерируются.
- **`expert.py`** — Экспертный агент: SSE-чат (`POST /api/products/{id}/ask`) + генерация документа (`POST /api/products/{id}/ask/doc`).
- **`integrations.py`** — Список/тест/pull из коннекторов интеграций; git-коннекторы создают `CodebaseORM`, non-git — узлы знаний.
- **`knowledge.py`** — CRUD дерева знаний, загрузка через markitdown, переключатель verified, AI-саммари продукта (`generate_product_summary(product, codebases, specs, nodes)`).
- **`products.py`** — Per-type создание/удаление/обновление: `POST/DELETE/PUT .../codebases|specs|links/{id}`.
- **`public.py`** — Эндпоинты с API-токен-аутентификацией: экспорт верифицированных знаний (markdown/json с ключами `codebases`/`specs`/`links`/`nodes`), ask, пуш в Confluence/git.

#### `api/auth/` — Пакет аутентификации

- **`local.py`** — Локальный логин/пароль (passlib bcrypt + JWT-сессия в cookie).
- **`keycloak.py`** — Keycloak OIDC (authlib).
- **`deps.py`** — FastAPI-зависимости (`get_current_user`, `require_admin`).
- **`tokens.py`** — API-токены (sha256-хэш; plaintext возвращается один раз).
- **`bootstrap.py`** — Одноразовый bootstrap-админ через `BOOTSTRAP_ADMIN_*`.

`AUTH_PROVIDER` выбирает режим: `local` | `keycloak` | `both` | `none`.

#### `api/integrations/` — Фреймворк коннекторов

Масштабируемый фреймворк. Авто-обнаружение через `pkgutil`. Каждый коннектор реализует `test()`, `list_spaces()`, `pull(source_id, opts)`:

- **`github.py`** / **`gitlab.py`** — Список репозиториев, клонирование + документирование как `CodebaseORM` (`repo_url`/`repo_type`).
- **`confluence.py`** — Список пространств, pull страниц (рекурсивно, с конвертацией вложений через markitdown) как узлы знаний.
- **`mcp.py`** — Model Context Protocol. Транспорт `http` (JSON-RPC `initialize` + `tools/call`) и `stdio` (документированный stub).
- **`base.py`** / **`registry.py`** — Базовый класс `IntegrationConnector` + реестр авто-обнаружения.
- **`_git_base.py`** — Общая логика git-коннекторов.

Добавьте новый `api/integrations/<name>.py` с подклассом `IntegrationConnector` — без правок ядра.

#### `api/rag.py`

Реализация RAG. Кастомные `Memory`/`CustomConversation`/`DialogTurn` (обход adalflow bug). `RAG` класс управляет retriever, FAISS-индексами, запросами.

Параметры RAG: text splitter (350 слов, 100 overlap), retriever top_k=20.

#### `api/data_pipeline.py`

Клонирование репозиториев (GitHub/GitLab, shallow `--depth=1`), чтение файлов с include/exclude фильтрами, `DatabaseManager` (FAISS-индексы).

#### `api/docgen/`

Пакет генерации документации. Без диспетчера — каждый generate-эндпоинт вызывает свой генератор напрямую. `codebase.py:generate_codebase_docs` (RLM для длинного контекста ≥20k символов, иначе стандартный LLM, 7 секций), `spec.py:generate_openapi_docs`/`generate_asyncapi_docs` (stdlib parse + skeleton + LLM enrichment). `jobs.py` (async 202+poll worker, принимает `entity_type`). `_common.py` (общий `_StandardLLM`, `_index_in_background`). Все пути индексируют в cognee в фоне и сохраняют `generated_docs` + `pages`.

#### `api/expert/`

Пакет экспертного агента. `chat.py` (cognee-recall + маршрутизация RLM + стриминг LLM), `generate.py` (standalone doc). Тела промптов в `refs/prompts/expert_agent_*.md`.

#### `api/cognee/`

Интеграция cognee. `_runtime.py` настраивает локальный OpenAI-совместимый сервер для LLM (`cognify` entity extraction) и эмбеддингов — без облачных ключей. `init_cognee()`, `add_and_index_document()`, `query_cognee()`, `reindex_product_knowledge_graph()` — все async, все non-fatal.

#### `api/rlm/runner.py`

Обёртка fast-rlm (Deno + Pyodide). `run_rlm_task(query, model)` — async, для длинноконтекстных рассуждений. Единый путь: admin config → `LOCAL_OPENAI_BASE_URL` → default.

#### `api/config/`

Центральный пакет конфигурации. `__init__.py` (JSON-загрузчик, `${ENV_VAR}` плейсхолдеры), `settings.py` (шифруемое key/value-хранилище, Fernet через `SETTINGS_SECRET_KEY`), `timeout.py` (per-key таймауты), `ssl.py` (TLS-патч для корпоративных шлюзов).

#### `api/models.py`

SQLAlchemy 2.0 ORM: `UserORM`, `ProductORM`, `CodebaseORM`, `SpecORM`, `LinksORM`, `KnowledgeNodeORM`, `SettingORM`, `ApiTokenORM`.

- **ProductORM**: `id, name, summary, owner_id, created_at, updated_at`. Владеет `codebases`, `specs`, `links`.
- **CodebaseORM** (таблица `codebases`): `id, product_id (FK CASCADE), name, repo_url, repo_type, token, generated_docs, pages (JSON), verified, source, timestamps`.
- **SpecORM** (таблица `specs`): `id, product_id (FK CASCADE), name, kind (openapi|asyncapi), content, verified, source, timestamps`.
- **LinksORM** (таблица `links`): `id, product_id (FK CASCADE), name, content (JSON-массив), verified, source, timestamps`.
- **KnowledgeNodeORM**: `id, product_id (FK), parent_id, title, slug, node_type (page|folder|branch), content_md, source, verified, created_at, updated_at`.

Persisted in Postgres via SQLAlchemy (`api/db.py`: `init_db()` on startup — `Base.metadata.create_all`, идемпотентно, non-fatal; `get_db()` FastAPI dependency).

#### `api/prompts.py`

Реестр и загрузчик промптов. `load_prompt_file()` применяет `_wrap_prompt(content, language)` после загрузки. `WIKI_SECTIONS`, `load_prompt_file()` загружают тела из `refs/prompts/*.md`. Тела промптов НЕ в коде (внешние — быстрые правки без изменения кода). Подстановка через `str.replace` (не `.format`) — Mermaid/JSON literal braces остаются неэкранированными.

#### `api/docgen/codebase.py`

Последовательная 7-секционная генерация wiki (Overview → Architecture → Functional → Technical → CI/CD → LLD → Data Model). Каждая секция строит на предыдущих. Тела секций из `refs/prompts/<section>.md`.

#### `api/config/__init__.py`

Центральная конфигурация. JSON из `api/config/`, `${ENV_VAR}` плейсхолдеры. Единый OpenAI-compatible путь (один клиент покрывает LM Studio, llama.cpp, vLLM). Ключевые глобалы: `configs`, `LOCAL_OPENAI_BASE_URL`.

---

## Frontend (Next.js 15)

### Технологический стек

- **Next.js 15** (App Router, Turbopack), **React 19**, **TypeScript 5**
- **Tailwind CSS 4**, **next-intl** (i18n), **next-themes** (dark/light)
- **mermaid** (диаграммы), **react-markdown** + rehype-raw + remark-gfm (Markdown)
- **react-syntax-highlighter** (подсветка), **svg-pan-zoom** (SVG pan/zoom)
- **bun** (package manager)

### Структура директорий

```
src/
├── app/
│   ├── page.tsx                              # Дашборд продуктов
│   ├── products/[productId]/
│   │   ├── page.tsx                          # Детали продукта + codebases/specs/links
│   │   └── artifacts/[artifactId]/page.tsx   # Универсальный вьюер доков (codebase/spec/links)
│   ├── api/                                  # Next.js API routes (прокси к backend)
│   └── wiki/projects/page.tsx               # Legacy: список проектов
├── components/
│   ├── Ask.tsx                               # Чат с RAG + Deep Research
│   ├── Markdown.tsx                          # Рендеринг Markdown
│   ├── Mermaid.tsx                           # Mermaid с pan/zoom + auto-fix
│   └── ui.tsx                                # Shared minimalist-ui primitives
├── contexts/
│   └── LanguageContext.tsx                   # i18n (next-intl)
├── lib/
│   └── types.ts                              # Общие типы фронтенда
└── utils/
    └── websocketClient.ts                    # WebSocket клиент для /ws/chat
```

### API Proxy Pattern

Фронтенд не вызывает backend напрямую из браузера для большинства эндпоинтов. `next.config.ts` определяет rewrites, проксирующие `/api/*` запросы к `SERVER_BASE_URL` (по умолчанию `http://localhost:8001`). WebSocket-соединения идут напрямую к backend.

---

## Аутентификация

`AUTH_PROVIDER` выбирает режим:

- **`local`** (по умолчанию) — логин/пароль (passlib bcrypt + JWT-сессия в cookie). Настройка при первом запуске создаёт админа через UI или `BOOTSTRAP_ADMIN_*`.
- **`keycloak`** — OIDC-логин через Keycloak (`authlib`). Требуются `KEYCLOAK_*` переменные.
- **`both`** — локальный + Keycloak эндпоинты одновременно.
- **`none`** — аутентификация отключена; `get_current_user` возвращает системного админа.

Админы управляют пользователями (создание с временным паролем + reset-токеном, повышение/понижение роли) и API-токенами (sha256-хэш).

---

## Интеграции

Коннекторы авто-обнаруживаются в `api/integrations/`. Каждый реализует `test()`, `list_spaces()` и `pull(source_id, opts)`:

- **GitHub** / **GitLab** — список репозиториев, клонирование + документирование как `CodebaseORM` (`repo_url`/`repo_type`).
- **Confluence** — список пространств, pull страниц (рекурсивно, с конвертацией вложений через markitdown) как узлы знаний.
- **MCP** — коннектор Model Context Protocol. Транспорт `http` (JSON-RPC `initialize` + `tools/call`) и `stdio`.

Админы настраивают коннекторы (credentials шифруются в settings store) и тестируют связность из админ-панели. Подтянутый контент индексируется в продуктовом датасете cognee `prod_{product_id}` в фоне.

---

## Дерево знаний

У каждого Продукта есть дерево **узлов знаний** в стиле Confluence (`page`/`folder`/`branch`). Узлы можно создавать вручную, подтягивать из интеграций Confluence/MCP или загружать как файлы (конвертация в Markdown через markitdown). Узлы, codebases, specs и links можно отмечать как **верифицированные**; только верифицированный контент экспортируется или пушится через публичный API.

---

## Экспертный агент

`POST /api/products/{id}/ask` стримит ответ экспертного чата как SSE; `POST /api/products/{id}/ask/doc` генерирует самодостаточный Markdown-документ. Агент использует cognee-recall + маршрутизацию RLM + стриминг локального LLM.

---

## Конфигурация системы

### Файлы конфигурации (`api/config/`)

JSON-файлы с поддержкой `${ENV_VAR}` плейсхолдеров (разрешаются при загрузке в `config.py`):

1. **`generator.json`** — Провайдеры и модели LLM (`openai`, `openai_local`).
2. **`embedder.json`** — Модели эмбеддингов, retriever (`top_k: 20`), text splitter (350 слов, 100 overlap).
3. **`repo.json`** — Фильтры файлов (исключаемые директории/файлы) и лимиты размера репозитория.

Кастомная директория конфигов через `DEEPWIKI_CONFIG_DIR`.

### Переменные окружения

Полный список — в `.env.example`. Ключевые группы:

- **Postgres**: `DB_PROVIDER`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USERNAME`, `DB_PASSWORD`; `VECTOR_DB_PROVIDER=pgvector`.
- **cognee LLM (local OpenAI-compatible)**: `LLM_PROVIDER=openai`, `LLM_ENDPOINT`, `LLM_MODEL`, `LLM_API_KEY=not-needed`.
- **cognee embeddings**: `EMBEDDING_PROVIDER=openai_compatible`, `EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5`, `EMBEDDING_ENDPOINT`, `EMBEDDING_DIMENSIONS=768`, `HUGGINGFACE_TOKENIZER`.
- **fast-rlm**: `RLM_MODEL_BASE_URL`, `RLM_MODEL_API_KEY=not-needed`, `RLM_MODEL_NAME`.
- **Auth**: `AUTH_PROVIDER`, `BOOTSTRAP_ADMIN_USERNAME`, `BOOTSTRAP_ADMIN_PASSWORD`, `SETTINGS_SECRET_KEY`.
- **Keycloak**: `KEYCLOAK_URL`, `KEYCLOAK_CLIENT_ID`, `KEYCLOAK_CLIENT_SECRET`, `KEYCLOAK_REALM`.
- **Enterprise Git**: `GITHUB_ENTERPRISE_URL`, `GITLAB_SELF_HOSTED_URL`.
- **Логирование**: `LOG_LEVEL`, `LOG_FORMAT` (`logfmt`/`json`).

### Системные промпты

Все тела промптов вынесены в `refs/prompts/*.md` (7 секций wiki, доки для spec/documentation/testcase, экспертный агент, итерации deep research, системные промпты RAG/simple-chat). Редактируйте напрямую — без изменения кода.

---

## Генерация wiki

### 7-секционный pipeline

Wiki генерируется последовательно в 7 этапов, каждый строит на предыдущих:

1. **Overview** — общая информация о проекте
2. **Architecture** — системная архитектура (C4 нотация, Mermaid-диаграммы)
3. **Functional** — функциональное описание (use cases, API endpoints)
4. **Technical** — технические детали (технологический стек, конфигурация)
5. **CI/CD** — конвейеры развёртывания (Docker, GitHub Actions, etc.)
6. **LLD** — Low Level Design (по шаблону `refs/LLD.md`)
7. **Data Model** — модель данных (по шаблону `refs/DataModel.md`)

### Deep Research

Многораундовый режим углублённого исследования:

- Активируется тегом `[DEEP RESEARCH]` в сообщении
- До 5 итераций исследования
- Отдельные промпты для первой, промежуточных и финальной итераций
- Каждая итерация строит на результатах предыдущих

---

## Docker-развёртывание

`docker-compose.yml` поднимает:

- **postgres** — `pgvector/pgvector:pg18-trixie` (Postgres + pgvector для продуктов/артефактов и cognee).
- **deepwiki** — приложение (FastAPI на `:8001`, Next.js на `:3000`).

Данные сохраняются в `~/.adalflow` (репозитории, FAISS-индексы, wiki-кеш) и в volume `postgres_data`.

---

## Локальная разработка

### Предварительные требования

- **Python 3.11+** и **Node.js** с **bun**
- **Локальный OpenAI-совместимый сервер** (LM Studio, llama.cpp, vLLM), запущенный с generation-моделью и embedding-моделью (например, `qwen/qwen3.6-27b` и `text-embedding-nomic-embed-text-v1.5`)
- **PostgreSQL + pgvector** (опционально; `docker-compose up postgres`)

### Запуск

**Backend**:
```bash
python -m pip install poetry==2.0.1 && poetry install -C api
python -m api.main              # uvicorn на порту 8001 (hot-reload в dev)
```

**Frontend**:
```bash
bun install
bun run dev        # порт 3000 с turbopack
bun run build      # production build
bun run lint       # ESLint
```

### Тестирование

Единый набор тестов в директории `tests/`:
```bash
# Pytest (единая директория tests/)
pytest                                  # все тесты в tests/
pytest tests/unit/                       # только unit-тесты
pytest tests/integration/                # только интеграционные тесты
pytest tests/unit/test_extract_repo_name.py # одиночный файл

# Запуск через тестовый раннер
python tests/run_tests.py               # все категории
python tests/run_tests.py --unit        # tests/unit/
python tests/run_tests.py --integration # tests/integration/
```

Pytest конфиг в `pytest.ini` (testpaths=tests, strict markers, short tracebacks).

---

## Кеширование

1. **Wiki-кеш** (`~/.adalflow/wikicache/`) — JSON-файлы со структурой wiki и сгенерированными страницами.
2. **FAISS-индексы** (`~/.adalflow/databases/`) — векторные индексы для RAG.
3. **Репозитории** (`~/.adalflow/repos/`) — клонированные Git-репозитории (shallow clone, `--depth=1`).

---

## Лицензия

MIT License — подробности в файле `LICENSE`.

Productarium — форк проекта deepwiki-open (MIT).
