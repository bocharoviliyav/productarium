# Productarium — Техническая спецификация

## Обзор проекта

**Productarium** — продакто-центричная платформа документации, генерирующая структурированную, навигируемую документацию из кодовых баз, спецификаций и внешних источников знаний с использованием **полностью локальных LLM** (Ollama или любой OpenAI-совместимый сервер). Облачные API-ключи не требуются.

**Продукт** (микросервис, монолит или databus-сервис) владеет **Артефактами** (codebase, spec, links, documentation, guides) и деревом **узлов знаний** (Knowledge Nodes). Каждый артефакт документируется, индексируется в графе знаний **cognee** (Postgres + pgvector) и становится доступным для запросов через экспертного чат-агента и RAG.

### Ключевые характеристики

- **Продукто-центричная модель**: Продукты → Артефакты (codebase, spec, links, documentation, guides) + дерево знаний
- **Локальные LLM**: Ollama (по умолчанию) или локальный OpenAI-совместимый API — без облачных ключей
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
│   Фронтенд      │     │   Бэкенд        │     │   Ollama /      │
│   (Next.js 15)  │◄───►│   (FastAPI)     │◄───►│   Локальный LLM │
│   Порт: 3000    │     │   Порт: 8001    │     │   Порт: 11434   │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                        ┌────────▼────────┐
                        │   Postgres +    │
                        │   pgvector      │
                        │   (cognee +     │
                        │    продукты)    │
                        └─────────────────┘
```

### Поток данных (Продукт → Артефакт → Документация)

1. Создаётся **Продукт** и добавляются **Артефакты** (codebase через URL репозитория, содержимое спецификации, документация, ссылки, руководства).
2. **Генерация** доков:
   - **codebase**: бэкенд клонирует репозиторий (shallow, `--depth=1`), читает файлы, собирает длинный контекст → **fast-rlm** (если ≥20k символов) или стандартный LLM генерирует 7 секций wiki из `refs/prompts/*.md` → `generated_docs` + `pages` сохраняются → репозиторий индексируется в cognee (в фоне).
   - **spec**: спецификация парсится (stdlib json/yaml) → markdown-скелет + LLM-обогащение → индексация в cognee.
   - **documentation/guides**: содержимое рендерится/обогащается через LLM → индексация в cognee.
3. Фронтенд-вьюер рендерит `artifact.pages` (дерево навигации) + markdown/Mermaid; **Экспертный агент** и панель **Ask** используют RAG (FAISS, top_k=20) с дополнением через cognee.

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

Главное FastAPI-приложение. CRUD продуктов/артефактов, `/api/products/{id}/artifacts/{id}/generate`, `/api/rlm/run`, legacy wiki-кеш, конфиг моделей. `startup_event()` вызывает `init_db()`, затем `init_cognee()`. Подключает все роутеры через `include_all_routers(app)`.

#### `api/routers/` — Авто-обнаруживаемые роутеры

Роутеры авто-обнаруживаются: добавьте `api/routers/<name>.py` с модулем-уровня `router = APIRouter(...)` — он подключится автоматически.

- **`admin.py`** — CRUD под защитой админа + тесты связности для `models`, `git`, `confluence`, `integrations`, `users`, `apitokens`. Секреты шифруются при сохранении, редактируются при чтении.
- **`auth/`** (в `api/auth/`) — Локальный login/me/logout, настройка при первом запуске, смена/сброс пароля, Keycloak OIDC login/callback.
- **`expert.py`** — Экспертный агент: SSE-чат (`POST /api/products/{id}/ask`) + генерация документа (`POST /api/products/{id}/ask/doc`).
- **`integrations.py`** — Список/тест/pull из коннекторов интеграций; создание артефактов или узлов знаний из подтянутого контента.
- **`knowledge.py`** — CRUD дерева знаний, загрузка через markitdown, переключатель verified, AI-саммари продукта.
- **`public.py`** — Эндпоинты с API-токен-аутентификацией: экспорт верифицированных знаний, ask, пуш в Confluence/git.

#### `api/auth/` — Пакет аутентификации

- **`local.py`** — Локальный логин/пароль (passlib bcrypt + JWT-сессия в cookie).
- **`keycloak.py`** — Keycloak OIDC (authlib).
- **`deps.py`** — FastAPI-зависимости (`get_current_user`, `require_admin`).
- **`tokens.py`** — API-токены (sha256-хэш; plaintext возвращается один раз).
- **`bootstrap.py`** — Одноразовый bootstrap-админ через `BOOTSTRAP_ADMIN_*`.

`AUTH_PROVIDER` выбирает режим: `local` | `keycloak` | `both` | `none`.

#### `api/integrations/` — Фреймворк коннекторов

Масштабируемый фреймворк. Авто-обнаружение через `pkgutil`. Каждый коннектор реализует `test()`, `list_spaces()`, `pull(source_id, opts)`:

- **`github.py`** / **`gitlab.py`** — Список репозиториев, клонирование + документирование как артефакты `codebase`.
- **`confluence.py`** — Список пространств, pull страниц (рекурсивно, с конвертацией вложений через markitdown).
- **`mcp.py`** — Model Context Protocol. Транспорт `http` (JSON-RPC `initialize` + `tools/call`) и `stdio` (документированный stub).
- **`base.py`** / **`registry.py`** — Базовый класс `IntegrationConnector` + реестр авто-обнаружения.
- **`_git_base.py`** — Общая логика git-коннекторов.

Добавьте новый `api/integrations/<name>.py` с подклассом `IntegrationConnector` — без правок ядра.

#### `api/rag.py`

Реализация RAG. Кастомные `Memory`/`CustomConversation`/`DialogTurn` (обход adalflow bug). `RAG` класс управляет retriever, FAISS-индексами, запросами.

Параметры RAG: text splitter (350 слов, 100 overlap), retriever top_k=20.

#### `api/data_pipeline.py`

Клонирование репозиториев (GitHub/GitLab, shallow `--depth=1`), чтение файлов с include/exclude фильтрами, `DatabaseManager` (FAISS-индексы).

#### `api/artifact_docgen.py`

Пайплайн документирования артефактов. `generate_artifact_documentation()` диспетчеризует по типу: codebase→RLM (fast-rlm, если ≥20k символов) или стандартный LLM; spec→stdlib parse + markdown skeleton + LLM enrichment; documentation/guides→LLM. Все пути индексируют в cognee в фоне и сохраняют `generated_docs` + `pages`.

#### `api/expert_agent.py`

Экспертный агент (cognee-recall + маршрутизация RLM + стриминг LLM). Тела промптов в `refs/prompts/expert_agent_*.md`.

#### `api/cognee_manager.py`

Интеграция cognee. Настраивается на локальный Ollama для LLM (`cognify` entity extraction) и эмбеддингов — без облачных ключей. `init_cognee()`, `add_and_index_document()`, `query_cognee()` — все async, все non-fatal.

#### `api/rlm_runner.py`

Обёртка fast-rlm (Deno + Pyodide). `run_rlm_task(query, model)` — async, для длинноконтекстных рассуждений. По умолчанию указывает на local Ollama.

#### `api/settings_store.py`

Шифруемое key/value-хранилище настроек админки (модели, git-credentials, confluence, интеграции). Fernet-шифрование через `SETTINGS_SECRET_KEY`.

#### `api/models.py`

SQLAlchemy 2.0 ORM: `UserORM`, `ProductORM`, `ArtifactORM`, `KnowledgeNodeORM`, `SettingORM`, `ApiTokenORM`.

- **ProductORM**: `id, name, description, type (microservice|monolith|databus_service), artifacts[], created_at, updated_at`.
- **ArtifactORM**: `id, product_id (FK, cascade delete), name, type (codebase|spec|links|documentation|guides), repo_url, repo_type, token, content, allure_url, generated_docs, pages (JSON), created_at, updated_at`.
- **KnowledgeNodeORM**: `id, product_id (FK), parent_id, title, type (page|folder|branch), content, source, verified, created_at, updated_at`.

Persisted in Postgres via SQLAlchemy (`api/db.py`: `init_db()` on startup, `get_db()` FastAPI dependency).

#### `api/prompts.py`

Реестр и загрузчик промптов. `WIKI_SECTIONS`, `wrap_prompt()`, `load_prompt_file()` загружают тела из `refs/prompts/*.md`. Тела промптов НЕ в коде (внешние — быстрые правки без изменения кода). Подстановка через `str.replace` (не `.format`) — Mermaid/JSON literal braces остаются неэкранированными.

#### `api/wiki_generator.py`

Последовательная 7-секционная генерация wiki (Overview → Architecture → Functional → Technical → CI/CD → LLD → Data Model). Каждая секция строит на предыдущих. Тела секций из `refs/prompts/<section>.md`.

#### `api/config.py`

Центральная конфигурация. JSON из `api/config/`, `${ENV_VAR}` плейсхолдеры, провайдеры/эмбеддеры. Ключевые глобалы: `configs`, `OLLAMA_HOST`, `LOCAL_OPENAI_BASE_URL`.

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
│   │   ├── page.tsx                          # Детали продукта + артефакты
│   │   └── artifacts/[artifactId]/page.tsx   # Вьюер доков артефакта
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

- **GitHub** / **GitLab** — список репозиториев, клонирование + документирование как артефакты `codebase`.
- **Confluence** — список пространств, pull страниц (рекурсивно, с конвертацией вложений через markitdown) как артефакты `documentation` или узлы знаний.
- **MCP** — коннектор Model Context Protocol. Транспорт `http` (JSON-RPC `initialize` + `tools/call`) и `stdio`.

Админы настраивают коннекторы (credentials шифруются в settings store) и тестируют связность из админ-панели. Подтянутый контент индексируется в продуктовом датасете cognee `prod_{product_id}` в фоне.

---

## Дерево знаний

У каждого Продукта есть дерево **узлов знаний** в стиле Confluence (`page`/`folder`/`branch`). Узлы можно создавать вручную, подтягивать из интеграций Confluence/MCP или загружать как файлы (конвертация в Markdown через markitdown). Узлы и артефакты можно отмечать как **верифицированные**; только верифицированный контент экспортируется или пушится через публичный API.

---

## Экспертный агент

`POST /api/products/{id}/ask` стримит ответ экспертного чата как SSE; `POST /api/products/{id}/ask/doc` генерирует самодостаточный Markdown-документ. Агент использует cognee-recall + маршрутизацию RLM + стриминг локального LLM.

---

## Конфигурация системы

### Файлы конфигурации (`api/config/`)

JSON-файлы с поддержкой `${ENV_VAR}` плейсхолдеров (разрешаются при загрузке в `config.py`):

1. **`generator.json`** — Провайдеры и модели LLM (`ollama`, `openai_local`).
2. **`embedder.json`** — Модели эмбеддингов, retriever (`top_k: 20`), text splitter (350 слов, 100 overlap).
3. **`repo.json`** — Фильтры файлов (исключаемые директории/файлы) и лимиты размера репозитория.

Кастомная директория конфигов через `DEEPWIKI_CONFIG_DIR`.

### Переменные окружения

Полный список — в `.env.example`. Ключевые группы:

- **Ollama**: `OLLAMA_HOST` (по умолчанию `http://localhost:11434`).
- **Postgres**: `DB_PROVIDER`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USERNAME`, `DB_PASSWORD`; `VECTOR_DB_PROVIDER=pgvector`.
- **cognee LLM (local Ollama)**: `LLM_PROVIDER=ollama`, `LLM_ENDPOINT`, `LLM_MODEL`, `LLM_API_KEY=not-needed`.
- **cognee embeddings**: `EMBEDDING_PROVIDER=ollama`, `EMBEDDING_MODEL=nomic-embed-text`, `EMBEDDING_ENDPOINT`, `EMBEDDING_DIMENSIONS=768`, `HUGGINGFACE_TOKENIZER`.
- **fast-rlm**: `RLM_MODEL_BASE_URL`, `RLM_MODEL_API_KEY=not-needed`, `RLM_MODEL_NAME`.
- **Auth**: `AUTH_PROVIDER`, `BOOTSTRAP_ADMIN_USERNAME`, `BOOTSTRAP_ADMIN_PASSWORD`, `SETTINGS_SECRET_KEY`.
- **Keycloak**: `KEYCLOAK_URL`, `KEYCLOAK_CLIENT_ID`, `KEYCLOAK_CLIENT_SECRET`, `KEYCLOAK_REALM`.
- **Enterprise Git**: `GITHUB_ENTERPRISE_URL`, `GITLAB_SELF_HOSTED_URL`.
- **Логирование**: `LOG_LEVEL`, `LOG_FILE_PATH`.

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
- **Ollama** с моделями:
  ```bash
  ollama pull qwen3:8b           # или qwen3.5:9b, gemma3:12b, etc.
  ollama pull nomic-embed-text   # обязательно для эмбеддингов
  ```
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
