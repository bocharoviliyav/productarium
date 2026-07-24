# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Build & Development Commands

### Prerequisites
- **Python 3.11+** and **Node.js** with **bun** (the frontend uses bun, not yarn)
- **Ollama** must be running locally with at least one generation model and the embedding model:
  ```bash
  ollama pull qwen3:8b           # or qwen3.5:9b, gemma3:12b, etc.
  ollama pull nomic-embed-text   # required for embeddings
  ```
- **PostgreSQL + pgvector** for product/artifact persistence and the cognee knowledge graph.
  `docker-compose up postgres` starts `pgvector/pgvector:pg18-trixie` (user/db: `cognee`/`cognee_db`).
  If Postgres is unreachable, `init_db()`/`init_cognee()` log a warning and fall back (app still starts).

### Backend (Python FastAPI)
```bash
python -m pip install poetry==2.0.1 && poetry install -C api
python -m api.main              # starts uvicorn on port 8001 (hot-reload in dev)
```

### Frontend (Next.js 15 + Turbopack, bun)
```bash
bun install
bun run dev        # port 3000 with turbopack
bun run build      # production build
bun run lint       # ESLint (next/core-web-vitals + next/typescript)
```

### Testing
Single unified test suite in the `tests/` directory:
```bash
pytest                                  # runs all tests in tests/
pytest tests/unit/                       # unit tests only
pytest tests/integration/                # integration tests only
pytest tests/unit/test_extract_repo_name.py # single test file

# Alternative via test runner script
python tests/run_tests.py               # all categories
python tests/run_tests.py --unit        # tests/unit/
python tests/run_tests.py --integration # tests/integration/
```
Pytest config is in `pytest.ini` (testpaths=test, strict markers, short tracebacks).

### Docker
```bash
docker-compose up       # builds + runs (postgres, API port 8001, frontend port 3000)
```
Data persists in `~/.adalflow` mounted into the container; Postgres data in the `postgres_data` volume.
The Dockerfile installs bun (frontend) and Deno (for fast-rlm) alongside Python.

## Architecture Overview

Productarium is a **product-centric** documentation platform. The top-level entity is a **Product** (microservice / monolith / databus service), which owns **Artifacts** (codebase, spec, links, documentation, guides) and a **Knowledge Node** tree. Each codebase artifact can be cloned from GitHub/GitLab and documented; specs are parsed/rendered; documentation/guides are enriched via LLM. A **cognee** knowledge graph (postgres + pgvector) indexes every artifact and knowledge node for RAG, and **fast-rlm** (Recursive Language Models) handles long-context doc generation. An **Expert Agent** provides streaming chat + document generation over indexed knowledge. Fully local — no cloud API keys required. Core framework: **adalflow** (RAG pipeline + FAISS vector storage).

### Product-Centric Data Model
- **Product** (`api/models.py:ProductORM`): `id, name, description, type (microservice|monolith|databus_service), artifacts[], created_at, updated_at`.
- **Artifact** (`api/models.py:ArtifactORM`): `id, product_id (FK, cascade delete), name, type (codebase|spec|links|documentation|guides), repo_url, repo_type, token, content, allure_url, generated_docs, pages (JSON), created_at, updated_at`.
- **KnowledgeNode** (`api/models.py:KnowledgeNodeORM`): `id, product_id (FK), parent_id, title, type (page|folder|branch), content, source, verified, created_at, updated_at`.
- **User** (`api/models.py:UserORM`): `id, username, password_hash, role (admin|user), created_at`.
- **Setting** (`api/models.py:SettingORM`): `key, value (encrypted)` — admin settings store.
- **ApiToken** (`api/models.py:ApiTokenORM`): `id, user_id (FK), name, token_hash, created_at, last_used_at`.
- Persisted in Postgres via SQLAlchemy (`api/db.py`: `init_db()` on startup, `get_db()` FastAPI dependency). String PKs (`prod_…`/`art_…`) keep frontend compatibility.
- REST: `GET/POST /api/products`, `GET/PUT/DELETE /api/products/{id}`, `POST/DELETE /api/products/{id}/artifacts`, `POST /api/products/{id}/artifacts/{id}/generate`, `POST /api/products/{id}/ask` (expert agent SSE), `POST /api/products/{id}/ask/doc` (expert doc), `POST /api/rlm/run`.

### Two-Process Architecture
- **Frontend**: Next.js app on port 3000. Proxies API calls to the backend via rewrites in `next.config.ts`.
- **Backend**: FastAPI app on port 8001 (`api/api.py` is the main app, started via `api/main.py`).
- Communication: REST (SSE streaming) + WebSocket (`/ws/chat` endpoint).

### Backend Modules (`api/`)
- **`main.py`** — Entry point. Loads `.env`, configures logging, starts uvicorn.
- **`api.py`** — Main FastAPI app. REST endpoints (legacy wiki + Product/Artifact CRUD + generate + RLM run + expert agent), Pydantic models, wiki cache management, model config. `startup_event()` calls `init_db()` then `init_cognee()`. Connects all routers via `include_all_routers(app)`.
- **`routers/`** — Auto-discovered routers (add `api/routers/<name>.py` with a module-level `router = APIRouter(...)` — it connects automatically):
  - **`admin.py`** — Admin-protected CRUD + connectivity tests for `models`, `git`, `confluence`, `integrations`, `users`, `apitokens`. Secrets encrypted on save, masked on read.
  - **`auth/`** (in `api/auth/`) — Local login/me/logout, first-run setup, password change/reset, Keycloak OIDC login/callback.
  - **`expert.py`** — Expert agent: SSE chat (`POST /api/products/{id}/ask`) + document generation (`POST /api/products/{id}/ask/doc`).
  - **`integrations.py`** — List/test/pull from integration connectors; create artifacts or knowledge nodes from pulled content.
  - **`knowledge.py`** — Knowledge tree CRUD, markitdown upload, verified toggle, AI product summary.
  - **`public.py`** — API-token-authenticated endpoints: export verified knowledge, ask, push to Confluence/git.
- **`auth/`** — Authentication package (`local.py`, `keycloak.py`, `deps.py`, `tokens.py`, `bootstrap.py`). `AUTH_PROVIDER` selects `local` | `keycloak` | `both` | `none`.
- **`integrations/`** — Scalable connector framework. Auto-discovered via `pkgutil`. Connectors: `github`, `gitlab`, `confluence`, `mcp` (stdio/http). Add new `api/integrations/<name>.py` subclassing `IntegrationConnector` — no core changes needed.
- **`rag.py`** — RAG implementation. Custom `Memory`/`CustomConversation`/`DialogTurn` classes (workaround for adalflow compatibility issues). `RAG` class manages retriever preparation, FAISS index loading, and query processing.
- **`data_pipeline.py`** — Repository processing: `download_repo()` (git clone with token auth for GitHub/GitLab), `read_all_documents()` (file reading with include/exclude filters), `DatabaseManager` (FAISS index creation/loading/saving), token counting via tiktoken.
- **`config.py`** — Central configuration. Loads JSON from `api/config/`, resolves `${ENV_VAR}` placeholders, manages provider/embedder client classes (`OllamaClient`, `OpenAIClient`). Key globals: `configs`, `OLLAMA_HOST`, `LOCAL_OPENAI_BASE_URL`.
- **`wiki_generator.py`** — Sequential 7-section wiki generation pipeline (Overview → Architecture → Functional → Technical → CI/CD → LLD → Data Model). Section bodies are loaded from `refs/prompts/<section>.md` and substituted via `str.replace` (NOT `.format` — keeps Mermaid/JSON braces unescaped). Each section builds on previous sections' content.
- **`prompts.py`** — Prompt **registry + loader only**. `WIKI_SECTIONS`, `wrap_prompt()`, and `load_prompt_file()` load bodies from `refs/prompts/*.md` with short inline fallbacks. No prompt bodies live in code (externalized for fast edits without code changes).
- **`artifact_docgen.py`** — Artifact documentation pipeline. `generate_artifact_documentation()` dispatches by type: codebase→RLM (fast-rlm) for long-context (≥20k chars) else standard LLM, 7 sections from refs; spec→stdlib parse + markdown skeleton + LLM enrichment; documentation/guides→LLM enrichment. All paths index into cognee in the background and persist `generated_docs` + `pages`.
- **`expert_agent.py`** — Expert agent (cognee-recall + RLM routing + LLM streaming). Prompt bodies in `refs/prompts/expert_agent_*.md`.
- **`settings_store.py`** — Encrypted key/value settings store for admin panel (models, git-credentials, confluence, integrations). Fernet encryption via `SETTINGS_SECRET_KEY`.
- **`openai_client.py`** — Custom OpenAI-compatible client for local LLM servers (llama.cpp, vLLM, etc.).
- **`ollama_patch.py`** — Ollama integration, model existence checks, document processing patches.
- **`tools/embedder.py`** — Factory function `get_embedder()` that creates `adal.Embedder` instances based on provider config.
- **`models.py`** — SQLAlchemy 2.0 ORM models (`UserORM`, `ProductORM`, `ArtifactORM`, `KnowledgeNodeORM`, `SettingORM`, `ApiTokenORM`).
- **`db.py`** — SQLAlchemy engine + `SessionLocal` + `get_db()` dependency + `init_db()` (`Base.metadata.create_all`, idempotent, non-fatal).
- **`cognee_manager.py`** — cognee integration. Configures cognee for **local Ollama** (LLM via `/v1`, embeddings via `/api/embed`, `LLM_API_KEY=not-needed`) so `cognify()` works with NO cloud key. `init_cognee()`, `add_and_index_document()`, `query_cognee()` — all async, all non-fatal.
- **`rlm_runner.py`** — fast-rlm wrapper. `run_rlm_task(query, model)` (async) runs Recursive Language Model reasoning; defaults to local Ollama (`RLM_MODEL_BASE_URL=http://localhost:11434/v1`). Used for long-context doc gen + Deep Research only.

### Frontend Structure (`src/`) — minimalist-ui (Notion/Linear editorial)
Visual language: warm monochrome (canvas `#FFFFFF`/`#F7F6F3`, 1px `#EAEAEA` borders), Geist font (self-hosted via `geist` pkg) + system serif for headings, Phosphor icons, bento grids, no gradients/heavy shadows, quiet motion. Built with **bun**.
- **`app/page.tsx`** — Products dashboard: bento grid of products, inline create, delete, empty state.
- **`app/products/[productId]/page.tsx`** — Product detail: header + type badge, artifacts bento, type-specific add-artifact form, per-artifact Generate, expert agent panel, knowledge tree.
- **`app/products/[productId]/artifacts/[artifactId]/page.tsx`** — Artifact docs viewer: page nav tree from `artifact.pages`, markdown + Mermaid render, scoped Ask panel (codebase artifacts).
- **`app/api/`** — Next.js API route handlers (proxied to backend for auth, chat, models, wiki).
- **`components/ui.tsx`** — Shared minimalist-ui primitives (cards, buttons, tags, inputs).
- **`lib/types.ts`** — Shared frontend types (Product, Artifact, WikiPage, etc.).
- **`components/Ask.tsx`** — Chat interface with RAG. Deep Research toggle (multi-turn, up to 5 iterations).
- **`components/Mermaid.tsx`** — Mermaid diagram renderer with SVG pan/zoom and auto-fix.
- **`components/Markdown.tsx`** — Markdown renderer (react-markdown + rehype-raw + remark-gfm + syntax highlighting).
- **`contexts/LanguageContext.tsx`** — i18n via `next-intl`. Auto-detects browser language, loads from `src/messages/{lang}.json`.
- **`utils/websocketClient.ts`** — WebSocket client for chat (`/ws/chat`).

### Data Flow (Product → Artifact → Docs)
1. User creates a **Product** (POST `/api/products`) and adds an **Artifact** (POST `/api/products/{id}/artifacts`).
2. Generate (POST `/api/products/{id}/artifacts/{id}/generate`):
   - **codebase**: backend clones to `~/.adalflow/repos/` (shallow, `--depth=1`), reads files, builds a long-context blob → **RLM** (fast-rlm, if ≥20k chars) or standard LLM generates 7 wiki sections from `refs/prompts/*.md` → `generated_docs` + `pages` persisted → repo indexed into **cognee** (background).
   - **spec**: spec parsed (stdlib json/yaml) → markdown skeleton + LLM enrichment → indexed into cognee.
   - **documentation/guides**: content enriched via LLM → indexed into cognee.
3. Frontend viewer renders `artifact.pages` (nav tree) + markdown/Mermaid; Ask panel uses RAG (FAISS, top_k=20) augmented with cognee recall over the artifact's dataset.
4. **Expert Agent** (`POST /api/products/{id}/ask`) streams SSE chat over all indexed knowledge (artifacts + knowledge nodes); `POST /api/products/{id}/ask/doc` generates a self-contained Markdown document.

### Knowledge Tree
Each Product has a tree of **Knowledge Nodes** (Confluence-style: `page`/`folder`/`branch`). Nodes can be created manually, pulled from Confluence/MCP integrations, or uploaded as files (converted to Markdown via markitdown). Nodes and artifacts can be marked **verified**; only verified content is exported or pushed via the public API.

### API Proxy Pattern
The frontend does NOT call the backend directly from the browser for most endpoints. `next.config.ts` defines rewrites that proxy `/api/*` requests to `SERVER_BASE_URL` (default `http://localhost:8001`). WebSocket connections go directly to the backend.

## Key Patterns

### JSON Configuration with Env Placeholders
`api/config/` JSON files support `${ENV_VAR}` placeholders resolved at load time by `replace_env_placeholders()` in `config.py`. Custom config directory via `DEEPWIKI_CONFIG_DIR`.

### Provider System
Two LLM providers in `api/config/generator.json`: `ollama` (default, via adalflow `OllamaClient`) and `openai_local` (via custom `OpenAIClient`). Both support custom models. Embedder provider separately controlled by `DEEPWIKI_EMBEDDER_TYPE`.

### Authentication
`AUTH_PROVIDER` selects mode: `local` (default — passlib bcrypt + JWT session cookie), `keycloak` (OIDC via authlib), `both` (local + Keycloak endpoints), or `none` (auth disabled). Bootstrap admin created on first startup via `BOOTSTRAP_ADMIN_*` env vars or UI setup. Admins manage users (create with temp password + reset token, role elevation) and API tokens (sha256 hash; plaintext shown once). Secrets in settings store encrypted via Fernet (`SETTINGS_SECRET_KEY`).

### Integration Framework
Connectors auto-discovered in `api/integrations/` via `pkgutil`. Each implements `test()`, `list_spaces()`, `pull(source_id, opts)`:
- **GitHub** / **GitLab** — list repos, clone + document as `codebase` artifacts.
- **Confluence** — list spaces, pull pages (recursively, attachments converted via markitdown) as `documentation` artifacts or knowledge nodes.
- **MCP** — Model Context Protocol connector. Supports `http` transport (JSON-RPC `initialize` + `tools/call`) and `stdio` transport (documented stub).

Admins configure connectors (credentials encrypted in settings store) and test connectivity from the admin panel. Pulled content is indexed into cognee product dataset `prod_{product_id}` in the background.

### System Prompts
All prompt bodies are externalized to `refs/prompts/*.md` (7 wiki sections, spec/documentation/guides doc, expert agent, deep research iterations, RAG/simple-chat system prompts). Edit directly — no code changes needed.

### Wiki Generation Pipeline
`wiki_generator.py` generates 7 sections sequentially, each building on previously generated content. **All prompt bodies live in `refs/prompts/*.md`** (externalized) and are loaded via `load_prompt_file()`; code only defines section order + variable mapping. Prompts are in Russian (designed for Qwen3.5-35b-a3b) with English technical terms. Substitution uses `str.replace` (not `.format`) so Mermaid/JSON literal braces stay unescaped.

### Recursive Language Models (RLM)
`rlm_runner.py` wraps **fast-rlm** (Deno + Pyodide). The RLM REPL is isolated from host Python — the codebase is passed as a long-context string; host FastAPI/cognee are NOT reachable inside the REPL. RLM is used **only for long-context tasks** (doc gen over large codebases, Deep Research); simple chat/Ask use the standard adalflow OllamaClient. Falls back to standard LLM if RLM is unavailable or the context is small (<20k chars).

### Knowledge Graph (cognee)
`cognee_manager.py` integrates **cognee** (v1.2.2) with Postgres + pgvector for the knowledge graph. cognee is pointed at **local Ollama** for both LLM (`cognify` entity extraction) and embeddings, so NO cloud key is required. cognee's config validator requires the full `{LLM_MODEL, LLM_ENDPOINT, LLM_API_KEY}` and `{EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, HUGGINGFACE_TOKENIZER}` groups to be set together — `cognee_manager.py` sets them all via `setdefault`. `init_cognee()` is non-fatal (falls back to SQLite/LanceDB if Postgres is down).

### Memory/Conversation in RAG
Custom `Memory` and `CustomConversation` classes in `rag.py` replace adalflow's built-in conversation management to work around list index errors. Dialog history is rebuilt from request messages on each call.

## Environment Variables

**No cloud API keys required.** Everything runs on local Ollama. See `.env.example` (tracked) for the full, documented list. Key groups:
- **Ollama**: `OLLAMA_HOST` (default `http://localhost:11434`).
- **Local OpenAI-compatible API**: `LOCAL_OPENAI_BASE_URL` / `LOCAL_OPENAI_API_KEY` (falls back to `not-needed`).
- **Embedder**: `DEEPWIKI_EMBEDDER_TYPE` (`ollama` default).
- **Postgres (products/artifacts + cognee)**: `DB_PROVIDER`, `DB_HOST` (default `localhost`; `postgres` in Docker), `DB_PORT`, `DB_NAME` (`cognee_db`), `DB_USERNAME`, `DB_PASSWORD`; `VECTOR_DB_PROVIDER=pgvector`.
- **cognee LLM (local Ollama)**: `LLM_PROVIDER=ollama`, `LLM_ENDPOINT` (`/v1`), `LLM_MODEL`, `LLM_API_KEY=not-needed`.
- **cognee embeddings**: `EMBEDDING_PROVIDER=ollama`, `EMBEDDING_MODEL=nomic-embed-text`, `EMBEDDING_ENDPOINT` (`/api/embed`), `EMBEDDING_DIMENSIONS=768`, `HUGGINGFACE_TOKENIZER=nomic-ai/nomic-embed-text-v1.5` (all required together by cognee's validator).
- **fast-rlm**: `RLM_MODEL_BASE_URL` (default `http://localhost:11434/v1`), `RLM_MODEL_API_KEY=not-needed`, `RLM_MODEL_NAME` (default `qwen3:8b`).
- **Auth**: `AUTH_PROVIDER` (`local` | `keycloak` | `both` | `none`), `BOOTSTRAP_ADMIN_USERNAME`, `BOOTSTRAP_ADMIN_PASSWORD`, `SETTINGS_SECRET_KEY` (Fernet key for settings encryption + JWT signing).
- **Keycloak OIDC** (when `AUTH_PROVIDER=keycloak|both`): `KEYCLOAK_URL` (default `http://localhost:8080`), `KEYCLOAK_CLIENT_ID` (default `productarium-frontend`), `KEYCLOAK_CLIENT_SECRET` (empty for public client), `KEYCLOAK_REALM` (default `productarium`).
- **Application**: `PORT` (default 8001), `SERVER_BASE_URL`, `DEEPWIKI_CONFIG_DIR`.
- **Enterprise Git (optional)**: `GITHUB_ENTERPRISE_URL`, `GITLAB_SELF_HOSTED_URL`.
- **Logging**: `LOG_LEVEL`, `LOG_FILE_PATH`.

## Additional Documentation
- `PROMPT.md` — Detailed technical specification (in Russian) covering architecture, modules, and API endpoints.
- `refs/` — `LLD.md`, `DataModel.md`, `current.json` reference docs, plus `refs/prompts/*.md` (all externalized prompt bodies: 7 wiki sections, spec/documentation/guides doc, expert agent, deep research iterations, RAG/simple-chat system prompts).
- `api/README.md` — Backend-specific documentation.
