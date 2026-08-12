# Productarium API

Backend API for Productarium — a product-centric documentation platform powered by fully local LLMs (Ollama or any OpenAI-compatible server). No cloud API keys required.

## Features

- **Product-Centric Model**: Products own Artifacts (codebase, spec, links, documentation, guides) and a Knowledge Node tree.
- **Local LLM**: Ollama (default) or local OpenAI-compatible API — no cloud keys.
- **Knowledge Graph**: cognee + pgvector index every artifact and knowledge node for RAG.
- **Expert Agent**: streaming chat (SSE) + document generation over indexed knowledge.
- **fast-rlm**: Recursive Language Models for long-context doc generation (≥20k chars) and Deep Research.
- **Authentication**: local (passlib bcrypt + JWT) and/or Keycloak OIDC.
- **Admin Panel**: models, git-credentials, Confluence, integrations, users, API tokens.
- **Integrations**: GitHub, GitLab, Confluence, MCP (stdio/http) — auto-discovered.

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** running locally:
  ```bash
  ollama pull qwen3:8b           # or qwen3.5:9b, gemma3:12b, etc.
  ollama pull nomic-embed-text   # required for embeddings
  ```
- **PostgreSQL + pgvector** (optional; `docker-compose up postgres`):
  `pgvector/pgvector:pg18-trixie` (user/db: `cognee`/`cognee_db`).
  If Postgres is unreachable, the app falls back gracefully.

### Install & Run

```bash
python -m pip install poetry==2.0.1 && poetry install -C api
python -m api.main              # uvicorn on port 8001 (hot-reload in dev)
```

### Environment Variables

All configuration is local. See `.env.example` in the project root for the full, documented list. Key variables:

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_HOST` | Local Ollama URL | `http://localhost:11434` |
| `LOCAL_OPENAI_BASE_URL` | Local OpenAI-compatible API URL (optional) | `http://localhost:8080/v1` |
| `LOCAL_OPENAI_API_KEY` | API key for local OpenAI API | `not-needed` |
| `DEEPWIKI_EMBEDDER_TYPE` | Embedder: `ollama` or `openai_local` | `ollama` |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USERNAME` / `DB_PASSWORD` | Postgres connection | `localhost` / `5432` / `cognee_db` / `cognee` / `cognee` |
| `LLM_PROVIDER` / `LLM_ENDPOINT` / `LLM_MODEL` / `LLM_API_KEY` | cognee LLM (local Ollama) | `ollama` / `…/v1` / `qwen3:8b` / `not-needed` |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | cognee embeddings (local Ollama) | `ollama` / `nomic-embed-text` / `768` |
| `RLM_MODEL_BASE_URL` / `RLM_MODEL_NAME` | fast-rlm | `…/v1` / `qwen3:8b` |
| `PORT` | API server port | `8001` |
| `AUTH_PROVIDER` | Auth mode: `local` / `keycloak` / `both` / `none` | `local` |
| `BOOTSTRAP_ADMIN_USERNAME` / `BOOTSTRAP_ADMIN_PASSWORD` | One-shot bootstrap admin | `admin` / `change-me` |
| `SETTINGS_SECRET_KEY` | Fernet key for settings encryption + JWT signing | (ephemeral dev key) |
| `KEYCLOAK_URL` / `KEYCLOAK_CLIENT_ID` / `KEYCLOAK_CLIENT_SECRET` / `KEYCLOAK_REALM` | Keycloak OIDC | `http://localhost:8080` / `productarium-frontend` / (empty) / `productarium` |

### Configuration Files (`api/config/`)

JSON files with `${ENV_VAR}` placeholder support (resolved at load time by `config.py`):

1. **`generator.json`** — LLM providers and models (`ollama`, `openai_local`).
2. **`embedder.json`** — Embedding models, retriever (`top_k: 20`), text splitter (350 words, 100 overlap).
3. **`repo.json`** — File filters (excluded dirs/files) and repository size limits.

Custom config directory via `DEEPWIKI_CONFIG_DIR`.

## Architecture

### Entry Point

- **`main.py`** — Loads `.env`, configures logging, starts uvicorn on `PORT` (8001).
- **`api.py`** — Main FastAPI app. Product/Artifact CRUD, generate, RLM run, legacy wiki cache, model config. `startup_event()` calls `init_db()` then `init_cognee()`. Connects all routers via `include_all_routers(app)`.

### Routers (`api/routers/`)

Auto-discovered: add `api/routers/<name>.py` with a module-level `router = APIRouter(...)` — it connects automatically.

- **`admin.py`** — Admin-protected CRUD + connectivity tests for `models`, `git`, `confluence`, `integrations`, `users`, `apitokens`. Secrets encrypted on save, masked on read.
- **`auth/`** (in `api/auth/`) — Local login/me/logout, first-run setup, password change/reset, Keycloak OIDC login/callback.
- **`expert.py`** — Expert agent: SSE chat (`POST /api/products/{id}/ask`) + document generation (`POST /api/products/{id}/ask/doc`).
- **`integrations.py`** — List/test/pull from integration connectors; create artifacts or knowledge nodes from pulled content.
- **`knowledge.py`** — Knowledge tree CRUD, markitdown upload, verified toggle, AI product summary.
- **`public.py`** — API-token-authenticated endpoints: export verified knowledge, ask, push to Confluence/git.

### Authentication (`api/auth/`)

- **`local.py`** — Local login/password (passlib bcrypt + JWT session cookie).
- **`keycloak.py`** — Keycloak OIDC (authlib).
- **`deps.py`** — FastAPI dependencies (`get_current_user`, `require_admin`).
- **`tokens.py`** — API tokens (sha256 hash; plaintext shown once).
- **`bootstrap.py`** — One-shot bootstrap admin via `BOOTSTRAP_ADMIN_*`.

`AUTH_PROVIDER` selects mode: `local` | `keycloak` | `both` | `none`.

### Integrations (`api/integrations/`)

Scalable connector framework. Auto-discovered via `pkgutil`. Each connector implements `test()`, `list_spaces()`, `pull(source_id, opts)`:

- **`github.py`** / **`gitlab.py`** — List repos, clone + document as `codebase` artifacts.
- **`confluence.py`** — List spaces, pull pages (recursively, attachments via markitdown) as `documentation` artifacts or knowledge nodes.
- **`mcp.py`** — Model Context Protocol. Supports `http` transport (JSON-RPC `initialize` + `tools/call`) and `stdio` transport.
- **`base.py`** / **`registry.py`** — Base class `IntegrationConnector` + auto-discovery registry.
- **`_git_base.py`** — Shared git connector logic.

Add new `api/integrations/<name>.py` subclassing `IntegrationConnector` — no core changes needed.

### Core Modules

- **`rag.py`** — RAG implementation. Custom `Memory`/`CustomConversation`/`DialogTurn` (adalflow workaround). `RAG` class manages retriever, FAISS indices, queries. Parameters: text splitter (350 words, 100 overlap), retriever top_k=20.
- **`data_pipeline.py`** — Repository cloning (GitHub/GitLab, shallow `--depth=1`), file reading with include/exclude filters, `DatabaseManager` (FAISS indices).
- **`docgen/`** — Artifact documentation package. `dispatcher.py` dispatches by type: codebase→RLM (fast-rlm, if ≥20k chars) or standard LLM; spec→parse + enrich; documentation/guides→LLM. Split across `codebase.py` / `spec.py` / `simple.py` / `_common.py` (shared helpers). All paths index into cognee and persist `generated_docs` + `pages`.
- **`expert_agent.py`** — Expert agent (cognee-recall + RLM routing + LLM streaming). Prompt bodies in `refs/prompts/expert_agent_*.md`.
- **`wiki_generator.py`** — Sequential 7-section wiki generation (Overview → Architecture → Functional → Technical → CI/CD → LLD → Data Model). Section bodies from `refs/prompts/<section>.md`. Substitution via `str.replace` (not `.format`).
- **`cognee_manager.py`** — cognee integration (local Ollama for LLM + embeddings; no cloud key). `init_cognee()`, `add_and_index_document()`, `query_cognee()` — all async, all non-fatal.
- **`rlm_runner.py`** — fast-rlm wrapper (Deno + Pyodide) for long-context reasoning.
- **`settings_store.py`** — Encrypted key/value settings store (Fernet via `SETTINGS_SECRET_KEY`).
- **`models.py`** — SQLAlchemy 2.0 ORM: `UserORM`, `ProductORM`, `ArtifactORM`, `KnowledgeNodeORM`, `SettingORM`, `ApiTokenORM`.
- **`db.py`** — SQLAlchemy engine + `SessionLocal` + `get_db()` + `init_db()`.
- **`prompts.py`** — Prompt registry + loader. Bodies in `refs/prompts/*.md` (externalized).
- **`config.py`** — Central configuration. JSON from `api/config/`, `${ENV_VAR}` placeholders, provider/embedder management.

### System Prompts

All prompt bodies are externalized to `refs/prompts/*.md` (7 wiki sections, spec/documentation/guides doc, expert agent, deep research iterations, RAG/simple-chat system prompts). Edit directly — no code changes needed.

## API Endpoints

### Products & Artifacts

| Endpoint | Method | Description |
|---|---|---|
| `/api/products` | GET | List all products |
| `/api/products` | POST | Create a product |
| `/api/products/{id}` | GET | Get product detail |
| `/api/products/{id}` | PUT | Update product |
| `/api/products/{id}` | DELETE | Delete product |
| `/api/products/{id}/artifacts` | POST | Add artifact to product |
| `/api/products/{id}/artifacts/{id}` | DELETE | Delete artifact |
| `/api/products/{id}/artifacts/{id}/generate` | POST | Generate docs for artifact |

### Expert Agent

| Endpoint | Method | Description |
|---|---|---|
| `/api/products/{id}/ask` | POST | Expert agent SSE chat |
| `/api/products/{id}/ask/doc` | POST | Generate standalone Markdown document |

### Knowledge Tree

| Endpoint | Method | Description |
|---|---|---|
| `/api/products/{id}/knowledge` | GET | List knowledge nodes |
| `/api/products/{id}/knowledge` | POST | Create knowledge node |
| `/api/products/{id}/knowledge/{id}` | PUT | Update knowledge node |
| `/api/products/{id}/knowledge/{id}` | DELETE | Delete knowledge node |
| `/api/products/{id}/knowledge/{id}/verified` | PATCH | Toggle verified |

### Integrations

| Endpoint | Method | Description |
|---|---|---|
| `/api/integrations` | GET | List available connectors |
| `/api/integrations/{name}/test` | POST | Test connector connectivity |
| `/api/integrations/{name}/spaces` | GET | List spaces/repos |
| `/api/integrations/{name}/pull` | POST | Pull content as artifact/knowledge node |

### Admin

| Endpoint | Method | Description |
|---|---|---|
| `/api/admin/models` | GET/POST | Model configuration |
| `/api/admin/git` | GET/POST | Git credentials |
| `/api/admin/confluence` | GET/POST | Confluence settings |
| `/api/admin/integrations` | GET/POST | Integration settings |
| `/api/admin/users` | GET/POST | User management |
| `/api/admin/apitokens` | GET/POST | API token management |

### Auth

| Endpoint | Method | Description |
|---|---|---|
| `/api/auth/login` | POST | Local login |
| `/api/auth/me` | GET | Current user |
| `/api/auth/logout` | POST | Logout |
| `/api/auth/keycloak/login` | GET | Keycloak OIDC redirect |
| `/api/auth/keycloak/callback` | GET | Keycloak OIDC callback |

### Public (API-token authenticated)

| Endpoint | Method | Description |
|---|---|---|
| `/api/public/products/{id}/export` | GET | Export verified knowledge |
| `/api/public/products/{id}/ask` | POST | Ask expert agent |
| `/api/public/products/{id}/push` | POST | Push to Confluence/git |

### RLM

| Endpoint | Method | Description |
|---|---|---|
| `/api/rlm/run` | POST | Run fast-rlm task |

## Storage

All data is stored locally:
- Cloned repositories: `~/.adalflow/repos/`
- FAISS indices: `~/.adalflow/databases/`
- Wiki cache: `~/.adalflow/wikicache/`
- Products/Artifacts/Knowledge Nodes: Postgres (`products`, `artifacts`, `knowledge_nodes` tables)
- cognee knowledge graph: Postgres + pgvector

## Data Flow (Product → Artifact → Docs)

1. Create a **Product** and add **Artifacts** (codebase via repo URL, spec content, documentation, links, guides).
2. **Generate** docs:
   - **codebase**: clone repo (shallow), read files, build long-context → **fast-rlm** (if ≥20k chars) or standard LLM generates 7 wiki sections → `generated_docs` + `pages` persisted → indexed in cognee (background).
   - **spec**: parse (stdlib json/yaml) → markdown skeleton + LLM enrichment → indexed in cognee.
   - **documentation/guides**: LLM enrichment → indexed in cognee.
3. Frontend viewer renders `artifact.pages` (nav tree) + markdown/Mermaid; Ask panel uses RAG (FAISS, top_k=20) augmented with cognee recall.
4. **Expert Agent** streams SSE chat over all indexed knowledge; `ask/doc` generates a standalone Markdown document.
