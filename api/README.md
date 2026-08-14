# Productarium API

Backend API for Productarium — a product-centric documentation platform powered by fully local LLMs (Ollama or any OpenAI-compatible server). No cloud API keys required.

## Features

- **Product-Centric Model**: Products own typed Codebase, Spec, and Links entities plus a Knowledge Node tree (no polymorphic artifact entity).
- **Local LLM**: single OpenAI-compatible path (Ollama, LM Studio, llama.cpp, vLLM) — no cloud keys.
- **Knowledge Graph**: cognee + pgvector index every entity and knowledge node for RAG.
- **Expert Agent**: streaming chat (SSE) + document generation over indexed knowledge.
- **fast-rlm**: Recursive Language Models for long-context doc generation (≥20k chars) and Deep Research.
- **Authentication**: local (passlib bcrypt + JWT) and/or Keycloak OIDC.
- **Admin Panel**: models, git-credentials, Confluence, integrations, rlm, ssl, cognee, timeouts, users, API tokens, prompts.
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
| `LOCAL_OPENAI_BASE_URL` | Local OpenAI-compatible API URL | `http://localhost:1234/v1` |
| `LOCAL_OPENAI_API_KEY` | API key for local OpenAI API | `not-needed` |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USERNAME` / `DB_PASSWORD` | Postgres connection | `localhost` / `5432` / `cognee_db` / `cognee` / `cognee` |
| `LLM_PROVIDER` / `LLM_ENDPOINT` / `LLM_MODEL` / `LLM_API_KEY` | cognee LLM (local Ollama) | `ollama` / `…/v1` / `qwen3:8b` / `not-needed` |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | cognee embeddings (local Ollama) | `ollama` / `nomic-embed-text` / `768` |
| `RLM_MODEL_BASE_URL` / `RLM_MODEL_NAME` | fast-rlm | `…/v1` / `qwen/qwen3.6-27b` |
| `PORT` | API server port | `8001` |
| `AUTH_PROVIDER` | Auth mode: `local` / `keycloak` / `both` / `none` | `local` |
| `BOOTSTRAP_ADMIN_USERNAME` / `BOOTSTRAP_ADMIN_PASSWORD` | One-shot bootstrap admin | `admin` / `change-me` |
| `SETTINGS_SECRET_KEY` | Fernet key for settings encryption + JWT signing | (ephemeral dev key) |
| `LOG_FORMAT` | Log format: `logfmt` or `json` (console-only) | `logfmt` |
| `KEYCLOAK_URL` / `KEYCLOAK_CLIENT_ID` / `KEYCLOAK_CLIENT_SECRET` / `KEYCLOAK_REALM` | Keycloak OIDC | `http://localhost:8080` / `productarium-frontend` / (empty) / `productarium` |

### Configuration Files (`api/config/`)

JSON files with `${ENV_VAR}` placeholder support (resolved at load time by `api/config/__init__.py`):

1. **`generator.json`** — LLM models (single OpenAI-compatible path).
2. **`embedder.json`** — Embedding models, retriever (`top_k: 20`), text splitter (350 words, 100 overlap).
3. **`repo.json`** — File filters (excluded dirs/files) and repository size limits.

Custom config directory via `DEEPWIKI_CONFIG_DIR`.

## Architecture

### Entry Point

- **`main.py`** — Loads `.env`, configures logging, starts uvicorn on `PORT` (8001).
- **`api.py`** — Main FastAPI app. Connects all routers via `include_all_routers(app)`. `startup_event()` calls `init_db()` then `init_cognee()`. Both non-fatal.

### Routers (`api/routers/`)

Auto-discovered: add `api/routers/<name>.py` with a module-level `router = APIRouter(...)` — it connects automatically.

- **`admin.py`** — Admin-protected CRUD + connectivity tests for `models`, `git`, `confluence`, `integrations`, `rlm`, `ssl`, `cognee`, `timeouts`, `users`, `apitokens`, `prompts`. Secrets encrypted on save, masked on read.
- **`auth/`** (in `api/auth/`) — Local login/me/logout, first-run setup, password change/reset, Keycloak OIDC login/callback.
- **`docgen.py`** — Per-type generate endpoints: `POST .../codebases/{id}/generate` + status, `POST .../specs/{id}/generate` + status. Links do not generate.
- **`expert.py`** — Expert agent: SSE chat (`POST /api/products/{id}/ask`) + document generation (`POST /api/products/{id}/ask/doc`).
- **`integrations.py`** — List/test/pull from integration connectors; git connectors create `CodebaseORM`, non-git pulls create knowledge nodes only.
- **`knowledge.py`** — Knowledge tree CRUD, markitdown upload, verified toggle, AI product summary (`generate_product_summary(product, codebases, specs, nodes)`).
- **`products.py`** — Per-type create/delete/update: `POST/DELETE/PUT .../codebases|specs|links/{id}`.
- **`public.py`** — API-token-authenticated endpoints: export verified knowledge (markdown/json with `codebases`/`specs`/`links`/`nodes` keys), ask, push to Confluence/git.

### Authentication (`api/auth/`)

- **`local.py`** — Local login/password (passlib bcrypt + JWT session cookie).
- **`keycloak.py`** — Keycloak OIDC (authlib).
- **`deps.py`** — FastAPI dependencies (`get_current_user`, `require_admin`).
- **`tokens.py`** — API tokens (sha256 hash; plaintext shown once).
- **`bootstrap.py`** — One-shot bootstrap admin via `BOOTSTRAP_ADMIN_*`.

`AUTH_PROVIDER` selects mode: `local` | `keycloak` | `both` | `none`.

### Integrations (`api/integrations/`)

Scalable connector framework. Auto-discovered via `pkgutil`. Each connector implements `test()`, `list_spaces()`, `pull(source_id, opts)`:

- **`github.py`** / **`gitlab.py`** — List repos, clone + document as `CodebaseORM` (`repo_url`/`repo_type`).
- **`confluence.py`** — List spaces, pull pages (recursively, attachments via markitdown) as knowledge nodes.
- **`mcp.py`** — Model Context Protocol. Supports `http` transport (JSON-RPC `initialize` + `tools/call`) and `stdio` transport.
- **`base.py`** / **`registry.py`** — Base class `IntegrationConnector` + auto-discovery registry.
- **`_git_base.py`** — Shared git connector logic.

Add new `api/integrations/<name>.py` subclassing `IntegrationConnector` — no core changes needed.

### Core Modules

- **`rag.py`** — RAG implementation. Custom `Memory`/`CustomConversation`/`DialogTurn` (adalflow workaround). `RAG` class manages retriever, FAISS indices, queries. Parameters: text splitter (350 words, 100 overlap), retriever top_k=20.
- **`data_pipeline.py`** — Repository cloning (GitHub/GitLab, shallow `--depth=1`), file reading with include/exclude filters, `DatabaseManager` (FAISS indices).
- **`docgen/`** — Documentation generation package. **No dispatcher** — each generate endpoint calls its generator directly. `codebase.py:generate_codebase_docs` (RLM for long-context ≥20k chars else standard LLM, 7 sections from refs), `spec.py:generate_openapi_docs`/`generate_asyncapi_docs` (stdlib parse + skeleton + LLM enrichment). `jobs.py` (async 202+poll worker, takes `entity_type`). `_common.py` (shared `_index_in_background`). All paths index into cognee and persist `generated_docs` + `pages`.
- **`expert/`** — Expert agent package. `chat.py` (cognee-recall + RLM routing + LLM streaming), `generate.py` (standalone doc). Prompt bodies in `refs/prompts/expert_agent_*.md`.
- **`cognee/`** — cognee integration (`_runtime.py` configures local Ollama for LLM + embeddings; no cloud key). `init_cognee()`, `add_and_index_document()`, `query_cognee()`, `reindex_product_knowledge_graph()` — all async, all non-fatal.
- **`rlm/runner.py`** — fast-rlm wrapper (Deno + Pyodide) for long-context reasoning. Single path: admin config → `LOCAL_OPENAI_BASE_URL` → default.
- **`config/`** — Central configuration package. `__init__.py` (JSON loader, `${ENV_VAR}` placeholders), `settings.py` (encrypted key/value store, Fernet via `SETTINGS_SECRET_KEY`), `timeout.py` (per-key timeout overrides), `ssl.py` (TLS patch for corporate gateways).
- **`clients/`** — `openai_client.py` (custom OpenAI-compatible client for local LLM servers). Single client (no `OllamaClient`).
- **`utils/`** — `logging.py` (console-only, `LOG_FORMAT` env: `logfmt`/`json`), `llm_helpers.py` (`cap(text, limit)` char-based), `llm_tokens.py` (`get_model_context_window`, `_count_tokens`).
- **`models.py`** — SQLAlchemy 2.0 ORM: `UserORM`, `ProductORM`, `CodebaseORM`, `SpecORM`, `LinksORM`, `KnowledgeNodeORM`, `SettingORM`, `ApiTokenORM`.
- **`db.py`** — SQLAlchemy engine + `SessionLocal` + `get_db()` + `init_db()` (`Base.metadata.create_all`, idempotent, non-fatal).
- **`prompts.py`** — Prompt registry + loader. `load_prompt_file()` applies `_wrap_prompt(content, language)` after loading. Bodies in `refs/prompts/*.md` (externalized).

### System Prompts

All prompt bodies are externalized to `refs/prompts/*.md` (7 wiki sections, spec doc, expert agent, deep research iterations, RAG/simple-chat system prompts). Edit directly — no code changes needed. `load_prompt_file()` applies `_wrap_prompt(content, language)` after loading.

## API Endpoints

### Products, Codebases, Specs, Links

| Endpoint | Method | Description |
|---|---|---|
| `/api/products` | GET | List all products |
| `/api/products` | POST | Create a product |
| `/api/products/{id}` | GET | Get product detail |
| `/api/products/{id}` | PUT | Update product |
| `/api/products/{id}` | DELETE | Delete product |
| `/api/products/{id}/codebases` | POST | Add codebase |
| `/api/products/{id}/specs` | POST | Add spec |
| `/api/products/{id}/links` | POST | Add links |
| `/api/products/{id}/codebases/{id}` | DELETE/PUT | Delete/update codebase |
| `/api/products/{id}/specs/{id}` | DELETE/PUT | Delete/update spec |
| `/api/products/{id}/links/{id}` | DELETE/PUT | Delete/update links |
| `/api/products/{id}/codebases/{id}/generate` | POST | Generate codebase docs |
| `/api/products/{id}/specs/{id}/generate` | POST | Generate spec docs |

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
| `/api/integrations/{name}/pull` | POST | Pull content as codebase/knowledge node |

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
| `/api/public/products/{id}/knowledge` | GET | Export verified knowledge (markdown/json) |
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
- Products/Codebases/Specs/Links/Knowledge Nodes: Postgres (`products`, `codebases`, `specs`, `links`, `knowledge_nodes` tables)
- cognee knowledge graph: Postgres + pgvector

## Data Flow (Product → Codebase/Spec → Docs)

1. Create a **Product** and add a **Codebase** (via repo URL), **Spec** (yaml/json content), or **Links** (JSON array).
2. **Generate** docs (per-type endpoint):
   - **codebase**: clone repo (shallow), read files, build long-context → **fast-rlm** (if ≥20k chars) or standard LLM generates 7 wiki sections → `generated_docs` + `pages` persisted → indexed in cognee (background, async 202+poll).
   - **spec**: parse (stdlib json/yaml) → markdown skeleton + LLM enrichment → indexed in cognee.
   - **links**: no generation (content storage only).
3. Frontend viewer renders `codebase.pages` (nav tree) + markdown/Mermaid; Ask panel uses RAG (FAISS, top_k=20) augmented with cognee recall over the codebase's dataset.
4. **Expert Agent** streams SSE chat over all indexed knowledge (codebases + specs + links + knowledge nodes); `ask/doc` generates a standalone Markdown document.
