# Productarium API

Backend API for Productarium — a product-centric documentation platform powered by fully local LLMs (any OpenAI-compatible server). No cloud API keys required.

## Features

- **Product-Centric Model**: Products own typed Codebase, Spec, and Links entities plus a Knowledge Node tree (no polymorphic artifact entity).
- **Local LLM**: single OpenAI-compatible path (LM Studio, llama.cpp, vLLM) — no cloud keys.
- **Semantic Memory**: pgvector indexes generated docs per product for RAG recall (no external KG service).
- **Expert Agent**: streaming chat (SSE) + document generation over indexed knowledge.
- **Deep Research**: bounded multi-iteration research loop over product knowledge.
- **Authentication**: local (passlib bcrypt + JWT) and/or Keycloak OIDC.
- **Admin Panel**: models, git-credentials, Confluence, integrations, ssl, embedder, memory, timeouts, users, API tokens, prompts.
- **Integrations**: GitHub, GitLab, Confluence — auto-discovered.
- **MCP Platform**: outbound tool integrations via langchain-mcp-adapters (tools auto-appended to the expert agent) + Productarium itself exposed as an MCP server (FastMCP at `/api/mcp`).

## Quick Start

### Prerequisites

- **Python 3.11+**
- **An OpenAI-compatible server** running locally (LM Studio, llama.cpp, vLLM) with a generation model and an embedding model:
  ```bash
  # e.g. pull a model in LM Studio / llama.cpp / vLLM (see .env.example)
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
| `LOCAL_OPENAI_BASE_URL` | Local OpenAI-compatible API URL | `http://localhost:1234/v1` |
| `LOCAL_OPENAI_API_KEY` | API key for local OpenAI API | `not-needed` |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USERNAME` / `DB_PASSWORD` | Postgres connection | `localhost` / `5432` / `cognee_db` / `cognee` / `cognee` |
| `RLM_MODEL_CONTEXT_WINDOW` | Context-window override (tokens) for prompt budgeting | (from live API metadata / model name) |
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
- **`api.py`** — Main FastAPI app. Connects all routers via `include_all_routers(app)`, mounts the inbound MCP server at `/api/mcp` (best-effort — degrades with a warning if `mcp` is missing). Lifespan calls `init_db()` then memory init; both non-fatal.

### Routers (`api/routers/`)

Auto-discovered: add `api/routers/<name>.py` with a module-level `router = APIRouter(...)` — it connects automatically.

- **`admin.py`** — Admin-protected CRUD + connectivity tests for `models`, `git`, `confluence`, `integrations`, `ssl`, `embedder`, `memory`, `timeouts`, `users`, `apitokens`, `prompts`. Secrets encrypted on save, masked on read.
- **`auth/`** (in `api/auth/`) — Local login/me/logout, first-run setup, password change/reset, Keycloak OIDC login/callback.
- **`docgen.py`** — Per-type generate endpoints: `POST .../codebases/{id}/generate` + status, `POST .../specs/{id}/generate` + status. Links do not generate.
- **`expert.py`** — Expert agent: SSE chat (`POST /api/products/{id}/ask`) + document generation (`POST /api/products/{id}/ask/doc`).
- **`integrations.py`** — List/test/pull from integration connectors; git connectors create `CodebaseORM`, non-git pulls create knowledge nodes only.
- **`knowledge.py`** — Knowledge tree CRUD, markitdown upload, verified toggle, AI product summary (`generate_product_summary(product, codebases, specs, nodes)`).
- **`mcp_admin.py`** — Admin MCP server registry: CRUD + `POST .../test` (bounded health-check; persists status) + `GET .../tools` (cached discovery) under `/api/admin/mcp/servers`.
- **`product_mcp.py`** — Per-product MCP bindings CRUD under `/api/products/{product_id}/mcp` (enable a server, optional `allowed_tools` allowlist).
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
- **`base.py`** / **`registry.py`** — Base class `IntegrationConnector` + auto-discovery registry.
- **`_git_base.py`** — Shared git connector logic.

### MCP Platform (`api/mcp/`)

Model Context Protocol in two directions:
- **`manager.py`** — Outbound tool manager over langchain-mcp-adapters `MultiServerMCPClient`. Per-server clients + `tools/list` discovery results cached by config fingerprint (transport/url/command/args/headers/env ciphertext/updated_at); every connect is bounded by `MCP_DISCOVERY_TIMEOUT_SECONDS` (default 10 s). Applies the per-binding `allowed_tools` allowlist and dedupes tool names. `get_tools_for_product()` is best-effort: a failing server is skipped, never fatal to the expert agent.
- **`secrets.py`** — Fernet encrypt/decrypt + masking for server `headers`/`env` (stored encrypted in `mcp_servers`, masked on every read).
- **`inbound.py`** — Inbound MCP server (`mcp.server.fastmcp.FastMCP`, streamable HTTP) mounted at `/api/mcp`. Auth: Bearer API tokens (sha256 lookup in `api_tokens`, updates `last_used_at`). Tools: `list_products`, `get_product_knowledge`, `search_knowledge`, `ask_expert` (bounded by `MCP_ASK_TIMEOUT_SECONDS`, default 120 s).

Expert agent integration: `build_expert_agent` receives the bound enabled servers' tools (after allowlist) as `extra_tools` — no separate endpoint.

### Core Modules

- **`memory/`** — Semantic memory: `pgvector_backend.py` (chunks + embeddings in `knowledge_chunks`, HNSW cosine recall), `resolver.py` (`memory.backend` admin setting; pgvector only), backend-agnostic facade (`index_document` / `query_memory` / `reindex_product_memory`).
- **`repositories/`** — `product_repo.py` (Product/Codebase/Spec/Links/Database ORM↔Pydantic + persistence), `documents.py` (repo clone orchestration via `api.clients.git` + symlink-safe file reading).
- **`docgen/`** — Documentation generation package. **No dispatcher** — each generate endpoint calls its generator directly. `codebase.py:generate_codebase_docs` (deepagents orchestrator with parallel section subagents + standard-LLM fallback, 7 sections from refs), `spec.py:generate_openapi_docs`/`generate_asyncapi_docs` (stdlib parse + skeleton + agent/LLM enrichment), `database.py` (MCP introspection reverse-engineering). `jobs.py` (async 202+poll worker, takes `entity_type`). `_common.py` (shared `_index_in_background`). All paths index into the pgvector memory backend and persist `generated_docs` + `pages`.
- **`expert/`** — Expert agent package. `chat.py` (SSE chat + sessions), `deep_research.py` (bounded multi-iteration loop), `generate.py` (standalone doc). Prompt bodies in `refs/prompts/expert_agent_*.md`.
- **`config/`** — Central configuration package. `__init__.py` (JSON loader, `${ENV_VAR}` placeholders), `settings.py` (encrypted key/value store, Fernet via `SETTINGS_SECRET_KEY`), `timeout.py` (per-key timeout overrides), `ssl.py` (TLS patch for corporate gateways).
- **`clients/`** — `git.py` (GitHub/GitLab shallow clone + remote file content APIs).
- **`utils/`** — `logging.py` (console-only, `LOG_FORMAT` env: `logfmt`/`json`), `llm_helpers.py` (`cap(text, limit)` char-based), `llm_tokens.py` (`get_model_context_window`, `_count_tokens`).
- **`models.py`** — SQLAlchemy 2.0 ORM: `UserORM`, `ProductORM`, `CodebaseORM`, `SpecORM`, `LinksORM`, `KnowledgeNodeORM`, `SettingORM`, `ApiTokenORM`, `McpServerORM`, `ProductMcpServerORM`.
- **`db.py`** — SQLAlchemy engine + `SessionLocal` + `get_db()` + `init_db()` (`Base.metadata.create_all`, idempotent, non-fatal).
- **`prompts.py`** — Prompt registry + loader. `load_prompt_file()` applies `_wrap_prompt(content, language)` after loading. Bodies in `refs/prompts/*.md` (externalized).

### System Prompts

All prompt bodies are externalized to `refs/prompts/*.md` (7 wiki section contracts in `docgen_sections.md`, subpage contracts in `docgen_subpages.md`, decomposer plan in `docgen_decomposer.md`, spec doc, expert agent, deep research iterations). Edit directly — no code changes needed. `load_prompt_file()` applies `_wrap_prompt(content, language)` after loading.

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

### MCP

| Endpoint | Method | Description |
|---|---|---|
| `/api/admin/mcp/servers` | GET/POST | List/create MCP servers (secrets masked) |
| `/api/admin/mcp/servers/{id}` | PUT/DELETE | Update/delete a server (cascades bindings) |
| `/api/admin/mcp/servers/{id}/test` | POST | Bounded health-check + tool discovery; persists status |
| `/api/admin/mcp/servers/{id}/tools` | GET | Cached tool list (no reconnect) |
| `/api/products/{id}/mcp` | GET/POST | List/create product bindings |
| `/api/products/{id}/mcp/{binding_id}` | PUT/DELETE | Update/delete a binding |
| `/api/mcp` | POST (JSON-RPC) | Inbound MCP server (streamable HTTP; Bearer API token) |

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

## Storage

All data is stored locally:
- Cloned repositories: managed state dir (`PRODUCTARIUM_STATE_DIR`, `~/.adalflow` legacy compat)
- Products/Codebases/Specs/Links/Databases/Knowledge Nodes: Postgres (`products`, `codebases`, `specs`, `links`, `databases`, `knowledge_nodes` tables)
- Semantic memory chunks: Postgres + pgvector (`knowledge_chunks` + HNSW index)

## Data Flow (Product → Codebase/Spec → Docs)

1. Create a **Product** and add a **Codebase** (via repo URL), **Spec** (yaml/json content), or **Links** (JSON array).
2. **Generate** docs (per-type endpoint):
   - **codebase**: clone repo (shallow) → **deepagents** pipeline (standard-LLM fallback) generates 7 wiki sections with per-capability/per-API/per-store subpages → `generated_docs` + `pages` persisted → indexed into the pgvector memory backend (background, async 202+poll).
   - **spec**: parse (stdlib json/yaml) → markdown skeleton + agent/LLM enrichment → indexed into the pgvector memory backend.
   - **links**: no generation (content storage only).
3. Frontend viewer renders `codebase.pages` (nav tree) + markdown/Mermaid; Ask panel uses semantic recall (pgvector, top_k=20) over the product's indexed chunks.
4. **Expert Agent** streams SSE chat over all indexed knowledge (codebases + specs + links + knowledge nodes); `ask/doc` generates a standalone Markdown document.
