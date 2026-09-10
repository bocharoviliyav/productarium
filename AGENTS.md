# AGENTS.md

This file provides guidance to AI agents (Warp/Claude Code/Codex) working with code in this repository.

## Build & Development Commands

### Prerequisites
- **Python 3.11+** and **Node.js** with **bun** (the frontend uses bun, not yarn)
- **A local OpenAI-compatible server** (LM Studio, llama.cpp, vLLM) must be running with at least one generation model and an embedding model (e.g. `qwen/qwen3.6-27b` + `text-embedding-nomic-embed-text-v1.5`).
- **PostgreSQL + pgvector** for products/entities, chat sessions, LangGraph checkpoints, and semantic memory.
  `docker-compose up postgres` starts `pgvector/pgvector:pg18-trixie` (user/db: `cognee`/`cognee_db`).
  If Postgres is unreachable, the app logs a warning and falls back to SQLite (degraded: no cosine recall, but it starts).

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
Single unified hermetic test suite in `tests/` (SQLite in-memory, mocked LLMs — no Postgres or model server needed):
```bash
poetry -C api run sh -c 'cd "$(git rev-parse --show-toplevel)" && python -m pytest -q'  # all tests
pytest tests/unit/                       # unit tests only
pytest tests/integration/                # integration tests only
pytest tests/unit/test_extract_repo_name.py  # single test file
python tests/run_tests.py                # via the test runner script
```
Pytest config is in `pytest.ini` (`testpaths=test`, strict markers, short tracebacks).
`tests/conftest.py:build_test_client()` is the standard app factory; it overrides `auth_deps.get_current_user` with a fixed admin unless called with `default_admin_auth=False` (needed for tests that assert real 401/403 semantics).

### Docker
```bash
docker-compose up       # builds + runs (postgres, API port 8001, frontend port 3000)
```
Postgres data in the `postgres_data` volume; managed state dir (repo clones, SQLite checkpointer fallback) is mounted.

## Architecture Overview

Productarium is a **product-centric** documentation platform. The top-level entity is a **Product** (microservice / monolith / databus service), which owns typed **Codebase**, **Spec**, **Links**, and **Database** entities plus a **Knowledge Node** tree. The agent stack is **LangChain / LangGraph**: **deepagents** drives codebase doc generation (with a verification pipeline: citations, LLM judge, Mermaid verify/repair), a **Deep Research** loop answers multi-iteration research questions, and databases are **reverse-engineered through MCP introspection tools**. Semantic memory is **pgvector-direct** (cosine recall over `knowledge_chunks`, HNSW index — no external KG service). The **MCP platform** is bidirectional: external MCP servers' tools are bound to products and appended to the expert agent; Productarium itself is exposed as an MCP server at `/api/mcp`. Fully local — no cloud API keys required.

### Product-Centric Data Model
No polymorphic artifact entity — separate typed ORM models (all in `api/models.py`):
- **Product** (`products`): `id, name, description, summary, owner_id, created_at, updated_at`. Owns `codebases`, `specs`, `links`, `databases` (each `cascade="all, delete-orphan"`).
- **Codebase** (`codebases`): `id, product_id (FK CASCADE), name, repo_url, repo_type, token, generated_docs (Text), pages (JSON tree), verified, verified_by, verified_at, source, timestamps`.
- **Spec** (`specs`): `id, product_id, name, kind (openapi|asyncapi), content (Text — yaml/json), verified/…, source, timestamps`.
- **Links** (`links`): `id, product_id, name, content (Text — JSON array of {url, description}), verified/…, source, timestamps`.
- **Database** (`databases`): `id, product_id, name, dsn_masked (raw DSN masked on acceptance — never persisted), mcp_server_id (optional pin), generated_docs, pages, verified/…, source, timestamps`.
- **KnowledgeNode** (`knowledge_nodes`): `id, product_id, parent_id, title, slug, node_type (page|folder|branch), content_md, source, verified, verified_by, verified_at, created_by, timestamps`.
- **User** (`productarium_users`), **Setting** (`settings` — Fernet-encrypted values), **ApiToken** (`api_tokens` — sha256 hashes).
- **ChatSession / ChatMessage** — per-user expert chat sessions + transcripts.
- **McpServer** (`mcp_servers`) — admin-managed outbound MCP registry (headers/env Fernet-encrypted at rest, masked on read).
- **ProductMcpServer** (`product_mcp_servers`) — per-product binding with optional `allowed_tools` allowlist.
- **KnowledgeChunk** (`knowledge_chunks`) — pgvector memory chunks (`product_id, source_type, source_id, content, embedding` + HNSW).
- Persisted via SQLAlchemy 2.0 (`api/db.py`: `init_db()` = `create_all` + pgvector extension/index; idempotent, non-fatal; no migrations). `get_db()` is the FastAPI dependency.
- REST: `GET/POST /api/products` (GET serves **light rows in a bare JSON array** — SQL-counted child totals only, `limit`/`offset` query params, filtered total in the `X-Total-Count` header, per-user visibility filter; the full object only via `GET /{id}`), `GET/PUT/DELETE /api/products/{id}`, `POST/DELETE/PUT /api/products/{id}/codebases|specs|links|databases/{id}`, `POST .../codebases|specs|databases/{id}/generate` + status (202 + job_id), `POST /api/products/{id}/ask` (SSE) + `/ask/doc`, chat sessions CRUD, `/api/admin/mcp/servers`, `/api/products/{id}/mcp` (bindings), inbound MCP at `/api/mcp`.

### Two-Process Architecture
- **Frontend**: Next.js on port 3000. Proxies API calls to the backend via rewrites in `next.config.ts` (`/api/*` → `SERVER_BASE_URL`, default `http://localhost:8001`).
- **Backend**: FastAPI on port 8001 (`api/api.py` is the main app, started via `api/main.py`).
- Communication: REST (SSE streaming) + WebSocket.

### Backend Modules (`api/`)
- **`main.py`** — entry point: loads `.env`, logging, uvicorn.
- **`api.py`** — app assembly: CORS from `CORS_ORIGINS` (explicit allowlist; `*` disables credentials), `include_all_routers(app)`, inbound MCP mount, lifespan (`init_db()` + memory init + admin bootstrap, all non-fatal).
- **`llm/`** — LLM foundation: `client.py` (single OpenAI-compatible stack: langchain `ChatOpenAI`/`OpenAIEmbeddings` over the patched openai SDK), `stream.py`, `generate.py`.
- **`agents/`** — `runtime.py` (process-wide LangGraph checkpointer: Postgres, SQLite fallback), `expert.py` (expert graph), `tools.py` (read-only path-confined codebase tools, symlink-safe via `open_read_nofollow`).
- **`expert/`** — `chat.py` (SSE + session persistence), `generate.py` (standalone doc), `deep_research.py` (bounded multi-iteration loop), `knowledge.py`, `prompt.py`, `llm.py`, `types.py`.
- **`docgen/`** — `codebase.py` (deepagents pipeline + repo tools + DB digest in the repo brief), `spec.py` (skeleton + LangGraph enrichment), `database.py` (MCP introspection RE: role-classified walk with dbhub/oracle adapters + read-only SQL catalog packs → root pages + per-entity subpages + batched strict-JSON LLM enrichment), `introspection_cache.py` (walk-payload disk cache, `CACHE_FORMAT_VERSION` 3, budget-keyed), `corroborate.py` (grounding of LLM text against the introspection payload), `verification.py` (citations, LLM judge `DOCGEN_JUDGE_ENABLED`, Mermaid verify/repair, provenance diff, DSN masking), `jobs.py` (async 202+poll, `DOCGEN_MAX_WORKERS`), `summary.py`, `citation_guard.py` / `prose_dedup.py` / `fact_fold.py` (text-quality passes), `_common.py`.
- **`memory/`** — `resolver.py` (backend picker), `pgvector_backend.py` (index + cosine recall over `knowledge_chunks`), `base.py` (interface). SQLite degraded → no recall, callers fall back.
- **`mcp/`** — `manager.py` (outbound tool cache over `langchain-mcp-adapters` `MultiServerMCPClient`; fingerprint-keyed discovery cache; negative cache; bounded by `MCP_*` timeouts), `secrets.py` (Fernet encrypt/mask of headers/env), `policy.py` (stdio command validation — no shells/interpreters), `inbound.py` (FastMCP at `/api/mcp`, Bearer API-token auth; tools `list_products`/`get_product_knowledge`/`search_knowledge`/`ask_expert`).
- **`repositories/product_repo.py`** — all entity DB access for the products router family.
- **`routers/`** — auto-discovered (add `api/routers/<name>.py` with module-level `router = APIRouter(...)` — it connects automatically): `products.py`, `databases.py`, `docgen.py`, `expert.py`, `knowledge.py`, `integrations.py`, `mcp_admin.py`, `product_mcp.py`, `public.py` (API-token export/ask/push of verified knowledge), `admin.py`, `misc.py` (`/lang/config`, `/health` — public by design; the browser reaches them via the Next `/api` proxy rewrite). Products/databases/docgen enforce router-level auth (`get_current_user`).
- **`auth/`** — `deps.py` (`get_current_user`, `require_admin`, `require_api_token`), `local.py` (bcrypt + reset tokens), `keycloak.py` (OIDC), `tokens.py` (session JWTs, httpOnly cookie, `COOKIE_SECURE`), `bootstrap.py`. `AUTH_PROVIDER`: `local` | `keycloak` | `both` | `none`.
- **`integrations/`** — auto-discovered via `pkgutil`; connectors implement `test()`/`list_spaces()`/`pull()`: `github`, `gitlab` (+`_git_base`), `confluence`, `mcp`. Git pulls create Codebases; non-git pulls create knowledge nodes.
- **`config/`** — `__init__.py` (JSON loader, `${ENV_VAR}` placeholders), `settings.py` (Fernet-encrypted admin store), `timeout.py` (**the authoritative timeout registry** — every key: env var + default + floor), `ssl.py`, `abstraction.py`.
- **`prompts.py`** — prompt registry + loader; bodies in `refs/prompts/*.md`; `load_prompt_file()` wraps via `_wrap_prompt(content, language)`.
- **`llm/client.py`** — low-level OpenAI-compatible client + `is_local_endpoint` guard (exact-hostname localhost check; no suffix matching).
- **`tools/rate_limiter.py`** — embedder rate limiting (semaphore + spacing + 429 retry).
- **`utils/`** — `logging.py` (console-only; logfmt/json), `fs.py` (`open_read_nofollow` — O_NOFOLLOW), `llm_helpers.py` (`cap`), `llm_tokens.py` (context window via `RLM_MODEL_CONTEXT_WINDOW`, token counting).
- **`formats/mermaid.py`** — Node-based Mermaid verification + bounded LLM repair (`MERMAID_VERIFY` master switch).

### Frontend Structure (`src/`) — minimalist-ui (Notion/Linear editorial)
Warm monochrome, Geist font + system serif headings, Phosphor icons, bento grids, no gradients. Built with **bun**.
- `app/page.tsx` — products dashboard (bento grid, inline create/delete).
- `app/products/[productId]/page.tsx` — product detail: codebase cards + generate, spec sidebar, links spoiler, database cards (masked DSN, MCP pin, RE generate), expert agent panel, knowledge tree, MCP bindings.
- `app/products/[productId]/artifacts/[artifactId]/page.tsx` — entity docs viewer (codebase/spec/links/database via `findEntity`): pages nav tree, Markdown + Mermaid, scoped Ask, editor.
- `components/` — `Ask.tsx` (chat + Deep Research toggle), `ExpertChat.tsx` (SSE + sessions), `Mermaid.tsx` (pan/zoom + auto-fix), `Markdown.tsx`, `knowledge/KnowledgeTree.tsx`, shared `ui.tsx`.
- `contexts/LanguageContext.tsx` — i18n via next-intl (`src/messages/{lang}.json`).
- Proxy pattern: `next.config.ts` rewrites `/api/*` → `SERVER_BASE_URL`; WebSockets go direct.

### Data Flow (Product → Entity → Docs)
1. User creates a **Product** and adds a **Codebase**/**Spec**/**Links**/**Database**.
2. Generate (per-type async endpoint, 202 + job_id):
   - **codebase**: shallow clone (`--depth=1`) into the managed state dir → **deepagents** agent explores the clone with read-only path-confined tools (`repo_list_files`/`repo_read_file`/`repo_grep`) → 7 wiki sections from `refs/prompts/*.md` (Functional/Technical/Data Model are parents with per-capability/per-API/per-store subpages planned by a decomposer LLM call; units share a notes workspace for context reuse) → per-unit **verification** (citations, LLM judge, Mermaid verify/repair, provenance diff) → `generated_docs` + `pages` persisted → background pgvector indexing.
   - **spec**: stdlib parse → Markdown skeleton + LangGraph LLM enrichment → verification → indexing.
   - **database**: MCP introspection walk over the product's bound MCP servers (role classifier; dbhub/oracle adapters; read-only SQL catalog packs; disk cache `introspection_cache.py`) → deterministic skeleton → page tree = Overview + Tables root (brief descriptions, Relationships, Mermaid ER from the FK graph; with zero FK edges an LLM may infer relations, marked "inferred") + per-table subpages (columns/indexes/constraints/DDL, `relatedPages` by FK adjacency) + category roots (views+matviews / triggers / routines / sequences / types) with per-object subpages; subpage cap `DB_DOCGEN_MAX_SUBPAGES` folds surplus onto root pages (never drops); LLM enrichment is batched strict-JSON (`DB_DOCGEN_ENRICH_BATCH`, admin-tunable) with corroborate/judge/Mermaid-repair verification and per-page provenance → indexing. Raw DSN is masked on acceptance and never stored.
   - **links**: storage only.
3. **Expert Agent** (`POST /api/products/{id}/ask`) streams SSE chat over all indexed knowledge with persistent per-user sessions (LangGraph checkpointer); Deep Research = bounded multi-iteration loop; `POST /api/products/{id}/ask/doc` → self-contained Markdown.

## Key Patterns

- **Single OpenAI-compatible path**: one client stack (`api/llm/`) covers every local server (LM Studio, llama.cpp, vLLM). No provider threading. `api/config/generator.json` lists models.
- **Externalized prompts**: bodies in `refs/prompts/*.md`, loaded via `load_prompt_file()`; substitution uses `str.replace` (not `.format`) so Mermaid/JSON braces stay unescaped.
- **Verification-first docgen**: `docgen/verification.py` — citations checked against sources the agent actually read; LLM judge (`DOCGEN_JUDGE_ENABLED`); Node-based Mermaid verify + bounded repair; tree-hash provenance fingerprint decides reuse vs regenerate.
- **Bounded everything**: every external call has a timeout from `api/config/timeout.py` (admin store > env > default, per-key floor). MCP tool calls also result-capped (`MCP_TOOL_RESULT_MAX_CHARS`).
- **Secret hygiene**: settings-store secrets and MCP headers/env Fernet-encrypted at rest, masked on read; DSNs masked before persistence; client-facing error details generic, details only in server logs.
- **Symlink-safe file tools**: `api/utils/fs.py:open_read_nofollow` (O_NOFOLLOW) + realpath confinement on every agent/docgen file read.
- **Non-fatal initialization**: DB down, memory backend down, dead MCP server → warnings + fallbacks; the app always starts.
- **Auto-discovery**: routers (module-level `router`), integrations (`pkgutil`).
- **RBAC + grants**: global roles (`admin|manager|viewer_global|user`) on `productarium_users.role` plus per-product `ro|rw` grants (`product_grants`); enforced via `api/auth/deps.py:require_product_access` / `require_role`. Keycloak role mapping configurable (`KEYCLOAK_ROLE_MAPPING` / admin panel).
- **Write-only secrets in API responses**: `Codebase.token` is Fernet-encrypted at rest and never returned — responses expose `has_token: bool`; raw DB DSNs are masked on acceptance (`dsn_masked`).
- **Rate limits**: per-user token buckets on docgen generate / expert ask / public ask; per-IP on `/login` and `/reset-password` (`api/utils/rate_limit.py`; 429 + `Retry-After`, configurable in admin settings).
- **Entity locks**: API entity writes serialize with running docgen jobs via the refcounted per-entity lock (`api/docgen/jobs.py:lock_for_entity`); contention maps to HTTP 409 `EntityBusyError`.
- **Light product list, bare-list contract**: `GET /api/products` returns a plain JSON array of light rows (counters via correlated SQL subqueries in `product_repo.list_products_light` — no Text payloads loaded); pagination via `limit`/`offset`, filtered total via the `X-Total-Count` header. The response shape stays a bare list (no `{items,total}` envelope) for backward compatibility.

## Environment Variables

**No cloud API keys required.** See `.env.example` for the full, documented list. Key groups:
- **Local OpenAI-compatible API**: `LOCAL_OPENAI_BASE_URL` / `LOCAL_OPENAI_API_KEY` / `LOCAL_OPENAI_MODEL`.
- **Database**: `DB_PROVIDER` (`postgres`|`sqlite`), `DB_HOST/PORT/NAME/USERNAME/PASSWORD`, `PRODUCTARIUM_STATE_DIR`, `PRODUCTARIUM_ALLOW_LOCAL_CLONES`.
- **Timeouts** (registry in `api/config/timeout.py`): `LLM_REQUEST_TIMEOUT_SECONDS`, `LLM_RETRY_MAX_TIME_SECONDS`, `MODEL_LIST_TIMEOUT_SECONDS`, `PROVIDER_TEST_TIMEOUT_SECONDS`, `DOCGEN_INDEXING_DRAIN_SECONDS`, `MEMORY_QUERY_TIMEOUT_SECONDS`, `INTEGRATION_HTTP_TIMEOUT_SECONDS`, `GIT_FILE_CONTENT_TIMEOUT_SECONDS`, `MCP_STDIO_WAIT_SECONDS`, `MERMAID_VERIFY_TIMEOUT`, `MERMAID_REPAIR_TIMEOUT`, `MERMAID_MAX_REPAIR_ATTEMPTS`.
- **Docgen/expert/DB-RE**: `DOCGEN_MAX_WORKERS`, `DOCGEN_JUDGE_ENABLED`, `MERMAID_VERIFY`, `RLM_MODEL_CONTEXT_WINDOW` (legacy-named context-window knob), `DEEP_RESEARCH_TIMEOUT_SECONDS`, `DB_INTROSPECTION_TIMEOUT_SECONDS`, DB-RE counts (admin panel → Timeouts → Databases; admin store > env > default, floor): `DB_DOCGEN_ENRICH_BATCH` (tables per LLM description batch, default 40), `DB_DOCGEN_MAX_SUBPAGES` (200), `DB_DOCGEN_MAX_DESCRIPTIONS` (250), `DB_FK_EVIDENCE_TABLES` (300) / `DB_SOURCE_OBJECTS` (100) (walk budgets, part of the introspection cache key), and `DOCGEN_DB_CONTEXT_ENABLED` (DB digest into codebase docgen briefs, default on).
- **MCP**: `MCP_DISCOVERY_TIMEOUT_SECONDS`, `MCP_ASK_TIMEOUT_SECONDS`, `MCP_TOOL_CALL_TIMEOUT_SECONDS`, `MCP_TOOL_RESULT_MAX_CHARS`, `MCP_NEGATIVE_CACHE_SECONDS`.
- **Embedder**: `EMBEDDER_MAX_CONCURRENCY`, `EMBEDDER_DELAY_SECONDS`, `EMBEDDER_RATE_LIMIT_RPS`.
- **Auth/security**: `AUTH_PROVIDER`, `BOOTSTRAP_ADMIN_*`, `SETTINGS_SECRET_KEY`, `JWT_SECRET_KEY`, `SESSION_TOKEN_TTL`, `COOKIE_SECURE`, `CORS_ORIGINS`; Keycloak: `KEYCLOAK_*`.
- **Integrations**: `GITHUB_ENTERPRISE_URL`, `GITLAB_SELF_HOSTED_URL`, `CONFLUENCE_*` (env fallbacks; admin panel is primary).
- **App/logging**: `PORT`, `SERVER_BASE_URL`, `DEEPWIKI_CONFIG_DIR`, `LOG_LEVEL`, `LOG_FORMAT`, `LOG_MAX_RECORD_CHARS`.

## Additional Documentation
- `README.md` — full architecture/docs (current).
- `PROMPT.md` — detailed technical specification (in Russian).
- `refs/` — reference docs + `refs/prompts/*.md` (all externalized prompt bodies).
- `api/README.md` — backend-specific documentation (partially outdated).
