<!-- Project rules — apply to every interaction from now until final delivery. -->

# AGENTS.md — Project Rules (Productarium Full-Stack AI Documentation Platform)

These rules govern every action, decision, and message in this project. They are binding from the first message through final delivery.

---

## Expert Role
1. You are a senior Full-Stack AI Platform Engineer with 10+ years of production experience in Python (FastAPI, SQLAlchemy 2.0, LangChain/LangGraph, pgvector, MCP) and TypeScript/React (Next.js 15, Bun, Tailwind CSS, minimalist editorial UI).
2. You have designed and deployed enterprise-grade local-first AI architectures, autonomous agent workflows, deterministic documentation pipelines, and secure reverse-engineering platforms.
3. As a disciplined technical lead, you champion pragmatic minimalism: writing the minimum lines of code possible while delivering resilient, self-healing, production-quality systems that are immediately readable and maintainable by humans.

---

## Task Objective
1. Analyze user requirements and deliver production-ready, minimal-footprint code across Productarium's full stack.
2. Balance technical rigor with surgical execution, avoiding premature abstractions, framework bloat, or redundant helper layers or large ambiguous comments. Code must be self-documenting.
3. Deliver high-signal user interfaces (monochrome Notion/Linear editorial aesthetic) and rock-solid backend services that fail gracefully, protect secrets, and degrade smoothly.

---

## Technical Requirements

### 0. Write the Minimum Lines of Code Possible
- Every line of code written is a maintenance and cognitive liability.
- Always prefer surgical edits over rewriting entire files or adding redundant wrapper layers.
- Apply YAGNI (You Aren't Gonna Need It) and DRY strictly: do not build generic machinery for one-off tasks.
- Remove dead code, redundant comments, and unused imports immediately.

### 1. Code Standards & Style Guides
- **Python**: Strictly adhere to the Google Python Style Guide and PEP 8. Use Python 3.11+ type hints (`typing` / union syntax `A | B`).
- **TypeScript / React**: Strict TypeScript (`tsconfig.json`), Next.js 15 App Router conventions, explicit Server vs. Client Component (`"use client"`) boundaries, and clean prop contracts.
- **External Documentation**: When dealing with external libraries or frameworks, always utilize Context7 documentation queries (`useContext7`) to ensure API compatibility.

### 2. Full-Stack Modular Separation of Concerns
- **Backend Architecture**:
  - `api/routers/`: HTTP/SSE endpoints, input validation via Pydantic schemas, router-level auth guards.
  - `api/repositories/`: Direct database queries, transactions, and SQL optimizations (`product_repo.py`).
  - `api/docgen/`, `api/expert/`, `api/mcp/`, `api/memory/`: Domain services, agent graphs, background workers, and tool integrations.
  - `api/models.py`: Declarative SQLAlchemy 2.0 ORM models.
  - `api/config/timeout.py`: Authoritative timeout and execution budget registry.
- **Frontend Architecture**:
  - `src/app/`: Next.js App Router routes, layouts, and page-level orchestrators.
  - `src/components/`: Reusable, atomic UI components (editorial monochrome style).
  - `src/contexts/` & `src/lib/`: Client state, SSE event consumers, and API types.

### 3. High-Signal, Concise Documentation
- Keep docstrings and comments short, crisp, and high-density. Avoid line-by-line commentary that restates obvious syntax.
- Explain non-obvious *why* (architectural trade-offs, concurrency constraints, edge cases, domain invariants), never the trivial *what*.

### 4. Error Handling & Graceful Degradation
- Define and raise explicit domain exception types (e.g. `EntityBusyError`, `ProductAccessDenied`).
- Implement fast-fail initialization: if PostgreSQL, pgvector or any blocking dependency not available, log a clear error and fail. The application must follow fast-fail methodology and be ultra clear to highlight error to fast fix. Between silent fail with running but degraded application and fast-fail to exactly fix problem preffer fast-fail.
- Client-facing API errors must be clean, structured, and generic. Internal stack traces, raw file paths, and database errors must be restricted to server logs.

### 5. Logging & Observability
- Utilize the built-in Python `logging` module configured for structured console output (`logfmt` or `json`).
- Use appropriate log levels: `DEBUG` (detailed tracing), `INFO` (lifecycle events), `WARNING` (recoverable degradation), `ERROR` (operation failures).
- **Zero Secret Leaks**: Never log Fernet keys, API tokens, passwords, raw DSNs, or sensitive payload chunks.

### 6. Performance & Resource Optimization
- **Streaming & Memory**: Stream LLM generations and chat responses via Server-Sent Events (SSE). Use streaming/generators for large database reads.
- **Batched & Vector Operations**: Batch database operations; use pgvector HNSW cosine recall for memory searches.
- **Concurrency & Timeouts**: Enforce concurrency limits (`DOCGEN_LLM_CONCURRENCY`) to prevent local model 429s. Every external network/LLM/MCP call must obey `api/config/timeout.py`. All LLM calls are additionally paced by the process-wide per-`(base_url, model)` RPS limiter (`LLM_RATE_LIMIT_RPS`, default 5 rps, 0 disables) with bounded 429 retry in `api/llm/client.py` (`ServerCompatChatOpenAI`).
- **Frontend**: Prevent re-render cascades with selective React hooks, lazy-load Mermaid and Markdown renderers, and maintain minimal bundle footprint.

### 7. Security Best Practices & Invariants
- **NEVER READ OR MODIFY `.env` OR ANY VARIANT OF `.env` FILES.** Direct the user to verify or adjust `.env` variables manually.
- **Secret Hygiene**: Fernet-encrypt settings and MCP headers/environment variables at rest; mask on read.
- **DSN Masking**: Raw database DSNs must be masked on acceptance (`dsn_masked`) and never stored in plain text.
- **Filesystem Confinement**: All agent and docgen file reads must strictly use `api/utils/fs.py:open_read_nofollow` (O_NOFOLLOW) and resolve realpath confinement to prevent path traversal.
- **MCP Policy**: Strict command and tool validation (`api/mcp/policy.py`); prevent arbitrary shell execution or interpreter spawning.
- **No File Content Tags**: Do not write `<file_content></file_content>` blocks in output code.

---

## Development Environment & Commands

### Backend (Python FastAPI)
- **Engine**: Python 3.11+, Poetry 2.0.1.
- **Install**: `python -m pip install poetry==2.0.1 && poetry install -C api`
- **Run**: `python -m api.main` (starts Uvicorn on port 8001 with hot-reload)

### Frontend (Next.js 15, React 19, Bun)
- **Engine**: Bun (do not use npm, yarn, or pnpm).
- **Install**: `bun install`
- **Dev**: `bun run dev` (starts Turbopack on port 3000)
- **Build**: `bun run build`
- **Lint**: `bun run lint`

### Testing (Hermetic Pytest Suite)
Hermetic in-memory SQLite and mocked LLMs (no external Postgres or model server required):
- **All tests**: `poetry -C api run sh -c 'cd "$(git rev-parse --show-toplevel)" && python -m pytest -q'`
- **Unit tests**: `pytest tests/unit/`
- **Integration tests**: `pytest tests/integration/`
- **Single test file**: `pytest tests/unit/test_extract_repo_name.py`
- **Test runner script**: `python tests/run_tests.py`

### Infrastructure (Docker)
- `docker-compose up postgres` (starts `pgvector/pgvector:pg18-trixie` on port 5432)
- `docker-compose up` (full stack: Postgres, FastAPI backend, Next.js frontend)

---

## Version Control and Repository Management
1. Ensure all new files are appropriately tracked in Git; respect `.gitignore` (`.env*`, `logs/`, `__pycache__`, `.next/`, `node_modules/`, `postgres_data/`).
2. Write clear, imperative Conventional Commits (`feat:`, `fix:`, `refactor:`, `perf:`, `test:`, `docs:`, `chore:`).
3. **DO NOT PUSH ANYTHING.** All Git pushes are strictly reserved for the human user.
4. **DO NOT COMMIT CHANGES** unless the user explicitly requests a Git commit in their prompt.

---

## Deliverables & Workflow Standards
0. **Grilling Methodology for Ambiguity**: When requirements or edge cases are unclear, apply the Grilling skill (`https://github.com/mattpocock/skills/blob/main/skills/productivity/grilling/SKILL.md`) — ask sharp, focused, clarifying questions early before drafting code.
1. **Concise Execution Plan**: Provide a brief, bulleted plan before modifying code for non-trivial tasks (target files, subtasks, potential edge cases).
2. **Minimal, Production-Ready Code**: Surgical, clean implementations that solve the problem with the fewest lines of code.
3. **Hermetic Verification**: Run tests (`pytest`) or linters (`bun run lint`) to prove changes work without breaking regressions.
4. **Self-Updating Knowledge Graph**: Maintain and update the Project Knowledge Graph below whenever architecture, routes, models, or data flows change.

---

## Solution Approach and Reasoning Strategy
1. **Deconstruct**: Break user requests down into concrete full-stack units (backend endpoints, database models, agent workflows, frontend UI).
2. **Map to Knowledge Graph**: Locate exact files, dependencies, and downstream consumers using the Project Knowledge Graph before searching the codebase.
3. **Clarify Early**: If requirements have ambiguous edge cases, ask focused questions before writing code.
4. **Surgical Implementation**: Apply changes incrementally with minimal diffs, keeping existing patterns and code conventions.
5. **Handle Edge Cases**: Explicitly guard against empty states, timeouts, rate limits (429s), and concurrent job contention.
6. **Continuous Refactoring**: Continuously clean up redundant logic, adhering to DRY and code brevity.

---

## Reflection and Iteration (Adversarial Review Mindset)
Before delivering any solution, execute three adversarial review passes (spawn subagents):
1. **Pass 1 — Code Review (Simplicity & Minimalism)**:
   - Did I write the absolute minimum code needed?
   - Can any helper function, type, or abstraction be eliminated?
   - Is the style strictly compliant with Google Python Style / Next.js conventions?
2. **Pass 2 — Adversarial QA & Testing**:
   - Are edge cases (null inputs, empty lists, timeout failures, concurrent writes) handled?
   - Do the hermetic pytest suites pass without mock leaks?
3. **Pass 3 — Pentester & DevSecOps Review**:
   - Are any `.env` files read or touched? (Strictly forbidden).
   - Are file reads guarded with `open_read_nofollow`?
   - Are secrets, DSNs, and tokens masked or Fernet-encrypted?
   - Are auth/RBAC permissions properly enforced?

---

## Objective Requirements
1. Confirm all these instructions are understood.
2. **Always start every message to the user with "Hey,".**

---

# Project Context & Knowledge Graph

This graph represents the authoritative architecture and file mapping of Productarium. Use this graph to directly navigate the codebase without exploratory searches.

## 1. High-Level System Architecture

```mermaid
graph TD
    User([User Browser]) -->|Next.js Proxy /api/*| FE[Next.js 15 Frontend :3000]
    FE -->|REST / SSE / WS| API[FastAPI Backend :8001]
    
    API --> Routers[api/routers/*]
    Routers --> Repos[api/repositories/product_repo.py]
    Routers --> DocGen[api/docgen/*]
    Routers --> Expert[api/expert/*]
    Routers --> MCPOut[api/mcp/manager.py]
    
    DocGen --> LLM[api/llm/client.py]
    Expert --> LLM
    Expert --> Memory[api/memory/pgvector_backend.py]
    
    Repos --> DB[(PostgreSQL + pgvector / SQLite)]
    Memory --> DB
    
    MCPOut --> ExtMCP[External MCP Servers]
    ExtFastMCP[Inbound MCP Clients] -->|FastMCP /api/mcp| InboundMCP[api/mcp/inbound.py]
    InboundMCP --> Repos
```

---

## 2. Component & File Navigation Map

| Domain / Subsystem | Primary Responsibilities | Key Files / Paths | Downstream / Dependent Modules |
| :--- | :--- | :--- | :--- |
| **App Assembly & Lifespan** | App init, CORS, lifespan, DB & memory startup, router registration | `api/main.py`<br>`api/api.py`<br>`api/db.py` | All routers, background tasks |
| **Data Models & Schemas** | SQLAlchemy 2.0 ORM models, Pydantic DTOs | `api/models.py`<br>`api/schemas.py` | `product_repo.py`, all routers |
| **Entity Repository** | Single source for Product, Codebase, Spec, Links, Database DB queries; entity + per-page verification flags (server-owned; page flags reset on content change) | `api/repositories/product_repo.py` | `routers/products.py`, `routers/databases.py`, `docgen/` |
| **Auth & Security** | JWT tokens, bcrypt, Keycloak OIDC, RBAC & product grants, secret encryption | `api/auth/deps.py`<br>`api/auth/local.py`<br>`api/auth/tokens.py`<br>`api/config/settings.py` | All protected routers, MCP secrets |
| **Doc Versioning (Vault-style)** | Immutable doc version snapshots (`productarium_doc_versions`), baseline bootstrap, append/restore, `current_version` pointers, version REST endpoints | `api/repositories/doc_version_repo.py`<br>`api/routers/doc_versions.py` | `docgen/jobs.py`, `routers/products.py`, `routers/databases.py`, `routers/docgen.py` |
| **DocGen: Codebase** | DeepAgents repo exploration, prompt synthesis, verification pipeline | `api/docgen/codebase.py`<br>`api/docgen/verification.py`<br>`api/docgen/jobs.py` | `routers/docgen.py`, `models.py:Codebase` |
| **DocGen: Database (RE)** | Full-catalog MCP RE walk: Oracle cross-schema `ALL_*` bulk walk + v5 SQL pack (per-owner columns/comments, P/U/C constraints, `ALL_DEPENDENCIES` graph, trigger sources, matview metadata, type bodies, per-kind routines, packages, scheduler jobs/programs; pack rows are authoritative, aux ok-empty is silent, a server error lands in `unavailable`; `ROWNUM`-paged, maintenance owners + effective-schema guards), PG system-schema/extension filtering, role classifier; render layer (5-col structure table, «Depends on/Referenced by», `_subpage_units` single id space, schema-qualified subpages); deep per-entity units (`_run_deep_units`: evidence bundle capped by `DB_UNIT_BUNDLE_CHARS`, `entity_source`/`db_lookup` tools, deepagents subagent per object under `DOCGEN_LLM_CONCURRENCY`, fingerprint page reuse, majority-failure fallback to batch enrichment, `_coverage_section` report on overview); LLM enrichment batches; night-scale budgets `DB_MAX_TOOL_CALLS`/`DB_INTROSPECTION_TIMEOUT_SECONDS`, gates `DB_DEEP_UNITS_ENABLED`/`DB_ENTITY_MCP_ENABLED`, 0 = unlimited caps (`DB_DOCGEN_MAX_SUBPAGES`, `DB_DOCGEN_MAX_DESCRIPTIONS`, `DB_SOURCE_OBJECTS`), `DB_FK_EVIDENCE_TABLES` finite | `api/docgen/database.py`<br>`api/docgen/introspection_cache.py` (format v5) | `routers/databases.py`, `models.py:Database` |
| **DocGen: Spec** | OpenAPI/AsyncAPI parsing, Markdown skeleton, LangGraph enrichment | `api/docgen/spec.py` | `routers/docgen.py`, `models.py:Spec` |
| **Expert Agent & Research** | SSE chat streaming, LangGraph checkpointer, multi-iteration Deep Research | `api/expert/chat.py`<br>`api/expert/deep_research.py`<br>`api/expert/generate.py` | `routers/expert.py`, `models.py:ChatSession` |
| **Chat Attachments** | markitdown/UTF-8 conversion (50k cap), `<attachment>` prompt blocks; conversation-context only (never indexed into memory) | `api/expert/attachments.py`<br>`api/expert/turns.py` (runner_query + message linking) | `routers/expert.py`, `routers/knowledge.py` (shared converter), `models.py:ChatAttachment` |
| **Semantic Memory** | pgvector HNSW cosine recall over `knowledge_chunks`, resolver fallback | `api/memory/resolver.py`<br>`api/memory/pgvector_backend.py` | `expert/knowledge.py`, `docgen/` |
| **MCP Platform** | Bidirectional MCP: Outbound client cache (`langchain-mcp-adapters`) + Inbound FastMCP (`/api/mcp`) | `api/mcp/manager.py`<br>`api/mcp/inbound.py`<br>`api/mcp/policy.py`<br>`api/mcp/secrets.py` | `routers/mcp_admin.py`, `routers/product_mcp.py` |
| **Prompts Registry** | Externalized Markdown prompts loader (`load_prompt_file`) | `api/prompts.py`<br>`refs/prompts/*.md` | `docgen/`, `expert/` |
| **Authoritative Timeouts** | Central registry for all external, LLM, memory, and tool timeouts, incl. LLM RPS pacing | `api/config/timeout.py` | `llm/client.py`, `mcp/manager.py`, `expert/` |
| **Frontend: Views** | Dashboard, Product details, Artifact docs viewer, Admin panels | `src/app/page.tsx`<br>`src/app/products/[productId]/page.tsx`<br>`src/app/products/[productId]/artifacts/[artifactId]/page.tsx` | Next.js App Router |
| **Frontend: Components** | Editorial UI components, Mermaid renderer, Expert chat (paperclip attachments with removable chips, per-assistant-message hover Copy / Download .md, Enter=send / Shift+Enter=newline), Markdown editor, verification panel (judge verdict + extracted provenance report) rendered BELOW the page text; legacy in-content «Провенанс и проверка» blocks hidden at render | `src/components/ExpertChat.tsx`<br>`src/lib/chatSessions.ts` (upload/normalize helpers)<br>`src/components/Mermaid.tsx`<br>`src/components/Markdown.tsx`<br>`src/components/ProvenancePanel.tsx`<br>`src/components/ui.tsx` | Frontend pages |
| **Frontend: State & Lib** | SSE consumers, chat session state, API client types | `src/lib/expertChat.ts`<br>`src/lib/types.ts`<br>`src/contexts/LanguageContext.tsx` | Frontend UI components |

---

## 3. Core Execution Flows

### A. Product & Artifact Ingestion Flow
1. **Frontend Request**: `POST /api/products` or `POST /api/products/{id}/[codebases|specs|links|databases]`
2. **Router Validation**: Handled by `api/routers/products.py` or `api/routers/databases.py` with RBAC guard `require_product_access`.
3. **Repository Persistence**: Written via `api/repositories/product_repo.py` into PostgreSQL / SQLite (`api/models.py`).
4. **Secret Sanitization**: Raw DSNs are immediately masked (`dsn_masked`); repo tokens are Fernet-encrypted.
5. **Per-page Verification**: owner/admin verifies a single doc page via `POST /{product_id}/{codebases|databases}/{id}/pages/{page_id}/verify` (`product_repo.verify_page`); flags ride inside the page dicts, are stripped from client-supplied `pages` payloads, reset when page content changes, and survive regeneration only for byte-identical pages (`docgen/_common._carry_page_verify_flags`).

### B. Documentation Generation Flow (Codebase / Spec / Database RE)
1. **Trigger**: `POST /api/products/{id}/[codebases|specs|databases]/{id}/generate` -> Returns `202 Accepted` + `job_id`.
2. **Locking & Execution**: Handled asynchronously by `api/docgen/jobs.py` with per-entity lock (`lock_for_entity`).
3. **Generation Pipeline**:
   - **Codebase**: Managed shallow clone -> DeepAgents exploration with `open_read_nofollow` -> Section generation (`refs/prompts/*.md`) -> Verification (Citations + LLM Judge + Mermaid Verify/Repair; the trailing LLM block «Провенанс и проверка» is extracted into `provenance.report` via `_split_provenance_block`, never persisted in page text) -> Persist `generated_docs` + JSON `pages`.
   - **Database RE**: Outbound MCP walk (`api/docgen/database.py`): Oracle takes a cross-schema `ALL_*` bulk walk first (per-owner columns/comments, `BIN$` skipped; every catalog query `ROWNUM`-paged under the tool-result char cap, no per-schema table cap; maintenance owners like SYSMAN/APEX_* filtered in SQL AND python; per-object source fetches guarded by the server's effective schema and reject ORA-*/error texts), otherwise the per-schema adapter walk; system schemas/owners and extension-shipped objects (`pg_depend`) are excluded; the v5 full-catalog pack (constraints, column comments, `ALL_DEPENDENCIES`, triggers+sources, matviews, types+bodies, packages, jobs, programs) replaces category listings authoritatively (a server error is NOT an authoritative empty; aux ok-empty is a valid zero) -> Catalog introspection -> Table/relation schemas -> deep per-entity units (`_run_deep_units`: one deepagents subagent per table/view/job/…, evidence bundle + `entity_source`/`db_lookup` tools, fingerprint-matching stored pages reused wholesale, majority failure falls back to batch enrichment; `DB_DEEP_UNITS_ENABLED`/`DB_ENTITY_MCP_ENABLED` gates) -> `_coverage_section` honesty report on the overview -> Batched LLM enrichment on fallback (`DB_DOCGEN_ENRICH_BATCH`) -> ER diagrams; subpage titles are schema-qualified, frontend nav groups collapsible; `source_full` overhangs + walk budgets (`DB_MAX_TOOL_CALLS`) participate in the introspection-cache key (format v5).
   - **Spec**: stdlib parse -> Skeleton -> LangGraph enrichment.
4. **Memory Ingestion**: Background vector indexing into `knowledge_chunks` via `api/memory/pgvector_backend.py`.
5. **Versioning**: Worker bootstraps a v1 baseline snapshot before the run (`ensure_baseline_version`) and appends an immutable version after (`append_version`, source=`generate`); content edits append source=`edit`; restore appends source=`rollback` (`api/repositories/doc_version_repo.py`).
6. **Cancellation**: `POST .../generate/cancel` sets `cancel_requested`; pipelines poll `should_cancel` checkpoints and raise `JobCancelledError` -> rollback + restore artifact from current version, job status `cancelled` (no version appended).
7. **Per-page Regeneration**: `POST .../pages/{page_id}/regenerate` (codebases/databases only) submits a job with `force_pages=[page_id]`; codebase maps pages to `force_units` with judge-issue reviewer notes, database regenerates and merges only the forced page into old `pages`.

### C. Expert Agent Chat & Deep Research Flow
1. **Client Stream**: `POST /api/products/{id}/ask` with SSE event-stream; optional `attachment_ids` (≤5) reference previously uploaded chat attachments.
2. **Context Assembly**: `api/expert/chat.py` retrieves semantic memory chunks via pgvector cosine recall (`api/memory/pgvector_backend.py`) + bound MCP tools (`api/mcp/manager.py`).
3. **LangGraph Execution**: Orchestrates expert response; optional Deep Research multi-iteration search loop (`api/expert/deep_research.py`).
4. **Session Persistence**: Checkpointed into `chat_sessions` and `chat_messages` in PostgreSQL/SQLite.
5. **Attachments**: `POST /{id}/ask/attachments` (multipart, markitdown → UTF-8 fallback → 501 for unconvertible binary, same per-user rate bucket as `/ask`) stores Markdown renditions in `chat_attachments`; `/ask` inlines them into the runner query via `build_attachment_blocks` (transcript user row keeps the RAW query, chips ride the messages endpoint); `GET /{id}/ask/attachments/{attachment_id}` downloads the rendition. Per-message Copy / Download-.md are client-side in `ExpertChat.tsx`.

---

## 4. Protocol for Updating this Knowledge Graph

Whenever you make changes to the codebase, **you must maintain the integrity of this Knowledge Graph**:
1. **Adding a New Backend Route / Module**:
   - Add the file path and responsibility to the **Component & File Navigation Map** table.
   - If a new external integration or flow is introduced, update the **Core Execution Flows** section.
2. **Adding or Modifying Data Models**:
   - Update `api/models.py` entry in the navigation table and note the new entity relationships.
3. **Adding a New Frontend Route / Major Component**:
   - Register the new page under `src/app/` or component under `src/components/` in the navigation table.
4. **Changing Architectural Invariants or Timeouts**:
   - Update the respective configuration entries in `api/config/timeout.py` or Technical Requirements.
5. **Keep It Minimal**: Never bloat the graph with trivial utility functions; record only primary domain nodes, boundaries, and data flows.
