# Docgen section contracts (parsed into SECTION_PROMPTS)

ONE file with the content contracts for all 7 wiki sections. Parsed at load
time by `api.prompts._parse_section_blocks` into `SECTION_PROMPTS`
(section_id -> contract body). The blocks are embedded into each section
subagent's task by `api.docgen.codebase`; the OUTPUT language is controlled
by `docgen_agent_system.md` (`{language_name}`), so the contracts below stay
in English.

Sections `functional`, `technical` and `datamodel` are PARENT pages: each is
decomposed into per-capability / per-API / per-store SUBPAGES whose
contracts live in `docgen_subpages.md` (planned by `docgen_decomposer.md`).
The parent contract describes only the overview/aggregation layer and
cross-references its subpages.

Format: `<section id="..."> ... </section>` blocks. One block per section,
ids must match `api.prompts.WIKI_SECTIONS`.

<section id="overview">
# Section: Overview (Общая информация)

## Goal
Describe what the project IS: purpose, tech stack, ALL architectural
capabilities, requirements, and project structure. This is the reader's
entry point and the input for later gap analysis.

## Where to look first
README*, package manifests (`package.json`, `pyproject.toml`,
`requirements.txt`, `go.mod`, `Cargo.toml`, `pom.xml`), entry points
(`main.*`, `app.*`, `index.*`), top-level config files.

## Content contract (Markdown headings in this order)
1. `## 1. Name and short description` — what the project does, its main goal,
   target audience (from README/manifests; no marketing inventions).
2. `## 2. Technology stack` — a table:
   `| Category | Technology | Version (if stated) | Purpose | Source |`
   Versions ONLY where explicitly pinned in manifests.
3. `## 3. Capabilities` — ALL architectural capabilities of the system, not
   only the key ones: this list is the input for gap analysis. Group when
   natural (core / supporting / cross-cutting). Each capability: 1-2
   sentences + links to implementing files. Include infrastructure and
   operational capabilities (auth, caching, background jobs, integrations,
   CLI, configuration, observability) when present in code.
4. `## 4. System requirements` — brief: runtime (language version, OS),
   required services (database, queues, external APIs), requests/limits
   (ports, memory/CPU when stated), deployment specifics.
5. `## 5. Project structure` — key directories and their purpose (a compact
   tree in a code block or a small Mermaid `flowchart`).
6. `## 6. Status and license` — only if present in the repo (version,
   license, repo/docs links).

## Rules
- Every stack/capability claim needs a `path` citation.
- Do not invent versions, dependencies or features.
- Keep the 6 headings even on small repos; compress descriptions instead of
  dropping headings.
</section>

<section id="architecture">
# Section: System Architecture (Системная архитектура, C4)

## Goal
Describe the system architecture in C4 notation, derived ONLY from the
actual code: entry points, containers, components, actors, external systems,
data flows.

## Where to look first
Entry points and server bootstrap files, router/API definitions, dependency
injection wiring, clients of external systems (DB SDKs, queues, HTTP
clients), deployment configs.

## Content contract (Markdown headings in this order)
1. `## 1. C4 Context` — Mermaid `C4Context` diagram: users/roles, the
   system, external systems it talks to.
2. `## 2. C4 Containers` — Mermaid `C4Container` diagram: applications,
   services, databases, queues; technology of each; relations between them.
3. `## 3. C4 Components` — Mermaid `C4Component` diagram detailing the key
   containers (skip only if the repo is genuinely too small to have
   components — then say so explicitly).
4. `## 4. Component descriptions` — for each component: purpose,
   technology/framework, key implementation files, related APIs.
5. `## 5. Data flows` — main request/data flows, message queues, API
   integrations.

## Diagram rules
- Every diagram node must map to a real module/file; cite the file next to
  each component description.
- If C4 syntax is not available in the target renderer, express the same
  structure with `flowchart`/`graph` — a correctly rendering diagram wins.
- Cross-reference the Overview and Technical Details sections by title
  instead of repeating their content.
</section>

<section id="functional">
# Section: Functional Description (Функциональное описание) — parent page

## Goal
The capability map: WHAT the system does for its users, at the overview
level. Each capability's detailed business-process description lives on its
own SUBPAGE (see `Subpages`); this page aggregates and cross-links them.

## Where to look first
Router/endpoint definitions, handler/controller modules, CLI commands,
background jobs/workers, feature flags, README feature lists.

## Content contract (Markdown headings in this order)
1. `## 1. Capability map` — ALL functional capabilities, each 1-2 sentences:
   business value, actors, entry points (endpoint/CLI/job). Show relations
   between capabilities (which uses/extends which). Link every capability to
   its subpage by title.
2. `## 2. Actor–capability matrix` — who (user roles, admins, service
   accounts, external systems) uses which capability, if visible in code.
3. `## 3. Constraints and assumptions` — ONLY what is explicitly present in
   the code: stubs, `TODO`/`FIXME`/`HACK`/`XXX` markers,
   `NotImplementedError`, mocked behavior, documented assumptions in
   comments/docstrings. Each item cites the exact file:line. No speculation.
4. `## 4. Subpages` — list of this section's subpages (one line each:
   title + what it covers).

## Rules
- Do NOT include use-case diagrams, user stories or business-logic
  narratives here — those live in the capability subpages.
- Endpoints/commands mentioned must come from actual route/CLI registrations.
- Use the provided subpage list verbatim (titles and page ids) when linking.
</section>

<section id="technical">
# Section: Technical Details (Технические детали) — parent page

## Goal
The engineering overview: the full API surface, configuration, security
posture, error handling and caching. The detailed per-endpoint / per-job /
per-integration specs live on SUBPAGES; this page aggregates and
links them.

## Where to look first
Config files and loaders (`.env.example`, `config/*`, settings modules),
logging setup, middleware, exception handlers, caches, rate limiters,
dependency versions in manifests, lockfiles (summarize, do not dump).

## Content contract (Markdown headings in this order)
1. `## 1. API surface` — table of ALL endpoints (sync and async) and jobs:
   `| Method | Path/Job | Purpose | Capability | Source |` (from route
   definitions and worker registrations only). Mark async/streaming
   endpoints. Link each row's item to its subpage when one exists.
2. `## 2. Configuration` — key config parameters and env vars:
   `| Parameter | Default | Purpose | Source |`. NEVER copy secret VALUES
   from `.env`-like files — names and purposes only.
3. `## 3. Security` — auth mechanisms, secret handling, path confinement,
   CORS, input validation as implemented.
4. `## 4. Error handling` — error strategies, retries, fallbacks, dead
   letter queues where present; the error reference itself lives in the
   relevant subpages.
5. `## 5. Caching and performance` — caches, connection pooling,
   concurrency limits, pagination — only mechanisms actually present in
   code.
6. `## 6. Subpages` — list of this section's subpages (one line each:
   title + kind + what it covers).

## Rules
- No recommendations for what the team SHOULD do — describe what IS.
- Missing area? Keep the heading with "No data in the provided context".
</section>

<section id="cicd">
# Section: CI/CD and SRE (Непрерывная интеграция, поставка и эксплуатация)

## Goal
Describe the actual build/deploy machinery (except testing details — see
the QA section) plus everything the repo has on SRE: metrics, logs, traces,
health checks, alerting, runbook/support information.

## Where to look first
`.github/workflows/*`, `.gitlab-ci.yml`, `Jenkinsfile`, `azure-pipelines.*`,
`Dockerfile*`, `docker-compose*`, `Makefile`, build scripts, release
configs, helm/k8s manifests; metrics/log/trace setup code (middleware,
exporters, instrumentation), health endpoints, alert rules, dashboards.

## Content contract (Markdown headings in this order)
1. `## 1. Pipelines` — each pipeline/workflow: triggers, stages, jobs.
   A compact Mermaid `flowchart` of the stage graph when there are 3+ stages.
2. `## 2. Build` — build tools, artifacts, caching.
3. `## 3. Containers` — Dockerfile strategy (multi-stage, base images),
   compose services and their purpose.
4. `## 4. Deployment` — deploy targets and mechanisms (from CI/CD files and
   manifests only).
5. `## 5. Release process` — versioning, tags, release automation when
   present.
6. `## 6. SRE` — only what actually exists in the repo, as subheadings:
   `### Metrics` (exporters, counters, dashboards-as-code), `### Logging`
   (format, levels, sinks), `### Tracing` (instrumentation, samplers),
   `### Health checks` (liveness/readiness endpoints and what they verify),
   `### Alerting` (alert rules, notifications), `### Runbook / support`
   (on-call info, operational procedures documented in the repo). Mark
   absent subareas explicitly with a one-line "not present in the
   repository" note.

## Rules
- Cite the workflow/manifest/instrumentation file for every claim.
- Absent CI: say explicitly what is absent — do not invent a pipeline.
- Testing configuration in CI is summarized here only as pipeline stages;
  the test suites themselves are documented in the QA section.
</section>

<section id="qa">
# Section: QA — Testing (Тестирование)

## Goal
Describe how the project is tested: frameworks and runners, the test
taxonomy actually present, what is really covered (from the test suites and
CI configs, not from aspiration), fixtures and factories, and the quality
tooling around the tests.

## Where to look first
Test directories (`tests/*`, `test/*`, `*_test.*`, `*.spec.*`),
pytest/jest/go-test configs (`pytest.ini`, `pyproject.toml`,
`jest.config.*`, `Makefile` test targets), CI workflow test steps,
fixtures/conftest, factories, coverage configs (`.coveragerc`,
istanbul/nyc configs).

## Content contract (Markdown headings in this order)
1. `## 1. Frameworks and runners` — test frameworks, key runner commands
   (from configs/CI/Makefile), how to run a single test.
2. `## 2. Test taxonomy` — which test TYPES actually exist
   (unit/integration/e2e/contract/smoke/load) and where each lives;
   explicit note for types that are absent.
3. `## 3. Actual coverage` — which components/flows have tests (map test
   files to the code they exercise, from imports and test names), and
   notable UNTESTED areas ("no tests found for X") — deterministic
   statements of fact, no estimated percentages.
4. `## 4. Fixtures, factories and doubles` — test data setup, builders,
   mocks/stubs/fakes, test databases/stores, hermeticity (what the suite
   needs externally and what it mocks).
5. `## 5. Quality tooling` — coverage measurement, linters, type checks,
   formatters wired into the test/CI flow, quality gates.

## Rules
- Every claim cites the test file / config / workflow file.
- Describe what the tests DO check (behavior under test), not just file
  lists.
- No invented coverage numbers: only report numbers a repo artifact states.
</section>

<section id="datamodel">
# Section: Data Model (Модель данных) — parent page

## Goal
Everything the system persists, maximally detailed: databases, schemas, ORM
entities, storage formats, message schemas. Each data layer's full detail
lives on its own SUBPAGE; this page holds the storage overview, the
cross-store entity map and the global ER view.

## Where to look first
ORM models, migrations (`migrations/*`, `alembic/*`, `prisma/*`), schema
files (`*.sql`, GraphQL/OpenAPI schemas), serializer/model classes, document
store collections, message payloads.

## Content contract (Markdown headings in this order)
1. `## 1. Storage overview` — table of ALL stores used:
   `| Store | Technology | Purpose | What lives there | Source |`
   (Postgres, SQLite, Redis, files, vector stores, queues), each linking to
   its subpage.
2. `## 2. Entity map` — per store: the entities/tables/collections it
   holds, one line each (name + purpose + source). Full field lists live in
   the subpages.
3. `## 3. ER diagram` — Mermaid `erDiagram` of the MAIN entities and
   relations ACROSS stores (per-store detail diagrams live in subpages).
4. `## 4. Migrations` — migration tooling and notable schema evolution
   steps.
5. `## 5. Data flows` — who writes/reads what (brief; link to Architecture
   for flows).
6. `## 6. Subpages` — list of this section's subpages (one line each:
   title + layer + what it covers).

## Rules
- Fields/types come from model/migration definitions only.
- No database in the repo? State that explicitly and describe config/DTO
  data shapes instead, if any.
</section>
