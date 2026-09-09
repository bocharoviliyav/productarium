# Docgen subpage contracts (parsed into SUBPAGE_CONTRACTS)

Content contracts for the SUBPAGES of the parent sections (`functional`,
`technical`, `datamodel`). Parsed at load time by
`api.prompts._parse_subpage_blocks` into `SUBPAGE_CONTRACTS`
(subpage_type -> contract body). The pipeline substitutes the per-item
identity ({item_title}, {item_focus}, {item_kind}) and the {siblings_list}
into the agent task built from `docgen_agent_section.md`; the contract body
below is embedded verbatim as the "section instruction". Contracts stay in
English (the OUTPUT language is controlled by `docgen_agent_system.md`).

Format: `<subpage id="..."> ... </subpage>` blocks, ids:
`functional_item`, `technical_item`, `datamodel_item`.

<subpage id="functional_item">
# Subpage: one functional capability

## Goal
Exhaustive description of ONE system capability: its business process, the
code that implements it, and its limits. This subpage is the deep-dive the
parent "Functional Description" page links to.

## Content contract (Markdown headings in this order)
1. `## 1. Overview` — what the capability does for the user, the actors
   involved, entry points (endpoints/CLI/jobs) and the expected outcome.
2. `## 2. Implementation` — modules and classes implementing the capability
   with their responsibilities; key functions/methods with REAL signatures
   (verify by reading); configuration knobs and feature flags affecting it.
3. `## 3. Business process (sequence)` — one or more Mermaid
   `sequenceDiagram`s showing the REAL interaction for the main scenario:
   caller → router/handler → service/domain → storage/external systems,
   with actual module/class names from the code.
4. `## 4. Activity (logic and branching)` — one or more Mermaid
   `flowchart`s with explicit start/stop nodes covering the decision logic:
   branches, loops, alternative and error paths of this capability.
5. `## 5. Limitations and edge cases` — behavior at the edges: what the
   code does NOT handle, explicit stubs/TODOs in this area, known
   constraints visible in the implementation.

## Rules
- Every claim cites `path` (with line spans when useful); signatures must
  match the code.
- Diagrams must reflect actual call chains read from the code, not
  invented idealized flows.
- Do not describe other capabilities except via one-line cross-references
  to their subpages or sibling sections.
</subpage>

<subpage id="technical_item">
# Subpage: one technical unit (kind-adaptive)

## Goal
The full technical specification of ONE unit of the system's API surface or
technical machinery. The `kind` given in your task selects the contract
variant below.

## Content contract — kind "endpoint"
1. `## 1. Specification` — method + full path, purpose, auth requirements,
   sync/async/streaming nature, relevant capability.
2. `## 2. Parameters` — table:
   `| Name | Type | Required | Default | Source (path/query/body/header) | Description |`
   from the actual handler/route definition and its models.
3. `## 3. Processing logic` — step-by-step what happens: validation,
   authorization, business calls, side effects; branching conditions made
   explicit.
4. `## 4. Errors` — table `| Code/Status | Message | When it happens |`
   from the actual error paths in the handler chain.
5. `## 5. Sequence diagram` — Mermaid `sequenceDiagram`:
   client → router → service → storage/external, real module names.
6. `## 6. Activity diagram` — Mermaid `flowchart` with start/stop covering
   the handler's branching. Add a Mermaid `stateDiagram-v2` INSTEAD when the
   unit is dominated by a state machine (orders, jobs, sessions).

## Content contract — kind "job"
1. `## 1. Specification` — what the job does, trigger (schedule/queue/event/
   manual), owning module.
2. `## 2. Processing logic` — step-by-step pipeline; inputs consumed,
   outputs produced, side effects.
3. `## 3. Reliability` — idempotency, retries, interruption behavior
   (what happens on crash mid-run), concurrency/locking, dead-letter
   handling — only mechanisms actually present.
4. `## 4. Sequence / activity diagrams` — the real trigger→process→effect
   chain; branching as a `flowchart` when non-trivial.

## Content contract — kind "integration"
1. `## 1. Specification` — external system, purpose, protocol (HTTP/DB/
   queue/SSE...), client module.
2. `## 2. Authentication` — how the client authenticates (mechanism only —
   never secret values).
3. `## 3. Exchange` — requests/responses/payloads (shapes, formats),
   direction, frequency/cadence when visible.
4. `## 4. Failure handling` — timeouts, retries, fallbacks, circuit
   breakers, error mapping as implemented.
5. `## 5. Sequence diagram` — the real exchange flow.

## Content contract — kind "reference"
1. `## 1. Purpose` — what this reference covers and where it lives in the
   code.
2. `## 2. Class/module responsibilities` — table or list: name,
   responsibility, key methods, source path. (For an error-reference unit:
   `| Code | Meaning | Raised where |`.)
3. `## 3. Usage patterns` — how the rest of the code uses these classes/
   codes, with 1-2 real call examples.

## Rules (all kinds)
- Every claim cites `path`; parameters/errors/defaults come from the actual
  code, never from convention.
- Do not duplicate the parent page's tables; this subpage is the DETAIL.
</subpage>

<subpage id="datamodel_item">
# Subpage: one data layer

## Goal
Maximally detailed documentation of ONE store/data layer: every entity,
every field, constraints, indexes, migrations.

## Content contract (Markdown headings in this order)
1. `## 1. Overview` — the store's technology, purpose, owning modules/
   clients, connection configuration (names only, never secret values).
2. `## 2. Entities` — per entity/table/collection:
   `| Field | Type | Constraints | Default | Source |` — ALL fields from
   the model/migration definitions, including indexes and uniqueness
   constraints (state them under the table or as a separate list).
3. `## 3. ER diagram` — Mermaid `erDiagram` of THIS layer's entities and
   relations, with key attributes.
4. `## 4. Migrations` — the layer's migration history/tooling and notable
   schema evolution steps.
5. `## 5. Access patterns` — who writes/reads this layer, through which
   modules; brief cross-references to the relevant capabilities/APIs.

## Rules
- Fields/types/constraints come from model/migration definitions only.
- Very wide tables: list all fields, but compress audit-style columns into
  one row ("+ 12 audit columns — see `path`").
- Non-relational stores: describe documents/keys/schemas in equivalent
  table form when possible.
</subpage>
