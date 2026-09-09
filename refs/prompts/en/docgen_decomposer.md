# Task: decompose wiki sections into subpages

You are a documentation planner. The repository brief below describes a code
repository. Three wiki sections are written as a PARENT page plus SUBPAGES,
and your job is to plan those subpages: what units each section splits into.

<repo_brief>
{repo_brief}
</repo_brief>

<wiki_sections>
{sections_list}
</wiki_sections>

<section_hints>
{section_hints}
</section_hints>

## What to plan

1. `"functional"` — one subpage per ARCHITECTURAL CAPABILITY of the system
   (a coherent user- or business-facing functionality: "User authentication",
   "Order processing", "Notifications", "Report export"). Capabilities must
   be visible in the code (routers, handlers, jobs, CLI commands, modules).
2. `"technical"` — one subpage per API surface unit and per notable technical
   reference:
   - `kind: "endpoint"` — one REST/GraphQL/RPC endpoint or a tightly-coupled
     group of endpoints on the same resource (e.g. "Products CRUD API");
   - `kind: "job"` — a background/scheduled/queue worker or long-running
     task;
   - `kind: "integration"` — an outbound integration with an external system;
   - `kind: "reference"` — a reference unit not tied to one endpoint: key
     class/module responsibilities, error code reference, client SDK.
3. `"datamodel"` — one subpage per DATA LAYER / store: a relational schema,
   a document store, a cache, a vector store, a file-based store, a message
   schema set (e.g. "Postgres schema", "Redis cache", "S3 artifacts").

## Output contract

Respond with ONLY a JSON object (no prose, no code fence):

```
{
  "functional": [
    {"slug": "auth", "title": "Аутентификация пользователей", "focus": "login flow, sessions, tokens; start at api/auth.py"},
    ...
  ],
  "technical": [
    {"slug": "products-api", "title": "Products API", "kind": "endpoint", "focus": "CRUD /api/products; routers/products.py"},
    ...
  ],
  "datamodel": [
    {"slug": "postgres", "title": "Postgres schema", "kind": "layer", "focus": "ORM models in api/models.py, pgvector chunks"},
    ...
  ]
}
```

Rules:
- Every array item: `slug` (short ascii kebab/id, lowercase), `title` (human
  name, may be in the wiki's language), `focus` (one sentence: what the
  subpage covers + where to start in the repo, max ~40 words). `technical`
  and `datamodel` items additionally carry `kind` (one of the values above).
- Ground every unit in the brief: do not invent capabilities, endpoints,
  jobs or stores that the tree/hints do not suggest. When genuinely unsure,
  prefer fewer, broader units.
- Coverage over quantity: the chosen units must COVER the section's material
  without large overlaps between them.
- Size bounds: at most 10 `functional` units, 12 `technical` units, 6
  `datamodel` units. Inside a bound, keep every unit meaningful (no
  catch-all "Misc").
- `datamodel`: when the repo has a single small store, one unit is fine; when
  it has no persistence at all, return an empty array for `datamodel`.
- The `slug` is advisory (the pipeline derives the page id from the title);
  uniqueness matters, wording does not.
