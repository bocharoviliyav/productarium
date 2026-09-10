# Task: the overview page of the database "{database_name}"

You are a database reverse-engineering expert. Write the OVERVIEW PAGE of the
database for engineers, based EXCLUSIVELY on the schema evidence collected
from its introspection tools (an MCP server connected to the live database).
Table structures, indexes and relationships are rendered as separate
deterministic pages — your page covers the GENERAL level.

## Masked connection string

<masked_connection>
{dsn_masked}
</masked_connection>

## Deterministic skeleton (ground truth — every schema/object listed)

<skeleton>
{skeleton}
</skeleton>

## Raw introspection evidence (schemas → tables → definitions → relations)

<schema_evidence>
{schema_dump}
</schema_evidence>

## Product context (supplementary, NOT schema evidence)

<product_context>
{product_context}
</product_context>

## Instructions

Write Markdown with exactly three sections:

1. **Overview** — the database's apparent purpose and overall shape: domain,
   role in the product, schema and object counts, naming conventions,
   workload character (OLTP/OLAP/mixed) inferred from table and index shape.
2. **Schema layout** — the namespaces and what each appears to contain (by
   the tables/views/routines of each schema). For a single schema, logical
   table groups instead of schemas.
3. **Design notes** — design decisions worth calling out: key styles
   (surrogate/natural, GUID/sequences), indexing policy, denormalization,
   audit columns, partitioning, naming conventions; what to watch when the
   schema evolves.

## Reminders

- Do NOT invent tables, columns, or constraints that are not present in the
  evidence; mark inferences explicitly as inferences (inferred from …).
- Product context is a supplementary source of meaning; database
  identifiers are grounded by introspection ONLY.
- Keep table/column identifiers exactly as the evidence spells them.
- Do not enumerate every table — this page is the general level; detailed
  structures live on subpages.
- Write in {language_name} (technical terms in English).
- Never include credentials or full connection strings; the masked DSN is
  the only connection reference allowed.
- Your FINAL message must contain ONLY the finished Markdown document.
