# Task: document the database "{database_name}"

You are a database documentation expert. Document the database for engineers
based EXCLUSIVELY on the schema evidence collected from its introspection
tools (an MCP server connected to the live database).

## Masked connection string

<masked_connection>
{dsn_masked}
</masked_connection>

## Deterministic skeleton (ground truth — every schema/table listed)

<skeleton>
{skeleton}
</skeleton>

## Raw introspection evidence (schemas → tables → definitions)

<schema_evidence>
{schema_dump}
</schema_evidence>

## Instructions

Produce complete, well-structured Markdown documentation covering:

1. **Overview** — the database's apparent purpose and overall shape (schema
   count, table count, naming conventions).
2. **Schema layout** — the namespaces and what each appears to contain.
3. **Tables** — every introspected table with its columns, types, primary
   keys and foreign keys (from the definitions above).
4. **Relationships** — inferred FK graph between tables; mark inferred
   links explicitly as inferences. An `erDiagram` Mermaid block is welcome
   when the keys support it.
5. **Design notes** — indexes, constraints, defaults, and notable design
   decisions worth calling out.

## Reminders

- Do NOT invent tables, columns, or constraints that are not present in the
  evidence; mark inferences explicitly as inferences.
- Keep table/column identifiers exactly as the evidence spells them.
- Write the documentation in {language_name} (technical terms in English).
- Never include credentials or full connection strings; the masked DSN is
  the only connection reference allowed.
- Your FINAL message must contain ONLY the finished Markdown document.
