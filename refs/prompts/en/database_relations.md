# Task: inferred table relations (strict JSON)

You are a database reverse-engineering expert. Introspection found NO explicit
foreign keys, but column names can suggest relations. For the tables below
infer the MOST OBVIOUS many-to-one relations.

## Tables (evidence)

<table_batch>
{table_batch}
</table_batch>

## Instructions

- Justify each relation by a column name matching another table's primary
  key / table name (e.g. user_id → users.id). Add only high-confidence ones.
- Every relation is an INFERENCE (introspection did not confirm it); they will
  be marked as inferences in the documentation.
- Return ONLY a JSON array — no prose, no markdown fences:

[{"from": "<table>", "from_cols": ["<column>"], "to": "<table>", "to_cols": ["<column>"]}]

- Take table and column names EXACTLY from the evidence. An empty array `[]`
  is a valid answer when no confident relations exist.
- Write in {language_name} (technical terms in English).
