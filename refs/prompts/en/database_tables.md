# Task: short table descriptions (batch, strict JSON)

You are a database reverse-engineering expert. For every table in the batch
below write a short description (1–2 sentences), based ONLY on the provided
evidence: columns, keys, foreign relations, schema comment and product context.

## Batch tables (evidence)

<table_batch>
{table_batch}
</table_batch>

## Product context (supplementary, NOT schema evidence)

<product_context>
{product_context}
</product_context>

## Instructions

- Describe the table's PURPOSE in the domain (what it stores, what role it plays).
- Add significant structure details to `notes` (usually empty): audit columns,
  soft deletes, JSONB sparseness, apparent partitioning.
- In `related` list only tables from the batch/skeleton with an explicit or
  obvious relation (existing FKs — always; guessed ones — only at high
  confidence, e.g. user_id → users).
- Write in {language_name} (technical terms in English).
- Return ONLY a JSON array — no prose, no markdown fences:

[{"name": "<table name exactly as in the batch>", "purpose": "<1-2 sentences>", "notes": "<optional>", "related": ["<table name>", "..."]}]

- Take table names EXACTLY from the batch. Do not invent tables absent from it.
