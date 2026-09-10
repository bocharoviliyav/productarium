# Task: descriptions of the "{category_title}" category objects (strict JSON)

You are a database reverse-engineering expert. For every object of the category
below write a short description (1–2 sentences), based ONLY on the provided
evidence: definition/source, metadata and product context.

## Category objects (evidence)

<objects>
{objects}
</objects>

## Product context (supplementary, NOT schema evidence)

<product_context>
{product_context}
</product_context>

## Instructions

- Describe the object's PURPOSE: what a view selects, when a trigger fires,
  what a procedure/function does, what a sequence yields, what a type describes.
- Mark inferences ("judging by the name/source …") explicitly.
- Write in {language_name} (technical terms in English).
- Return ONLY a JSON array — no prose, no markdown fences:

[{"name": "<object name exactly as in the list>", "purpose": "<1-2 sentences>"}]

- Take object names EXACTLY from the list. Do not invent objects absent from it.
