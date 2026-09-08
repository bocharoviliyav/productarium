# Task: generate the "{section_title}" wiki page

Repository: `{repo_url}` (`{repo_name}`)
Unit id: `{section_id}` — the ONLY page you write in this run.

## Repository brief (orientation; verify by reading files)

<repo_brief>
{repo_brief}
</repo_brief>

## The wiki this page belongs to

<sections_list>
{sections_list}
</sections_list>

Your page is "{section_title}". Cross-reference sibling sections by title
when relevant; do not write or repeat their content.

## Related units (cross-reference by title; do not duplicate their content)

<siblings_list>
{siblings_list}
</siblings_list>

When your page is a PARENT page, the list above are your SUBPAGES: link each
of them by title (the wiki navigates them as child pages). When your page is
one of those units, the list is your siblings: coordinate scope with them —
do not describe a sibling's material in depth.

## Router hints (advisory starting points, not restrictions)

<section_hints>
{section_hints}
</section_hints>

## Shared notes workspace (context reuse between units)

You are not the first agent on this repository. Previous units leave notes
for you, and you leave notes for the next ones:

- `notes_read(name)` — read a note from this run's workspace.
- `notes_write(name, content)` — save a note for later units.

Read BEFORE exploring: `repo_brief.md` (orientation) and the
`summary_*.md` notes of units related to yours (each is a short digest:
what the unit covered, key files, key decisions). Reading them saves you
from re-discovering the same files.

Write when you finish: `summary_{section_id}.md` — up to ~2000 chars:
what your page covers, the key files it is grounded in, decisions made
(e.g. chosen terminology, diagram scope). Keep it factual; never put
secrets in notes. A MISSING note simply means the unit ran without the
tools — explore on your own then.

## Section instruction (the content contract — follow it exactly)

<section_instruction>
{section_instruction}
</section_instruction>

## Reminders

- Explore the repository with the tools FIRST; base every claim on files you
  read. Hints are starting points, not evidence.
- Cite sources as inline-code paths (`src/api.py`, `src/api.py:42-58`).
- End the section with a compact provenance block (in the output language):
  `### Провенанс и проверка` — key sources, assumptions ("Допущения"),
  gaps ("Пробелы"), and a 1-2 sentence confidence summary.
- Your FINAL message must contain ONLY the finished section Markdown.
