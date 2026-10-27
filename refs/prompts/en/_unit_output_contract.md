<output_contract>
Binding rules for your FINAL message (intermediate working steps are unaffected):

- Start/end markers: the final message must START with the first element of
  the required deliverable (its opening `#` heading for a Markdown page; the
  first `[` or `{` for a JSON deliverable) and END with its last element —
  close every code fence and JSON bracket you open; a truncated deliverable
  is a failed deliverable.
- No prologue, no epilogue, no explanations: nothing before the first
  element and nothing after the last line — no "Here is…", no wrap-ups, no
  apologies, no questions.
- No tool calls in the final message: never emit `<tool_call>…</tool_call>`
  blocks, tool dialogs or logs, or special tokens like
  `<|im_start|>`/`<|im_end|>` — the final message reads as if tools did not
  exist. Need a tool? Call it in an intermediate step.
- Output language: write all prose in the output language this task
  requires; keep technical terms, code identifiers and file names in
  English.
- Heading format (Markdown deliverables): ATX headings (`#`), levels and
  titles exactly as the task's content contract lists them.
</output_contract>
