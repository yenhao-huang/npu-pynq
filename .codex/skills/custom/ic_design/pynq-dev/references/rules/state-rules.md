# State Rules

`STATE.md` is the execution record. For a new independent request, reset it from
`references/template/STATE.template.md`; resume existing state only when the user
asks to continue that run.

Allowed statuses are `pending`, `in_progress`, `completed`, `blocked`, and
`skipped`.

1. Fill Run ID, Instance, Started, Scope, OpenSpec change, and Last updated.
2. Mark a step `in_progress` before performing its edits or external actions.
3. Mark it `completed` only with concrete evidence: file paths, commands, exit
   status, test counts, report paths, or observed board output.
4. Use `blocked` for a required unavailable tool, license, board, network path,
   authority, or unresolved contract; state the exact condition and next action.
5. Use `skipped` only for a genuinely non-applicable optional step and record
   the impact-based rationale.
6. Keep OpenSpec task checkboxes synchronized immediately after each verified
   task. An incomplete checkbox cannot be overridden by `STATE.md` prose.
7. Preserve credentials, keys, tokens, and sensitive board configuration outside
   state. Do not paste full generated reports when a path and key metrics suffice.
8. Before handoff, record changed files, passed/not-applicable/blocked/failed
   gates, pre-existing user changes, and remaining risks.

Do not claim completion unless `STATE.md` and the selected OpenSpec change were
updated in the same run with objective evidence.
