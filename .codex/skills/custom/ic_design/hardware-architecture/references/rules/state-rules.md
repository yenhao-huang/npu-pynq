# State Rules

`STATE.md` records the current run. For each new request, copy
`references/template/STATE.template.md` to `STATE.md`, fill the run fields,
and mark a step `in_progress` before changing files. Use only `pending`,
`in_progress`, `completed`, `blocked`, or `skipped`.

Mark a step `completed` only after recording concrete evidence: the asset
path, directory inventory, XML parse result, rendered preview inspection, or
the exact reason a check was unavailable. Update the record when later edits
affect earlier evidence. Do not include tokens or other credentials.
