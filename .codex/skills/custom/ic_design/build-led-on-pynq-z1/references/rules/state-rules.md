# State Rule

For a new independent maintenance run, reset `STATE.md` from
`references/template/STATE.template.md`, fill in Run ID, Instance, Started, and
Scope, then mark the active step `in_progress`.

Allowed statuses are `pending`, `in_progress`, `completed`, `blocked`, and
`skipped`. Record concrete evidence before marking a step complete. Never put
credentials in state.
