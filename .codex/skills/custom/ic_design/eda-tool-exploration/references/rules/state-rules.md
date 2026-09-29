# State rules

Reset from references/template/STATE.template.md for a new run; preserve state
when resuming. Record run ID, date, issue, branch, scope and exact evidence paths.
Use pending, in_progress, completed, blocked or skipped. Mark a step in_progress
before starting it. Mark completed only after recording concrete evidence in the
same turn. Reopen affected steps when code or inputs change. Before stopping,
record the next command, blockers and remaining usage; never claim pending gates
passed. Keep credentials and raw private logs out of this file.
