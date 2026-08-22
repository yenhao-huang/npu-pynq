# Send to PYNQ Board State

Run ID: consolidate-pynq-skills-20260807
Instance: npu_repo_in_pynq/.codex/skills/deploy/send-to-pynq-board
Started: 2026-08-07
Scope: Consolidate and validate all project-local PYNQ skills under npu_repo_in_pynq/skills.

Last updated: 2026-08-07

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | User requested all project skills be organized under `npu_repo_in_pynq/skills`. | Classify this transfer/deployment workflow as `operations`. |
| 1. Read Relevant Context | completed | Read skill-create rules and found this as the only project skill outside `npu_repo_in_pynq/skills`. | Existing strict layout will be preserved. |
| 2. Execute Workflow | completed | Moved from the legacy project-local skill tree to `.codex/skills/deploy`; updated command, filetree, and state-template paths. | The old skill path was retired. |
| 3. Validate Result | completed | UTF-8 generic validator, required layout, PowerShell parser, hello_pynq dry run, canonical entrypoint, forbidden-directory, and stale-path checks passed. | Dry run performed no network transfer. |
| 4. Handoff Summary | completed | Canonical skill path is `npu_repo_in_pynq/.codex/skills/deploy/send-to-pynq-board`. | Ready for repository-local use. |
