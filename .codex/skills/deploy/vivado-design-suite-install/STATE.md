# Vivado Design Suite Install Skill State

Run ID: consolidate-pynq-skills-20260807
Instance: npu_repo_in_pynq/.codex/skills/deploy/vivado-design-suite-install
Started: 2026-08-07
Scope: Consolidate and validate all project-local PYNQ skills under npu_repo_in_pynq/skills.

Last updated: 2026-08-07

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | User requested all project skills be organized under `npu_repo_in_pynq/skills`. | Classify this installation workflow as `operations`. |
| 1. Read Relevant Context | completed | Read skill-create category, filetree, environment, and state rules; inspected all project skill entrypoints. | Existing skill is under `custom` while its own filetree says `operations`. |
| 2. Execute Workflow | completed | Moved to `.codex/skills/deploy`; normalized the state template to the shared five-step schema. | Domain references and SKILL.md were preserved. |
| 3. Validate Result | completed | UTF-8 generic validator passed; required layout, canonical entrypoint, forbidden-directory, and stale-path checks passed. | Validation used UTF-8 mode because the Windows default locale is CP950. |
| 4. Handoff Summary | completed | Canonical skill path is `npu_repo_in_pynq/.codex/skills/deploy/vivado-design-suite-install`. | Ready for repository-local use. |
