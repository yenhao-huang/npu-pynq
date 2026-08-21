# IC Design Repo Create State

Run ID: add-human-doc-governance-20260821
Instance: npu_repo_in_pynq/.codex/skills/custom/ic_design/ic-design-repo-create
Started: 2026-08-21T00:00:00+08:00
Scope: Add human-facing feature list, roadmap, weekly changelog, and mandatory
human confirmation before editing `docs/human/`.

Last updated: 2026-08-21T00:45:00+08:00

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | User explicitly requested the human docs, AGENTS guard, and skill workflow. | This request authorizes initial creation only. |
| 1. Inventory Sources and Products | completed | Read existing skill, templates, root `AGENTS.md`, and repository filetree; `docs/human/` is absent. | No source moves are required. |
| 2. Confirm Move Plan | completed | User specified the target documents and confirmation boundary. | `docs/human/feature-list.md` is the assumed human feature-list location. |
| 3. Create Directories and Move | completed | Added the human-doc rule and templates; updated skill workflow, directory/filetree rules, AGENTS template, root rules, and confirmed `docs/human/` files. | No existing source or generated output was moved. |
| 4. Install Ignore Rules | skipped | Not relevant to human documentation. |  |
| 5. Install CI and CD | skipped | Not relevant to human documentation. |  |
| 6. Verify Simulation Locally | skipped | Documentation/skill-only change. | Validate templates, links, layout, and rules instead. |
| 7. Handoff Summary | completed | Generic validator passed; 8 target files and 4 confirmation guards passed; skill-link audit passed; ISO week resolved to `2026-W34.md`; `git diff --check` passed. | Python UTF-8 mode was required on Windows for the UTF-8 skill content. |
