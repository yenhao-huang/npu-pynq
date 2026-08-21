# PYNQ Development State

Run ID: update-roadmap-20260821
Instance: `.codex/skills/custom/ic_design/pynq-dev`
Started: 2026-08-21T00:00:00+08:00
Scope: Update only `docs/human/roadmap.md` with the human-confirmed Phase 0–3
production NPU direction through a docs-only OpenSpec change.
OpenSpec change: `update-production-npu-roadmap`

Last updated: 2026-08-21T16:45:00+08:00

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | User confirmed the exact Phase 0, 1A, 1B, 1C, 2A, 2B, 2C, and 3 roadmap batch. | No dates, detailed architecture, or implementation edits. |
| 1. Read Context and Rules | completed | Read `AGENTS.md`, `docs/rules/filetree.md`, `docs/rules/human-docs.md`, current roadmap, and OpenSpec propose/apply skills. | Exact approved human-doc path is `docs/human/roadmap.md`. |
| 2. Prepare OpenSpec Change | completed | Created docs-only OpenSpec proposal, design, and two-task checklist with `skip_specs: true`. | No product behavior or spec capability changes. |
| 3. Update Roadmap | completed | Replaced the sole placeholder under `Planned milestones` with Phase 0, 1A, 1B, 1C, 2A, 2B, 2C, and 3 in the confirmed order. | Modified only `docs/human/roadmap.md`; descriptions remain phase-level. |
| 4. Validate | completed | Read roadmap back; phase audit passed `phase-count=8 order=0,1A,1B,1C,2A,2B,2C,3`; strict OpenSpec validation and `git diff --check` passed. | `feature-list.md` and `changelog/2026-W34.md` retain their earlier timestamps; this batch targeted only the roadmap. |
| 5. Handoff | completed | OpenSpec tasks 2/2 complete and final roadmap content prepared for handoff. | Change remains active; archive was not requested. |
| 6. Reorder by Priority | completed | Moved `Planned milestones` before `Confirmed direction`; read-back audit passed `headings=planned-before-direction phases=8`; strict OpenSpec validation and `git diff --check` passed. | Section content remained unchanged. |
| 7. Publish Roadmap | completed | Claimed parent issue `#1`; created Phase issues `#2` through `#9`; pushed branch `npu/npu-1-codex-a`; opened draft PR `#10` targeting `dev`. | Skill validation, both strict OpenSpec validations, and `git diff --check` passed; GNU Make is unavailable on this Windows host, so RTL lint/simulation remains a documented validation blocker. |
