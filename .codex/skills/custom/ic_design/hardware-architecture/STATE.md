# Hardware Architecture State

Run ID: rename-npu-accelerator-20261004T135140Z
Instance: .codex/skills/custom/ic_design/hardware-architecture
Started: 2026-10-04T13:51:40Z
Scope: Rename the RTL top to npu_accelerator and show PL → npu_accelerator → npu_matrix_core in the active diagram.
Last updated: 2026-10-04T13:55:35Z

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define scope | completed | User specified both the RTL top rename and the diagram hierarchy. | Preserve the existing visual style. |
| 1. Read source and rules | completed | Read repository AGENTS.md, filetree and simulation rules, active README SVG link, RTL tree, and Vivado references. | Existing runtime and test fixtures also use the top instance name. |
| 2. Edit diagram | completed | Added npu_accelerator boundary under PL and renamed the inner boundary npu_matrix_core. | SVG viewBox, cards, arrows, and palette remain as before. |
| 3. Validate rendering | completed | SVG XML parsed; Sharp rendered at 1600×930 and the image was visually inspected. | Both nested labels fit the existing composition. |
| 4. Handoff | completed | Report the updated SVG and RTL name, with RTL/Python/Vivado validation results. | No commit or push requested. |