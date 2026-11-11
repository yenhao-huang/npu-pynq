# Hardware Architecture State

Run ID: issue165-a
Instance: npu/issue165-a
Started: 2026-11-11
Scope: Redesign requested by the user: shrink csr, draw scalar_core as a matrix_core-style panel with its IF/ID/EX stages, controller and AXI4 master.
Last updated: 2026-11-11

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define scope | completed | Issue #165; user approved the plan in chat | Visual redesign explicitly requested, so layout preservation is waived |
| 1. Read source and rules | completed | README.md:117 links docs/assets/npu-hardware-architecture.svg; RTL dirs csr/, scalar_core/, matrix_core/{controller,matrix_datapath/{memory,systolic_array}} | Stage names from scalar_core.sv header |
| 2. Edit diagram | completed | docs/assets/npu-hardware-architecture.svg: viewBox 1600x930 -> 2020x930; csr 90x50 label-only; scalar_core panel with controller, IF/ID/EX, AXI4 master; matrix_core internals moved, sizes kept except controller/memory height 160 | csr status paths follow npu_accelerator.sv (isa_* and status_*) |
| 3. Validate rendering | completed | xml.etree parse OK; Chrome headless render at 2020x930 to build/arch/new.png, compared with build/arch/old.png; no label overlaps a border or arrowhead, all card text fits | The "control" label on the PS border is unchanged from the original |
| 4. Handoff | completed | Branch npu/issue165-a, PR to dev | README link unchanged |
