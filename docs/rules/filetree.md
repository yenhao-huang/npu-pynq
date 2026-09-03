# Filetree Rules

Keep this repository in:

```text
npu_repo_in_pynq/
|-- AGENTS.md
|-- CLAUDE.md
|-- README.md
|-- .gitignore
|-- .github/
|   |-- cd/
|   |   `-- *.ps1              automated deployment and acceptance scripts
|   `-- workflows/
|       |-- cd.yml
|       `-- ci.yml
|-- .codex/
|   `-- skills/
|       |-- dev/                 shared development workflows
|       |-- deploy/              installation and deployment workflows
|       `-- custom/
|           `-- ic_design/       repository-specific FPGA/NPU workflows
|-- src/
|   |-- hw/
|   |   |-- rtl/
|   |   |   `-- <design>/
|   |   |       `-- *.sv
|   |   |-- tb/
|   |   |   `-- <design>/
|   |   |       `-- tb_*.sv
|   |   |-- constraints/
|   |   |   `-- *.xdc
|   |   `-- vivado_tcl/
|   |       `-- <design>/
|   |           `-- build_overlay.tcl
|   |-- test/
|   |   |-- Makefile
|   |   |-- model/
|   |   |   `-- *.py
|   |   |-- cocotb/
|   |   |   `-- test_*.py
|   |   |-- vectors/
|   |   `-- waves/
|   |-- export/
|   |   `-- *.py
|   `-- runtime/
|       `-- *.py
|-- examples/
|   `-- <example>/
|       |-- README.md
|       |-- *.ipynb
|       |-- package_example.py
|       |-- run_on_board.py
|       |-- deploy_release.ps1
|       |-- model/
|       |   `-- .gitkeep
|       |-- scripts/
|       |   `-- *.py
|       |-- runtime/
|       |   `-- *.py
|       `-- tests/
|           `-- test_*.py
|-- docs/
|   |-- rules/
|   |   |-- index.md
|   |   |-- environment.md
|   |   |-- generated-artifacts.md
|   |   |-- human-docs.md
|   |   |-- simulation.md
|   |   |-- ci-cd.md
|   |   |-- filetree.md
|   |   `-- git/
|   |       |-- branch.md
|   |       |-- commit.md
|   |       |-- issues.md
|   |       `-- pull-request.md
|   |-- human/
|   |   |-- feature-list.md
|   |   |-- roadmap.md
|   |   `-- changelog/
|   |       `-- <YYYY-Www>.md
|   `-- <design>-spec.md
|-- openspec/
|   |-- changes/
|   `-- specs/
`-- mount/
```

## What each directory is for

`src/hw/` is the NPU itself: everything Vivado reads. `rtl/` is synthesized;
`tb/` never is. That boundary is hard, so the two stay separate directories.

`src/test/` verifies that the hardware computes the right answer. `model/` is
the numpy golden reference, `cocotb/` are the Python tests that compare RTL
against it, and `Makefile` is what CI invokes.

`src/export/` turns a trained model into whatever format the NPU executes.

`src/runtime/` loads an overlay on the board and runs an exported model on it.

`examples/` consumes the three above. Nothing under `src/` may import from it.
An example owns its application-specific runtime, notebooks, package builder,
board acceptance entry point, deployment wrapper, and focused host tests. The
package builder may copy an explicit allowlist of shared `src/runtime/` modules
into generated deploy output, but those copies are never committed.
Every user-facing example includes an output-free `.ipynb` demo. The notebook
is the canonical human validation entry point: its README may prepare and
deploy inputs, but must ultimately direct the user to the notebook. CLI board
entry points support the notebook and automation; they do not replace the
human demo. Download and conversion commands belong under its `scripts/`;
generated checkpoints, converted model packages, corpora, and model evidence
go under its `model/` workspace and remain ignored except for `.gitkeep`.
The canonical generated-data path is `examples/<example>/model/`.

`.github/cd/` owns non-interactive deployment and acceptance scripts used by
continuous delivery. Example-local `deploy_release.ps1` files only copy a
release for later human validation; they must not execute acceptance, request
`sudo`, claim a physical PASS, or collect evidence.

`.codex/skills/` contains repository-local Codex skills and is the only allowed
top-level location for them. Classify reusable development workflows under
`dev/`, environment setup and delivery workflows under `deploy/`, and
repository-specific IC design workflows under `custom/ic_design/`. Do not add
a top-level `skills/` directory or place IC design skills directly under
`.codex/skills/`.

`openspec/` contains change proposals and specifications used by the
development workflow. Keep planning artifacts here, separate from product
source under `src/`.

`docs/rules/` contains repository-wide rules. It is the stable authority for
contributors; skills may link to these files but must not be the only location
of Git, environment, CI, simulation, or generated-artifact rules.

`docs/human/` contains the human-owned feature list, roadmap, and weekly
changelog. Agents may read it, but every mutation requires explicit human
confirmation for the exact proposed batch under `docs/rules/human-docs.md`.

`mount/` is empty in a clean checkout. It receives build products staged for
the board and nothing is authored there.

## Rules

One directory per design under `src/hw/rtl/`, `src/hw/tb/`, and
`src/hw/vivado_tcl/`. The directory name is the design name and must match
across all three.

A testbench is named `tb_<module>.sv` and lives in `src/hw/tb/<design>/`.
`make sim` discovers tests by that pattern, so a testbench outside it never
runs in CI.

`src/test/model/` defines the numeric contract: quantization, rounding
direction, saturation bounds, accumulator width and overflow behaviour. When
`src/export/` starts depending on it, promote it to `src/model/` rather than
letting production code import from a test directory.

`src/test/waves/` and `src/test/build/` are generated. Only
`src/test/waves/.gitkeep` is tracked.

Each skill is a self-contained directory rooted by `SKILL.md`; its supporting
material belongs under that skill's `references/`. When a skill changes
category, move the complete directory and update all repository-local path
references in the same change.

Do not create, edit, append, format, rename, move, or delete anything under
`docs/human/` without explicit human confirmation. Approval for code, an issue,
a pull request, merge, or release does not authorize a human-document update.

Before adding any new top-level directory, update this file in the same change
with the directory's purpose, allowed contents, and validation expectations.
Do not create a directory that duplicates an existing role. Changes beneath
`.codex/skills/` must preserve the `dev/`, `deploy/`, and `custom/ic_design/`
classification contract.

## Not in this repository

Do not add a top-level `skills/`, `scripts/`, `sim/`, `sw/`, `configs/`,
`logs/`, `core/`, `test/`, or `tools/`. Simulation entry points belong in
`src/test/`, board software in `src/runtime/`, project-generating Tcl in
`src/hw/vivado_tcl/`, and agent skills in `.codex/skills/`.

Do not add `vivado_projects/`, `results/`, a bitstream, or any Vivado project
directory. Regenerate them from `src/hw/vivado_tcl/`; bitstreams attach to a
GitHub Release.
