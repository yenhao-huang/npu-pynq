# Filetree Rules

Keep this repository in:

```text
npu_repo_in_pynq/
|-- AGENTS.md
|-- CLAUDE.md
|-- README.md
|-- .gitignore
|-- .github/
|   `-- workflows/
|       |-- build.yml
|       `-- ci.yml
|-- src/
|   |-- rtl/
|   |   `-- <design>/
|   |       `-- *.sv
|   |-- tb/
|   |   `-- <design>/
|   |       `-- tb_*.sv
|   |-- constraints/
|   |   `-- *.xdc
|   `-- vivado_tcl/
|       `-- <design>/
|           `-- build_overlay.tcl
|-- sim/
|   |-- Makefile
|   |-- cocotb/
|   |   `-- test_*.py
|   `-- waves/
|-- sw/
|   `-- <design>/
|       |-- *.py
|       `-- test_*.py
|-- docs/
|   |-- rules/
|   |   `-- filetree.md
|   `-- <design>-spec.md
|-- skills/
|   |-- engineer/
|   `-- operations/
`-- mount/
```

## Rules

One directory per design under `src/rtl/`, `src/tb/`, `src/vivado_tcl/`, and
`sw/`. The directory name is the design name and must match across all four.

A testbench is named `tb_<module>.sv` and lives in `src/tb/<design>/`. `sim`
discovers tests by that pattern, so a testbench outside it never runs in CI.

`mount/` is empty in a clean checkout. It receives build products staged for the
board and nothing is authored there.

`sim/waves/` and `sim/build/` are generated. Only `sim/waves/.gitkeep` is
tracked.

## Not in this repository

Do not add a top-level `scripts/`, `configs/`, `logs/`, `core/`, `test/`, or
`tools/`. Simulation entry points belong in `sim/`, host software in `sw/`, and
project-generating Tcl in `src/vivado_tcl/`.

Do not add `vivado_projects/`, `results/`, a bitstream, or any Vivado project
directory. Regenerate them from `src/vivado_tcl/`; bitstreams attach to a
GitHub Release.
