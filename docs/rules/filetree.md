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
|   `-- *.ipynb
|-- docs/
|   |-- rules/
|   |   `-- filetree.md
|   `-- <design>-spec.md
|-- skills/
|   |-- engineer/
|   `-- operations/
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

## Not in this repository

Do not add a top-level `scripts/`, `sim/`, `sw/`, `configs/`, `logs/`, `core/`,
`test/`, or `tools/`. Simulation entry points belong in `src/test/`, board
software in `src/runtime/`, and project-generating Tcl in `src/hw/vivado_tcl/`.

Do not add `vivado_projects/`, `results/`, a bitstream, or any Vivado project
directory. Regenerate them from `src/hw/vivado_tcl/`; bitstreams attach to a
GitHub Release.
