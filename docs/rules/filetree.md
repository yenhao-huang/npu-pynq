# Filetree Rules

Keep this repository in:

```text
npu_repo_in_pynq/
|-- AGENTS.md
|-- CLAUDE.md
|-- README.md
|-- LICENSE
|-- .gitignore
|-- changelog/
|   `-- vMAJOR.MINOR.PATCH.md
|-- .github/
|   |-- cd/
|   |   |-- *.ps1             automated deployment and acceptance scripts
|   |   `-- *.py              release version selection for the workflows
|   `-- workflows/
|       |-- cd.yml
|       |-- ci.yml
|       |-- exp-board.yml       issue-119 experiment runs on the self-hosted runners
|       `-- release-publish.yml
|-- .codex/
|   `-- skills/
|       |-- dev/                 shared development workflows
|       |-- deploy/              installation and deployment workflows
|       |   `-- <skill>/
|       |       |-- SKILL.md
|       |       `-- references/
|       |           |-- rules/       environment and state rules
|       |           `-- scripts/     *.ps1 helpers the skill runs
|       `-- custom/
|           `-- ic_design/       repository-specific FPGA/NPU workflows
|-- src/
|   |-- hw/
|   |   |-- rtl/
|   |   |   `-- <design>/
|   |   |       |-- *.sv
|   |   |       `-- <module>/
|   |   |           |-- *.sv
|   |   |           `-- <child-module>/
|   |   |               `-- *.sv
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
|   |   |-- onnx_mlir_acceptance.py  independent ONNX fixtures and compiled host/board checks
|   |   |-- matrix_acceptance.py     physical full-tile, edge-tile and repeat checks
|   |   |-- model/
|   |   |   `-- *.py
|   |   |-- cocotb/
|   |   |   `-- test_*.py
|   |   |-- vectors/
|   |   `-- waves/
|   |-- vendors/                 one folder per accelerator; imported by export and runtime, imports neither
|   |   `-- npu_pynq/
|   |       |-- isa/             NPU instruction set: spec, layout, simulator; imports nothing
|   |       |-- compiler/        passes, weight packing, CPU target parameters; imports isa/ only
|   |       |   `-- mlir/        out-of-tree MLIR project: npu dialect, passes, npu-opt
|   |       `-- driver/          ISA drivers (Python and C), buffers, placement, ABI, DMA; imports isa/ only
|   |           `-- overlay/     build.py writes the BIT/HWH manifest; verify.py checks it before loading
|   |-- export/                  host-side model -> package (CPU code + ISA programs); needs MLIR/LLVM
|   |   |-- cli.py               export() (front end + compile) and the command line
|   |   |-- compile.py           compile_graph(): ModelGraph -> package (partition, pack, encode, codegen)
|   |   |-- pipeline.py          front end: load -> quantize -> to_mlir, and stage results on disk
|   |   |-- ir/                  data between stages: options, SourceModel, QuantizedModel, ModelGraph
|   |   |-- load/                stage 1: read sources unchanged (safetensors, ONNX, pins); format detection
|   |   |-- quantize/            stage 2: numerics (int8.py), weight plans, preserve recipe, AWQ
|   |   |-- model_arch/          stage 3: one Architecture per model; a model with source pins or
|   |   |                        download scripts is a directory (smollm/, resnet18/), others one module
|   |   |-- delivery/            standalone matrix and image deployment package builders
|   |   |-- compilers/mlir/      target-independent MLIR builder, operator library, LLVM toolchain
|   |   |-- compilers/onnx_mlir/ pinned ONNX import, MLIR parser adapter and verified CPU conversions
|   |   `-- targets.py           target registry: wires each src/vendors/<vendor>/compiler to the toolchain
|   |-- jit/                     optional, self-contained just-in-time stack
|   |   |-- model/               numeric and ResNet model contracts
|   |   |-- export/              legacy ResNet package exporter
|   |   |-- runtime/             interpreter, local DMA/overlay, acceptance
|   |   |-- examples/resnet18/    original JIT notebook and delivery helpers
|   |   |-- tests/               JIT-specific regression tests
|   |   |-- reference/           independent quantized graph reference
|   |   `-- delivery/            frozen optional JIT delivery helpers
|   `-- runtime/                 board-side execution; never imports src/export
|       |-- core/                generation loop, package loading, tokenizer, sampling
|       |-- cpu/                 optional host ONNX Runtime reference operators
|       |-- image_cli.py         compiled image inference and delivery verification
|       `-- cli.py               run and reference command line
|-- examples/
|   `-- <example>/              see docs/rules/create-example.md
|       |-- README.md            WSL deployment manual
|       |-- README.macos.md      macOS deployment manual
|       `-- <name>.ipynb
|-- docs/
|   |-- assets/
|   |   `-- *.png, *.svg       versioned images embedded by documentation
|   |-- rules/
|   |   |-- index.md
|   |   |-- environment.md
|   |   |-- generated-artifacts.md
|   |   |-- human-docs.md
|   |   |-- simulation.md
|   |   |-- ci-cd.md
|   |   |-- filetree.md
|   |   |-- create-example.md
|   |   `-- git/
|   |       |-- branch.md
|   |       |-- changelog.md
|   |       |-- commit.md
|   |       |-- issues.md
|   |       `-- pull-request.md
|   |-- human/
|   |   |-- feature-list.md
|   |   |-- roadmap.md
|   |   `-- changelog/
|   |       `-- <YYYY-Www>.md
|   |-- plans/
|   |   `-- <YYYY>-<MMDD>-<NNN>-<topic>.md
|   |-- acceptance/
|   |   `-- reproduce.md        how to reproduce a delivered capability
|   |-- <toolchain>/
|   |   `-- README.md           developer toolchain documentation
|   `-- <design>-spec.md
|-- tools/
|   `-- ic/                     agent-facing IC design toolchain
|       |-- pyproject.toml
|       |-- ic_core/            one folder per tool category
|       |   `-- tools/<category>/
|       |       |-- __init__.py    the category contract
|       |       `-- <backend>.py   one file per backend
|       |-- ic_daemon/          run records and job lifecycle
|       |-- ic_cli/             the `ic` command line
|       |-- ic_mcp/             MCP adapter
|       |-- integrations/pi/    pi extension and its verifier
|       |-- npm/                npm MCP bootstrap, pinned downloads and tests
|       |-- package.json        npm package and launcher entry point
|       |-- package-lock.json   locked launcher dependencies
|       |-- README.md           package overview and quick start
|       |-- docs/manual/        installation instructions and agent demos
|       |-- LICENSE             license included in the npm distribution
|       `-- tests/
|-- .pi/
|   |-- settings.json
|   `-- extensions/
|       `-- *.ts
|-- .mcp.json
|-- exp/
|   |-- 0911_performance_optimization/  compact hardware and board results
|   |-- 1007_sw_stack/          software-stack goal (#119): exp-board request, export
|   |                           and board scripts, host evaluation scripts and results
|   `-- ab_test/                IC-tool A/B report and experiment source
|-- openspec/
|   |-- changes/
|   `-- specs/
|-- .ic/                        run store, not tracked
`-- dist/
```

The ONNX-MLIR integration under `src/export/compilers/onnx_mlir/` owns pinned
compiler invocation, inventory, the C++ generic-IR parser adapter, and verified
CPU conversions. It imports neither model frontends nor vendor modules.
Production SmolLM export consumes imported IR before the existing CPU/NPU
partition. Tool builds and generated IR stay ignored under `build/`;
instructions and evidence live under `docs/plans/onnx-mlir-incoporate/`.
Native compiler overrides are explicitly unverified.

## What each directory is for

`src/hw/` is the NPU itself: everything Vivado reads. `rtl/` is synthesized;
`tb/` never is. That boundary is hard, so the two stay separate directories.

`src/test/` verifies that the hardware computes the right answer. `model/` is
the numpy golden reference, `cocotb/` are the Python tests that compare RTL
against it, and `Makefile` is what CI invokes.

`src/vendors/` holds everything specific to one accelerator, one folder per
vendor, as a layer below both `src/export/` and `src/runtime/`: both import
it, and it imports neither. Inside a vendor, `isa/` is the specification and
imports nothing; `compiler/` (the export side) and `driver/` (the board side)
each import `isa/` but never each other, so the board package ships `isa/` and
`driver/` without the compiler. `python -m unittest
src.test.tests.test_vendor_boundary` enforces these edges. A new accelerator is
one new folder here plus one registry line in `src/export/targets.py`.

`src/vendors/npu_pynq/isa/` is the single definition of the NPU instruction
set: `isa.py` encodes and decodes it, `layout.py` fixes how a tiled GEMM maps
onto DDR, and `sim.py` is the bit-accurate simulator the RTL, runtimes and
compiler are checked against. RTL opcodes must match it;
`src/test/tests/test_isa.py` enforces that. `compiler/` holds the `npu-opt`
MLIR project and the partition and lower passes, NPU weight tiling and the
Cortex-A9 target parameters; the LLVM toolchain is passed in by
`src/export/targets.py`. `driver/` drives the NPU on the board: the Python
and C ISA drivers, CMA buffers, program placement, the ctypes ABI, the legacy
ABI 1.0 DMA runtime in `dma.py`, and `overlay/`, which writes the overlay
manifest after Vivado and verifies it before the board loads the bitstream.

`src/export/` turns a model into a package on the host. The front end runs
three stages (`pipeline.py`): `load/` reads the source without changing any
number, `quantize/` turns it into final parameters, and the model's
architecture in `model_arch/` builds a `ModelGraph` from those
parameters only (`to_mlir`), using the operators in `compilers/mlir/ops/`.
`npu-opt` (built from
`src/vendors/npu_pynq/compiler/mlir/` with CMake against LLVM 22) partitions
INT8 contractions onto the NPU and encodes them as ISA programs; upstream MLIR
and LLVM compile the rest for the Cortex-A9 or the host. A new model adds one
module (or directory, when it brings source pins) under `model_arch/` and one
registry line in `model_arch/__init__.py`;
a new target adds one vendor under `src/vendors/` and one line in
`targets.py`. Imports point down: `cli.py`, `compile.py` and `pipeline.py`
may import anything; `model_arch/` imports `ir/`, `load/`, `quantize/` and `compilers/`;
`quantize/` imports `ir/` and never `compilers/` or `load/`; `load/` imports
`ir/` and, for the ONNX import, `compilers/onnx_mlir/`, never `quantize/`;
`compilers/` never imports `quantize/`, `load/` or `model_arch/`; and a model
in `model_arch/` imports only `base.py` and its own directory, never another
model. `python -m unittest
src.test.tests.test_export_layering` enforces these edges.
`src/vendors/npu_pynq/driver/npu_rt.c` is linked into every compiled model; `src/runtime/core/package.py` loads a
package on the host (simulated NPU) or the board.

`src/jit/` owns the optional just-in-time quantized graph interpreter, its
exporter, numeric contracts, examples, references, focused tests and delivery
helpers. It imports no other source packages. Active source, tests, public
examples and delivery workflows must not import it: removing `src/jit` must
leave AOT export and inference functional. Check both boundaries with
`python -m unittest src.test.tests.test_jit_boundary` and execute active tests
with JIT physically absent. The hardware numeric reference remains independently
available under `src/test/model/numeric.py`; its contract is unchanged.

`src/runtime/` runs packages on the board. `core/matrix.py` tiles logical matrix
jobs through the public vendor DMA runtime; `core/delivery.py` verifies image
delivery hashes without compiler dependencies. `image_cli.py` runs compiled image
inference and delivery verification. `core/` is the inference path:
`engine.py` schedules prefill, decode and sampling, and `package.py` loads a
compiled package and calls its entry points, driving the NPU through
`src/vendors/npu_pynq/driver/`; `cpu/` holds
optional ONNX Runtime operators for independent host references. The Cortex-A9
executes floating operators from its compiled model library. The ONNX golden
reference that interprets the source graph without the compiler lives in
`src/test/model/onnx_reference.py`.

`examples/` demonstrates existing source functionality. Its tracked file list,
platform deployment manuals, notebook requirements and generated-data locations
are defined in [create-example.md](create-example.md).

`.github/cd/` owns non-interactive deployment and acceptance scripts used by
continuous delivery.

`.codex/skills/` contains repository-local Codex skills and is the only allowed
top-level location for them. Classify reusable development workflows under
`dev/`, environment setup and delivery workflows under `deploy/`, and
repository-specific IC design workflows under `custom/ic_design/`. Do not add
a top-level `skills/` directory or place IC design skills directly under
`.codex/skills/`.

`tools/` holds developer and agent toolchains that act on this repository but
are not part of the product. `tools/ic/` is the IC design toolchain: `ic_core/`
holds one folder per tool category, each owning its schema and containing one
file per backend; `ic_daemon/`, `ic_cli/`, `ic_mcp/` and `integrations/pi/` are
thin clients over it and contain no tool knowledge. Adding a backend touches
one file and adding a category one folder; no client changes either way. See
[../ic-design-tools/README.md](../ic-design-tools/README.md).

`.pi/` and `.mcp.json` attach those tools to specific agents: `.pi/settings.json`
selects the native pi package and `.mcp.json` registers the MCP server.
They are configuration only. The skill that tells a model when to use the
tools lives with the other skills, under
`.codex/skills/custom/ic_design/ic-design-tools/`, so one copy serves pi,
Claude Code and Codex alike.

`docs/plans/` holds architecture and implementation plans.
`docs/goals/ic-design-tools/` holds the toolchain overview and its `acceptance/`
reproduction procedures, including commands, recorded outputs and validation
limitations. Package installation and demo instructions live under
`tools/ic/docs/manual/` and ship with the npm package.
`docs/goals/sw_optimal/` holds the software-stack goal (#119): the goal as
given, `Results.md` (key results), `Reproduce.md`, and `details/` (hardware
ISA, compiler, inference framework, quantization, board runs).
`docs/goals/widen-input-width/` holds the packed operand stream's simulation
report, design walkthrough, reproduction steps and recorded metrics (#93).
`docs/goal/` holds one prompt file per hardware goal. Each file records the goal
as given, the implementation, and the measured experiment that accepts it.

`.ic/` is the run store written by `tools/ic`. It is machine-local and not
tracked: `meta.json` records are kilobytes but the artifacts beside them are
gigabytes. Never commit it.

`openspec/` contains change proposals and specifications used by the
development workflow. Keep planning artifacts here, separate from product
source under `src/`.

`changelog/` contains commit-bounded release baselines and the change proposed
for the next upload. It is maintained under `docs/rules/git/changelog.md` and
does not replace the human-owned weekly changelog under `docs/human/`.

`docs/rules/` contains repository-wide rules. It is the stable authority for
contributors; skills may link to these files but must not be the only location
of Git, environment, CI, simulation, or generated-artifact rules.

`docs/assets/` contains versioned image assets embedded by repository
documentation. Keep source-controlled diagrams readable at normal README width
and use descriptive, stable file names.

`docs/human/` contains the human-owned feature list, roadmap, and weekly
changelog. Agents may read it, but every mutation requires explicit human
confirmation for the exact proposed batch under `docs/rules/human-docs.md`.

`dist/` is empty in a clean checkout. It receives build products staged for
the board and nothing is authored there.

## Rules

One directory per design under `src/hw/rtl/`, `src/hw/tb/`, and
`src/hw/vivado_tcl/`. The design directory name must match across all three.
RTL may be divided into nested module directories that follow the instance
hierarchy. In `npu_matrix`, `npu_accelerator/` contains the top module,
`dma_axi/`, and `npu_matrix_core/`; the core contains `controller/` and
`npu_matrix_datapath/`, and the datapath contains `memory/` and
`systolic_array/`. Testbenches remain in `src/hw/tb/<design>/`.

A testbench is named `tb_<module>.sv` and lives in `src/hw/tb/<design>/`.
`make sim` discovers tests by that pattern, so a testbench outside it never
runs in CI.

`src/test/model/` defines the numeric contract: quantization, rounding
direction, saturation bounds, accumulator width and overflow behaviour. When
`src/jit/export/` starts depending on it, promote it to `src/jit/model/` rather than
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

Only versioned Markdown release records belong in top-level `changelog/`; do
not add `unreleased.md`. Each file must pass the commit-boundary,
roadmap-evidence, table ordering, five-row batching, link, and whitespace
checks defined by `docs/rules/git/changelog.md`.

## User-requested experiment campaigns

`exp/0911_performance_optimization/` versions compact records from the 21-run
hardware campaign: the overview, matrix results and CSV, initial-phase report,
experiment plan, and selected full-board comparison. Every matrix row uses the
same cold 16x16, K=256 transaction and functional reference. The 21 distinct
configurations satisfy the user's revised count requirement independently of
their cycle counts.

The complete local archive also contains frozen RTL, experiment drivers,
per-run records, Vivado output and raw board evidence. These larger files are
not part of this Git snapshot. Generated `artifacts/`, `scratch/`, `sources/`,
configs, results JSON, projects, logs, checkpoints, bitstreams and board
payloads stay ignored. `hardware/README.md` records where the earlier issue-59
archive moved; the archive itself remains local. The board summary reports
experiments 09, 19 and 21, including complete-overlay resources and physical
ResNet-18 times.

`exp/ab_test/` versions the IC-tool A/B report, reproduction instructions,
harness, licensed task inputs, tests, and selected acceptance examples.
`analysis/` regenerates aggregate metrics from a local `results/` archive.
The archive and generated `evidence/` stay ignored.

`exp/.docs/goals/jit-independent-and-aot-support-resent18/` records the
user-requested JIT isolation and ResNet18 AOT campaign: design, experiment
scripts, Result.md, Reproduce.md and compact acceptance evidence. Downloaded
models and compiler products now default to ignored `dist/resnet18/model/`
and `build/`; no BIT/HWH files are versioned there.

## Not in this repository

Do not add a top-level `skills/`, `scripts/`, `sim/`, `sw/`, `configs/`,
`logs/`, `core/`, `test/`, or `tools/`. Simulation entry points belong in
`src/test/`, board software in `src/runtime/`, project-generating Tcl in
`src/hw/vivado_tcl/`, and agent skills in `.codex/skills/`.

Do not add `vivado_projects/`, `results/`, a bitstream, or any Vivado project
directory. Regenerate them from `src/hw/vivado_tcl/`; bitstreams attach to a
GitHub Release.
