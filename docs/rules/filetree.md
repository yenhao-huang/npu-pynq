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
|   |-- release/
|   |   |-- models.json         the models a release ships, and how each is checked
|   |   |-- board_check.sh      run a release archive on the board, bring its output back
|   |   `-- version.py          the release version changelog/ declares
|   `-- workflows/
|       |-- ci.yml
|       |-- exp-board.yml       issue-119 experiment runs on the self-hosted runners
|       |-- release.yml         validate a release/vX.Y.Z branch, leave a draft Release
|       `-- publish.yml         publish a validated, merged draft
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
|   |   |                    npu_accelerator/scalar_core/:
|   |   |                      scalar_core.sv   wrapper, SCALAR_IMPL selects
|   |   |                      custom/          custom_* modules, the NPU ISA sequencer
|   |   |                      cva6/            CVA6 integration: *.sv, cva6.f, waiver.vlt, README.md
|   |   |                        upstream/      openhwgroup/cva6 submodule, pinned tag, never modified
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
|   |   |   `-- risc_v_npu/      rv32asm.py, isa_to_rv.py: ISA test programs as RISC-V for the CVA6 bench
|   |   `-- waves/
|   |-- vendors/                 one folder per accelerator; imported by export and runtime, imports neither
|   |   `-- npu_pynq/
|   |       |-- spec/            hardware facts: isa.py, abi.py (registers), layout.py, quant.py (schemes accepted), board.py
|   |       |-- custom_dialect/  C++ (LLVM 22): the npu dialect, --npu-claim, Pack, Encode; a
|   |       |                    library pynqo-backend links; never shipped to the board
|   |       `-- driver/          board side; follows spec/
|   |           |-- npu_rt/      libnpu_rt (C++): the NPU delegate backend, PYNQ and SIM backends,
|   |           |                built on the target with its own g++ and CMake
|   |           `-- overlay/     build.py writes the BIT/HWH manifest; verify.py checks it before loading
|   |   `-- risc_v_npu/          the same NPU with the CVA6 RISC-V scalar core (SCALAR_IMPL="CVA6");
|   |       |                    fully separate from npu_pynq: matrix-core facts are copied, not imported
|   |       |-- spec/            abi.py (CSR map + capability bit 7, MMIO job registers, address map),
|   |       |                    layout.py, quant.py, board.py, gen_mmio_header.py
|   |       |-- custom_dialect/  C++ (LLVM 22): the rvnpu dialect, --rvnpu-claim, Pack, LowerToRiscv,
|   |       |   |                Programs (rv_programs.*); include/RvNpu/RvNpuMmio.h is generated
|   |       |   `-- runtime/     crt0.S and link.ld, linked into every generated RISC-V program
|   |       `-- driver/          board side; follows spec/
|   |           |-- rv_npu_rt/   librv_npu_rt (C++): loads RISC-V binaries, PYNQ backend only
|   |           `-- overlay/     manifest and verification of the CVA6 overlay
|   |-- export/                  host-side model -> package; never runs on the board
|   |   |-- cli.py               build <config>
|   |   |-- pipeline.py          load -> quantize -> export_graph -> to_mlir -> pynqo-backend
|   |   |-- configs/             one YAML file per model: source, quantization, graph, devices
|   |   |-- requirements.txt     the pinned front-end frameworks
|   |   |-- frontend/            Python
|   |   |   |-- load/            frameworks/ (transformers, torchvision): the float model
|   |   |   |-- quantization/    schemes.py; torchao/: our configs and the modules they swap in
|   |   |   |-- export_graph/    torch.export: one ExportedProgram per entry point
|   |   |   `-- to_mlir/         torch-mlir's importer (llvm/torch-mlir wheels): torch.mlir
|   |   |-- middle-end/          C++ pynqo-middle-end: torch.mlir -> model.mlir; builds torch-mlir;
|   |   |                        torch/transforms/: one file per rewrite on the torch dialect
|   |   `-- backend/             C++ pynqo-backend: partition, delegates/ (npu/, risc_v_npu/),
|   |                            package (weights.bin), cpu/toolchain (LLVM IR, llc, link),
|   |                            device_register
|   |-- deploy/                  deploy.sh (send to the board, or ARCHIVE= a release archive),
|   |                            resnet18_assets.py and its pins
|   |-- deprecate/               retired code, kept for reference; nothing active imports it
|   |   `-- jit/                 the self-contained just-in-time stack
|   |       |-- model/           numeric and ResNet model contracts
|   |       |-- export/          legacy ResNet package exporter
|   |       |-- runtime/         interpreter, local DMA/overlay, acceptance
|   |       |-- examples/resnet18/ original JIT notebook and delivery helpers
|   |       |-- tests/           JIT-specific regression tests
|   |       |-- reference/       independent quantized graph reference
|   |       `-- delivery/        frozen JIT delivery helpers
|   `-- runtime/                 board-side execution; never imports src/export
|       |-- cli.py               one entry point: python -m src.runtime.cli {llm,classifier} ...
|       |-- share/               knows no task: loader.py (load and call an pynqo-package/2), profile.py
|       |-- delegate/            the __delegate__ dispatcher (dispatch.c) and device backends
|       `-- tasks/               one package per manifest graph kind, each with its own cli.py
|           |-- llm/             engine, tokenizer, sampling
|           `-- classifier/      engine, preprocessing, top-k
|-- examples/
|   `-- <example>/              see docs/rules/create-example.md
|       |-- README.md            Linux deployment manual (WSL on Windows)
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
|   |   |-- lib/
|   |   |   `-- register_quant.md
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
|   |-- manual/
|   |   `-- <topic>/            model.md: commands to run one part of the stack, and their outputs
|   |-- plans/
|   |   `-- <YYYY>-<MMDD>-<NNN>-<topic>.md
|   |-- acceptance/
|   |   `-- reproduce.md        how to reproduce a delivered capability
|   |-- <toolchain>/
|   |   `-- README.md           developer toolchain documentation
|   |-- milestone.md          software-stack feature checklist: done and planned
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
|   |-- ab_test/                IC-tool A/B report and experiment source
|   `-- scalar_core_compare/    CUSTOM vs CVA6 scalar core A/B report (README.md) and run.py
|-- openspec/
|   |-- changes/
|   `-- specs/
|-- .ic/                        run store, not tracked
`-- dist/
```

## What each directory is for

`src/hw/` is the NPU itself: everything Vivado reads. `rtl/` is synthesized;
`tb/` never is. That boundary is hard, so the two stay separate directories.

`src/test/` verifies that the hardware computes the right answer. `model/` is
the numpy golden reference, `cocotb/` are the Python tests that compare RTL
against it, and `Makefile` is what CI invokes.

`src/vendors/` holds everything specific to one accelerator, one folder per
vendor, as a layer below both `src/export/` and `src/runtime/`: both import
it, and it imports neither. Inside a vendor, `spec/` states the hardware as
data; `custom_dialect/` (compile time) and `driver/` (run time) follow it and
never each other, so the board bundle ships `spec/` and `driver/` without the
compiler. `python -m unittest src.test.tests.test_vendor_boundary` enforces
these edges.

`src/vendors/npu_pynq/spec/` is the single definition of the NPU: `isa.py`
encodes and decodes the instruction set, `abi.py` is everything around it that
software uses to talk to the NPU (identity and version, capability bits, the
AXI-Lite register map, status bits, error codes; `src/test/tests/test_abi.py`
checks the RTL and `driver/npu_rt/` against it), `layout.py` fixes how a tiled GEMM
maps onto DDR, `quant.py` names the quantization schemes the NPU accepts, and
`board.py` the board's CPU. RTL opcodes must match `isa.py`;
`src/test/tests/test_isa.py` enforces that against the golden model
`src/test/model/isa_sim.py`. `custom_dialect/` claims INT8 contractions on
constant weights as `npu.matmul`, packs their weights as B tiles and encodes
ISA programs; its C++ encoder is a port of `isa.py` and `layout.py`.
`driver/npu_rt/` is libnpu_rt, which the runtime's `__delegate__` dispatcher
calls for every NPU region; its SIM backend is a C++ port of `isa_sim.py`.
`driver/overlay/` writes the overlay manifest after Vivado and verifies it
before the board loads the bitstream.

`src/export/` turns a model into a package on the host, entirely from a
config (`configs/*.yml`): `frontend/load/` builds the float model with its
framework; `frontend/quantization/` quantizes it with our torchao configs;
`frontend/export_graph/` captures it with
`torch.export`; `frontend/to_mlir/` imports it into the torch dialect with
torch-mlir; `middle-end/` (`pynqo-middle-end`, C++ against LLVM 22 and a pinned
torch-mlir it builds from source) rewrites layers while they are recognisable and
lowers them to linalg with torch-mlir's pipelines; and
`backend/` (`pynqo-backend`, C++ against LLVM 22, built by `pipeline.py` when
stale) lets each device delegate claim its operations, replaces them with
`__delegate__(id, in, out)` calls, moves the remaining weights into
`weights.bin`, and compiles the rest for the Cortex-A9 and the host. A new
model is a new config; a new device is a delegate under `backend/delegates/`
and its vendor folder. Front-end stages do not import each other; only
`pipeline.py` passes a model from one to the next. `python -m
unittest src.test.tests.test_export_layering` enforces that, and that
`src/runtime` and `src/vendors` never import `src/export`.

`src/deploy/deploy.sh` sends an export directory to the board: the package with
the board's library, the runtime and driver sources, the overlay and the
example's files. With `ARCHIVE=` it writes the same tree to a `.tar.gz`, which
is what a release ships and what its board check runs.

`src/deprecate/jit/` owns the deprecated just-in-time quantized graph interpreter, its
exporter, numeric contracts, examples, references, focused tests and delivery
helpers. It imports no other source packages. Active source, tests, public
examples and delivery workflows must not import it: removing `src/deprecate` must
leave AOT export and inference functional. Check both boundaries with
`python -m unittest src.test.tests.test_jit_boundary` and execute active tests
with JIT physically absent. The hardware numeric reference remains independently
available under `src/test/model/numeric.py`; its contract is unchanged.

`src/runtime/` runs packages on the board, or on the host with libnpu_rt's
simulator. `share/loader.py` loads a package's `model.so` for the CPU and
installs the `__delegate__` dispatcher (`delegate/dispatch.c`, built on the
target) and the device backends (`delegate/npu.py`: libnpu_rt with the
buffers `pynq.allocate` provides). Each task under `tasks/`, one directory
named after the manifest's graph kind, builds its engine on that: `llm/`
(prefill, decode, two KV cache sets swapped per call) and `classifier/`.
`cli.py` only picks the task; each task's `cli.py` owns its arguments. Tasks
never import each other, and `share/` and `delegate/` import no task.
`share/profile.py` times the stages an engine marks and collects the delegate
backends' phase counters (libnpu_rt's `profile.cpp`) behind each task's
`--profile DIR`.

`examples/` demonstrates existing source functionality. Its tracked file list,
platform deployment manuals, notebook requirements and generated-data locations
are defined in [create-example.md](create-example.md).

`.github/release/` holds what `release.yml` runs beyond the workflow itself:
the release model list, the board check, and the version resolver.

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
`docs/milestone.md` is the software stack's feature checklist: what the
inference framework and compiler do today and what is planned.
`docs/manual/` holds runnable examples of parts of the stack, each with the
output it writes; `torch_mlir_arch/model.md` gives the commands, using only
torch and torch-mlir's own tools, that take a one-layer model to its
ExportedProgram, torch-dialect and linalg forms. `venv/README.md` records
where this machine's Python environments are and how to recreate them. `models/<model>/README.md` introduces a model an example runs: where it
comes from, its architecture and how the export configures it.
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
hierarchy. In `npu_accelerator`, the design directory holds the top module
`npu_accelerator.sv`,
`csr/` (the control and status registers, reached over AXI-Lite),
`scalar_core/`, and `matrix_core/`; the matrix core holds its module
`matrix_core.sv` and contains `controller/`, `memory/` and
`systolic_array/`. Each core has a controller beside the modules it
controls: `custom_controller` is the program-level run control over the
CUSTOM scalar core's pipeline stages (start admission, run state, busy, faults
and the reported error), and `matrix_core_controller` sequences one job over the
operand buffers and the systolic array, cycle by cycle. Testbenches remain in `src/hw/tb/<design>/`.
Only the top module carries the `npu_` prefix. A module that belongs to one
core takes the core's name as its prefix (`matrix_core_controller`), a generic
building block says what it is (`axis_skid`), and a testbench is
`tb_<module>`.

`scalar_core/` holds the wrapper `scalar_core.sv`, whose `SCALAR_IMPL`
parameter selects an implementation, and one self-contained folder per
implementation; there is no shared folder between them:

- `custom/`: the 64-bit NPU instruction sequencer. Every module is prefixed
  `custom_` (`custom_scalar_core`, `custom_fetch`, `custom_dma`,
  `custom_sync_fifo`, ...).
- `cva6/`: a CVA6 RISC-V core that drives the matrix core through MMIO
  registers. Our modules and packages are prefixed `cva6_` or end in `_cva6`
  (`scalar_core_cva6`, `cva6_npu_mmio`, `cva6_dma`, `cva6_npu_config_pkg`).
  `cva6/upstream/` is the third-party `openhwgroup/cva6` git submodule, pinned
  to a release tag and never modified: customization goes through CVA6's own
  configuration package, and an unavoidable source change would live in
  `cva6/patches/`, applied at build time and documented in `cva6/README.md`.
  `cva6/cva6.f` is the only way to compile the CVA6 sources; the default
  `src/test/Makefile` flow excludes `upstream/` and the files that need CVA6
  packages.

SystemVerilog module and package names are global and every implementation
compiles together, so each name must be unique across implementations; that is
why each implementation carries its own prefix. A building block an
implementation needs (a FIFO, the tensor DMA) is a copy inside that
implementation's folder under its prefix, not a shared module.

A testbench is named `tb_<module>.sv` and lives in `src/hw/tb/<design>/`.
`make sim` discovers tests by that pattern, so a testbench outside it never
runs in CI.

`src/test/model/` defines the numeric contract: quantization, rounding
direction, saturation bounds, accumulator width and overflow behaviour. When
`src/deprecate/jit/export/` starts depending on it, promote it to `src/deprecate/jit/model/` rather than
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
