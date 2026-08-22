# npu_repo_in_pynq

NPU accelerator for the PYNQ-Z1 (Zynq-7020, `xc7z020clg400-1`): the hardware
design, its verification, the model export path, and the on-board runtime.

See [AGENTS.md](AGENTS.md) for working rules and
[docs/rules/filetree.md](docs/rules/filetree.md) for the full tree.

## Directory

```text
npu_repo_in_pynq/
|-- src/
|   |-- hw/             NPU hardware design
|   |   |-- rtl/            synthesizable SystemVerilog
|   |   |-- tb/             testbenches, never synthesized
|   |   |-- constraints/    .xdc timing and pin constraints
|   |   `-- vivado_tcl/     project-regenerating Tcl
|   |-- test/           verification
|   |   |-- Makefile        make lint / make sim
|   |   |-- model/          numpy golden reference
|   |   |-- cocotb/         Python tests comparing RTL against the model
|   |   `-- vectors/        test data
|   |-- export/         trained model -> NPU executable format
|   `-- runtime/        loads the overlay and runs a model on the board
|-- examples/           demos built on export and runtime
|-- docs/               specifications and repository rules
|-- .codex/
|   `-- skills/
|       |-- dev/                 shared development workflows
|       |-- deploy/              setup and deployment workflows
|       `-- custom/ic_design/    repository-specific IC design workflows
|-- openspec/           change proposals and specifications
`-- mount/              board deploy staging, empty by design
```

The data flows one way: `export` produces what `runtime` consumes, `runtime`
drives the circuit synthesized from `hw`, and `examples` ties the three
together.

Vivado projects and bitstreams are never committed. Regenerate projects from
`src/hw/vivado_tcl/`; bitstreams attach to a GitHub Release.

## Verify

```bash
make -C src/test lint
make -C src/test sim
```

Both targets run on open-source tools only (`verilator`, `iverilog`), which is
what lets CI run them on a GitHub-hosted runner. With no RTL or testbenches
present yet, both report that there is nothing to do and exit clean.

## Build a bitstream

Requires Vivado locally; this cannot run on a GitHub-hosted runner.

```bash
vivado -mode batch -source src/hw/vivado_tcl/<design>/build_overlay.tcl
```

## CI and CD

`ci.yml` runs lint and simulation on every push to `main` and `dev`.
`build.yml` runs synthesis and is gated behind `runs-on: [self-hosted, vivado]`;
it does nothing until such a runner is registered.

## Status

Empty scaffold. No design has been added yet.
