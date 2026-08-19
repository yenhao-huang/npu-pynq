# npu_repo_in_pynq

NPU accelerator for the PYNQ-Z1 (Zynq-7020, `xc7z020clg400-1`). RTL, testbenches,
simulation, and host software.

See [AGENTS.md](AGENTS.md) for the directory structure and working rules.

## Directory

```text
npu_repo_in_pynq/
|-- src/
|   |-- rtl/            synthesizable HDL, one directory per design
|   |-- tb/             testbenches, never synthesized
|   |-- constraints/    .xdc timing and pin constraints
|   `-- vivado_tcl/     project-regenerating Tcl
|-- sim/                Makefile and cocotb tests
|-- sw/                 PYNQ drivers, golden models, host tests
|-- docs/               specifications and repository rules
|-- skills/             agent skills
`-- mount/              board deploy staging, empty by design
```

[docs/rules/filetree.md](docs/rules/filetree.md) has the full tree and the rules
for what may be added where.

Vivado projects and bitstreams are never committed. Regenerate projects from
`src/vivado_tcl/`; bitstreams attach to a GitHub Release.

## Simulate

```bash
make -C sim lint
make -C sim sim
```

Both targets run on open-source tools only (`verilator`, `iverilog`), which is
what lets CI run them on a GitHub-hosted runner. With no RTL or testbenches
present yet, both report that there is nothing to do and exit clean.

## Build a bitstream

Requires Vivado locally; this cannot run on a GitHub-hosted runner.

```bash
vivado -mode batch -source src/vivado_tcl/<design>/build_overlay.tcl
```

## CI and CD

`ci.yml` runs lint and simulation on every push to `main` and `dev`.
`build.yml` runs synthesis and is gated behind `runs-on: [self-hosted, vivado]`;
it does nothing until such a runner is registered.

## Status

Empty scaffold. No design has been added yet.
