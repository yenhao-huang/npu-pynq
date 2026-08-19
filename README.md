# npu_repo_in_pynq

NPU accelerator for the PYNQ-Z1 (Zynq-7020, `xc7z020clg400-1`). RTL, testbenches,
simulation, and host software.

See [AGENTS.md](AGENTS.md) for the layout and working rules.

## Layout

```text
src/rtl/        synthesizable HDL          src/tb/          testbenches
src/vivado_tcl/ project-regenerating Tcl   src/constraints/ .xdc
sim/            Makefile and cocotb        sw/              drivers, models
docs/           specifications             mount/           board deploy staging
skills/         agent skills
```

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
