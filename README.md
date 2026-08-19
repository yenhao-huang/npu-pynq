# PYNQ-Z1 workspace

RTL sources, simulation, host software, and a Windows-to-board sync controller
for PYNQ-Z1 accelerator designs. Current design: `mac_npu`, an AXI4-Lite
multiply-accumulate unit.

See [AGENTS.md](AGENTS.md) for the full layout and working rules.

## Layout

```text
src/rtl/        synthesizable HDL          src/tb/          testbenches
src/vivado_tcl/ project-regenerating Tcl   src/constraints/ .xdc
sim/            Makefile and runners       sw/              drivers, models
docs/           specifications             mount/           board deploy payload
```

Vivado projects are never committed. Regenerate them from `src/vivado_tcl/`.

## Simulate

```bash
make -C sim lint
make -C sim sim
```

```bash
python -m unittest discover -s sw/mac_npu -p 'test_*.py' -v
```

## Build the overlay

Requires Vivado 2026.1 locally; this cannot run on a GitHub-hosted runner.

```powershell
& .\src\vivado_tcl\mac_npu\build_overlay.ps1
```

The bitstream and hardware handoff land in `mount/mac_npu/overlay/`.

## Deploy to the board

```powershell
& .\sw\mac_npu\deploy_and_test.ps1
```

This stages the board-runtime Python into `mount/`, syncs, and runs the hardware
smoke test. To sync only:

```powershell
& .\core\service\pynq_sync_controller.ps1 -Once -DryRun
```

Sync contract: local source of truth is `mount/`, destination is
`/home/xilinx/jupyter_notebooks/pynq_z1_repo/`, transport is SSH/SCP, direction
is Windows to board only, remote files are never deleted, and change detection
uses a SHA-256 manifest under `logs/`. Authentication uses the existing OpenSSH
configuration or an interactive prompt; credentials are never stored.

## CI and CD

`ci.yml` runs lint, simulation, and host tests on a GitHub-hosted runner.
`build.yml` runs synthesis and is gated behind `runs-on: [self-hosted, vivado]`;
it does nothing until such a runner is registered.
