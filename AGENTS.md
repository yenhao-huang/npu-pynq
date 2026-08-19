# AGENTS

NPU accelerator for the PYNQ-Z1 (Zynq-7020, `xc7z020clg400-1`). This repository
holds the RTL, its testbenches, and the host-side software for the NPU. It is a
fresh start: the earlier MAC prototype is not carried over.

## Layout

| Path | Contents | Tracked |
| --- | --- | --- |
| `src/rtl/<design>/` | synthesizable SystemVerilog | yes |
| `src/tb/<design>/` | testbenches, never synthesized | yes |
| `src/constraints/` | `.xdc` timing and pin constraints | yes |
| `src/vivado_tcl/<design>/` | project-regenerating Tcl | yes |
| `sim/` | Makefile and cocotb tests | yes (not `build/`, not `waves/`) |
| `sw/<design>/` | PYNQ driver, golden model, host tests | yes |
| `docs/` | specifications | yes |
| `skills/` | agent skills | yes |
| `mount/` | deploy staging, mirrored to the board | no, empty by design |
| `vivado_projects/`, `results/` | Vivado output | no |

## Rules

- Never commit a Vivado project directory or a bitstream. Regenerate the
  project with
  `vivado -mode batch -source src/vivado_tcl/<design>/build_overlay.tcl`, and
  attach bitstreams to a GitHub Release.
- `mount/` holds build products staged for the board. Nothing is authored
  there; it is empty in a clean checkout.
- Every RTL change needs a corresponding testbench change, or a note in the
  commit explaining why coverage is unchanged.
- Run `make -C sim lint sim` before pushing. CI runs the same two targets.
- Synthesis is self-hosted only. Do not add Vivado steps to `ci.yml`; GitHub
  hosted runners cannot run Vivado.
- Releases are tags (`v0.1.0-<design>`).

## Commands

```bash
make -C sim lint          # TOP=<module> to pin the lint root
make -C sim sim           # runs every src/tb/*/tb_*.sv it finds
make -C sim clean
```

```powershell
vivado -mode batch -source src/vivado_tcl/<design>/build_overlay.tcl
```

## Board

PYNQ-Z1 at `192.168.2.99`, user `xilinx`, over direct Ethernet. Transfers use
the `send-to-pynq-board` skill under `skills/operations/`. Credentials are never
stored in this repository.
