# AGENTS

PYNQ-Z1 accelerator workspace. RTL sources, simulation, host software, and the
board sync controller.

## Layout

| Path | Contents | Tracked |
| --- | --- | --- |
| `src/rtl/<design>/` | synthesizable SystemVerilog | yes |
| `src/tb/<design>/` | testbenches, never synthesized | yes |
| `src/constraints/` | `.xdc` timing and pin constraints | yes |
| `src/vivado_tcl/<design>/` | project-regenerating Tcl and its launcher | yes |
| `sim/` | Makefile, PowerShell runners, waveform output | yes (not `build/`) |
| `sw/<design>/` | PYNQ driver, golden model, host tests, deploy script | yes |
| `docs/` | specifications | yes |
| `configs/`, `core/`, `logs/` | board sync controller and its state | yes |
| `mount/` | deploy staging tree mirrored to the board | overlay only |
| `skills/` | agent skills | yes |
| `vivado_projects/`, `vivado/`, `results/` | Vivado output | no |

## Rules

- Never commit a Vivado project directory. Regenerate it with
  `vivado -mode batch -source src/vivado_tcl/<design>/build_overlay.tcl`.
- `sw/` is the source of truth for board-runtime Python. `mount/` receives
  staged copies from `sw/mac_npu/deploy_and_test.ps1`; never edit files under
  `mount/` directly.
- Every RTL change needs a corresponding testbench change, or a note in the
  commit explaining why coverage is unchanged.
- Run `make -C sim lint sim` and the host tests before pushing. CI runs the
  same commands.
- Synthesis is self-hosted only. Do not add Vivado steps to `ci.yml`; GitHub
  hosted runners cannot run Vivado.
- Releases are tags (`v0.1.0-mac-npu`). Bitstreams attach to a Release, never
  to a commit.

## Commands

```bash
make -C sim lint
make -C sim sim
python -m unittest discover -s sw/mac_npu -p 'test_*.py' -v
```

```powershell
& .\src\vivado_tcl\mac_npu\build_overlay.ps1     # synthesize, produces overlay
& .\sw\mac_npu\deploy_and_test.ps1               # stage, sync, board smoke test
& .\core\service\pynq_sync_controller.ps1 -Once  # sync mount/ only
```

## Board

Host `192.168.2.99`, user `xilinx`, remote root
`/home/xilinx/jupyter_notebooks/pynq_z1_repo`. Sync is Windows to board only and
never deletes remote files. Credentials are never stored in this repository.
