# AGENTS

NPU accelerator for the PYNQ-Z1 (Zynq-7020, `xc7z020clg400-1`). This repository
holds the hardware design, its verification, the model export path, and the
on-board runtime.

## Directory

| Path | Contents | Tracked |
| --- | --- | --- |
| `src/hw/rtl/<design>/` | synthesizable SystemVerilog | yes |
| `src/hw/tb/<design>/` | testbenches, never synthesized | yes |
| `src/hw/constraints/` | `.xdc` timing and pin constraints | yes |
| `src/hw/vivado_tcl/<design>/` | project-regenerating Tcl | yes |
| `src/test/` | Makefile, golden model, cocotb tests, vectors | yes (not `build/`, not `waves/`) |
| `src/export/` | trained model to NPU executable format | yes |
| `src/runtime/` | on-board overlay loading and execution | yes |
| `examples/` | demos built on export and runtime | yes |
| `docs/` | specifications and repository rules | yes |
| `docs/human/` | human-confirmed features, roadmap, and weekly changelog | yes |
| `.codex/skills/dev/` | shared development workflow skills | yes |
| `.codex/skills/deploy/` | setup and deployment skills | yes |
| `.codex/skills/custom/ic_design/` | repository-specific IC design skills | yes |
| `openspec/` | change proposals and specifications | yes |
| `mount/` | deploy staging, mirrored to the board | no, empty by design |
| `vivado_projects/`, `results/` | Vivado output | no |

[docs/rules/index.md](docs/rules/index.md) is the repository-wide rule index;
[docs/rules/filetree.md](docs/rules/filetree.md) is the authority on the tree
and on what may not be added.

## Rules

- Never commit a Vivado project directory or a bitstream. Regenerate the
  project with
  `vivado -mode batch -source src/hw/vivado_tcl/<design>/build_overlay.tcl`,
  and attach bitstreams to a GitHub Release.
- `src/test/model/` defines the numeric contract that `src/hw/` implements and
  `src/export/` must match: quantization, rounding, saturation, accumulator
  width and overflow. Change it only deliberately; both sides depend on it.
- Dependencies point one way. `examples/` may import from `src/`; nothing under
  `src/` may import from `examples/`.
- `mount/` holds build products staged for the board. Nothing is authored
  there; it is empty in a clean checkout.
- Every RTL change needs a corresponding testbench change, or a note in the
  commit explaining why coverage is unchanged.
- Run `make -C src/test lint sim` before pushing. CI runs the same two targets.
- Synthesis is self-hosted only. Do not add Vivado steps to `ci.yml`; GitHub
  hosted runners cannot run Vivado.
- Work from one claimed issue in a dedicated worktree on branch
  `npu/issue<issue-id>-<agent-id>` cut from `dev`, and merge it back into `dev`
  through a pull request. Never commit directly to `main` or `dev`.
- Never merge `dev` into `main`. `main` is the deploy version and that merge is
  a person's decision; prepare it and report what you could not verify.
- Every approved `dev` → `main` deployment merge must be followed by a GitHub
  Release. Releases are semantic-version tags on `main`, using the default
  patch sequence `v0.1.0` → `v0.1.1` → `v0.1.2` unless a release decision
  explicitly selects a new major or minor version.
- Agents may read `docs/human/`, but must obtain explicit human confirmation
  for the exact batch before creating, editing, appending, formatting,
  renaming, moving, or deleting anything under it. Code, issue, PR, merge, and
  release approval do not imply approval to update human documents.

Full Git rules:
[docs/rules/git/](docs/rules/git/) — branches, commits, pull requests, issues.

## Commands

```bash
make -C src/test lint     # TOP=<module> to pin the lint root
make -C src/test sim      # runs every src/hw/tb/*/tb_*.sv it finds
make -C src/test clean
```

```bash
vivado -mode batch -source src/hw/vivado_tcl/<design>/build_overlay.tcl
```

## Board

PYNQ-Z1 at `192.168.2.99`, user `xilinx`, over direct Ethernet. Transfers use
the `send-to-pynq-board` skill under `.codex/skills/deploy/`. Credentials are
never stored in this repository.
