# AGENTS

## Directory

| Path | Contents | Tracked |
| --- | --- | --- |
| `src/rtl/` | synthesizable HDL | yes |
| `src/tb/` | testbenches | yes |
| `src/constraints/` | `.xdc` | yes |
| `src/vivado_tcl/` | project-regenerating Tcl | yes |
| `sim/` | Makefile, cocotb tests | yes |
| `sw/` | drivers, reference models, notebooks | yes |
| `docs/` | specifications and human-facing project documents | yes |
| `docs/human/` | confirmed feature list, roadmap, weekly changelog | yes |
| `vivado_projects/` | Vivado output | no |

## Rules

- Never commit a Vivado project directory. Regenerate it from
  `src/vivado_tcl/` with `vivado -mode batch -source`.
- Every RTL change needs a corresponding testbench change or an explicit note
  saying why coverage is unchanged.
- Run `make -C sim lint sim` before pushing. CI runs the same two targets.
- Synthesis is self-hosted only; do not add Vivado steps to `ci.yml`.
- Releases are tags. Bitstreams attach to a Release, never to a commit.
- Agents may read `docs/human/`, but must obtain explicit human confirmation
  for the exact batch before creating, editing, appending, formatting,
  renaming, moving, or deleting anything under it. Code, issue, PR, merge, and
  release approval do not imply approval to update human documents.

## Commands

```bash
make -C sim lint          # static check
make -C sim sim           # run all testbenches
```

```bash
vivado -mode batch -source src/vivado_tcl/<design>/build_overlay.tcl
```
