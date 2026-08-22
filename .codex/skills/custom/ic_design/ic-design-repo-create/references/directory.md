# Repository Directory

```text
<repo>/
|-- src/
|   |-- rtl/            synthesizable HDL, one subdirectory per design
|   |-- tb/             testbenches; never synthesized
|   |-- constraints/    .xdc timing and pin constraints
|   `-- vivado_tcl/     project-regenerating Tcl, one per design
|-- sim/                simulation build layer: Makefile, cocotb, wrappers
|-- sw/                 host/PS-side drivers, notebooks, reference models
|-- docs/
|   |-- human/          confirmed feature list, roadmap, weekly changelog
|   `-- <specs>         specifications, register maps, block diagrams
|-- .codex/skills/      agent skills, classified by purpose
|-- .github/workflows/  ci.yml, build.yml
|-- AGENTS.md
|-- CLAUDE.md
`-- README.md
```

Ignored, never tracked: `vivado_projects/`, `build/`, `results/`, `sim/waves/`.

## Why each split exists

`src/rtl` vs `src/tb` — synthesis reads one, simulation reads both. A tool
needs to select by directory, not by filename convention.

`src/vivado_tcl` vs `vivado_projects/` — the Tcl is the source; the project
directory is its output. Generate the Tcl with `write_project_tcl -force`, and
regenerate the project with `vivado -mode batch -source`. Committing the project
instead means absolute paths, binary churn, and gigabyte growth.

`sim/` vs `src/tb/` — `src/tb/` holds HDL, `sim/` holds the build logic that
decides which files compile, with which simulator, and how pass/fail becomes an
exit code. Keeping build logic out of `src/` is what makes CI a two-line job.

`sw/` — driver and reference-model code is neither RTL nor build tooling. For
accelerator projects the reference model is the test oracle, so it is a source.

`docs/human/` — human-owned product and workflow communication. Agents may read
it freely but must follow `references/rules/human-docs.md` before any mutation.

## Accelerator-stack variant

When the repository is not just an IP block but a whole accelerator stack —
hardware plus a compiler plus an on-board runtime — split `src/` by role
instead:

```text
src/
|-- hw/         rtl/, tb/, constraints/, vivado_tcl/
|-- test/       Makefile, model/, cocotb/, vectors/
|-- export/     trained model to device format
`-- runtime/    on-board loading and execution
examples/       demos built on export and runtime
```

The golden model moves under `src/test/model/` because verification is its
first consumer. Promote it to `src/model/` once `src/export/` depends on it too;
production code importing from a test directory is a smell worth one rename.

`npu_repo_in_pynq` uses this variant.

## Naming

Use `vivado_tcl`, not `vicado_tcl`. Use `CLAUDE.md` in uppercase; Linux runners
are case-sensitive and will not find `Claude.md`.
