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
|   |-- cd/
|   |   `-- *.ps1              automated deployment and acceptance scripts
|   `-- workflows/
|       |-- cd.yml
|       `-- ci.yml
|-- .codex/
|   `-- skills/
|       |-- dev/                 shared development workflows
|       |-- deploy/              installation and deployment workflows
|       `-- custom/
|           `-- ic_design/       repository-specific FPGA/NPU workflows
|-- src/
|   |-- hw/
|   |   |-- rtl/
|   |   |   `-- <design>/
|   |   |       `-- *.sv
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
|   |   `-- waves/
|   |-- export/
|   |   `-- *.py
|   `-- runtime/
|       `-- *.py
|-- examples/
|   `-- <example>/
|       |-- README.md
|       |-- *.ipynb
|       |-- *-source.json        pinned download metadata
|       |-- package_example.py
|       |-- run_on_board.py
|       |-- deploy_release.ps1
|       |-- model/
|       |   `-- .gitkeep
|       |-- scripts/
|       |   `-- *.py
|       |-- runtime/
|       |   `-- *.py
|       |-- tests/
|       |   `-- test_*.py
|       `-- docs/
|           `-- <study>/          example-scoped measurement studies
|               |-- README.md
|               |-- scripts/
|               |-- data/
|               `-- *.html
|-- docs/
|   |-- assets/
|   |   `-- *.png              versioned images embedded by documentation
|   |-- rules/
|   |   |-- index.md
|   |   |-- environment.md
|   |   |-- generated-artifacts.md
|   |   |-- human-docs.md
|   |   |-- simulation.md
|   |   |-- ci-cd.md
|   |   |-- filetree.md
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
|   |-- plans/
|   |   `-- <YYYY>-<MMDD>-<NNN>-<topic>.md
|   |-- acceptance/
|   |   `-- reproduce.md        how to reproduce a delivered capability
|   |-- <toolchain>/
|   |   `-- README.md           developer toolchain documentation
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
|       `-- tests/
|-- .pi/
|   |-- settings.json
|   `-- extensions/
|       `-- *.ts
|-- .mcp.json
|-- openspec/
|   |-- changes/
|   `-- specs/
|-- .ic/                        run store, not tracked
`-- mount/
```

## What each directory is for

`src/hw/` is the NPU itself: everything Vivado reads. `rtl/` is synthesized;
`tb/` never is. That boundary is hard, so the two stay separate directories.

`src/test/` verifies that the hardware computes the right answer. `model/` is
the numpy golden reference, `cocotb/` are the Python tests that compare RTL
against it, and `Makefile` is what CI invokes.

`src/export/` turns a trained model into whatever format the NPU executes.

`src/runtime/` loads an overlay on the board and runs an exported model on it.

`examples/` consumes the three above. Nothing under `src/` may import from it.
An example owns its application-specific runtime, notebooks, package builder,
board acceptance entry point, deployment wrapper, and focused host tests. The
package builder may copy an explicit allowlist of shared `src/runtime/` modules
into generated deploy output, but those copies are never committed.
Every user-facing example includes an `.ipynb` demo, committed output-free so
the file stays diffable. `examples/resnet18/resnet18.ipynb` is the one
exception: it is committed with the outputs of a real PYNQ-Z1 run, so the
board's figures and predicted labels are readable on GitHub without running
anything. Re-executing it replaces those outputs wholesale, so commit a run you
intend to publish. The notebook
is the canonical human validation entry point: its README may prepare and
deploy inputs, but must ultimately direct the user to the notebook. CLI board
entry points support the notebook and automation; they do not replace the
human demo. Download and conversion commands belong under its `scripts/`;
generated checkpoints, converted model packages, corpora, and model evidence
go under its `model/` workspace and remain ignored except for `.gitkeep`.
A `*-source.json` file pins one download set by URL, length, SHA-256, and
license; the downloaded bytes themselves are never committed.
The canonical generated-data path is `examples/<example>/model/`.

`examples/<example>/docs/<study>/` holds a self-contained measurement study
for that example: its `README.md` states the conclusions, `scripts/` holds the
entry points that produced them, `data/` holds the machine-readable
measurements, and a rendered `*.html` visualization may sit at the study root.
A study is committed, unlike the `model/` workspace, because its conclusions
have to remain readable and auditable without re-running anything. Input
corpora stay out: they belong under `model/` and stay ignored. Keep a study's
committed bytes proportionate; a rendered page and its summary JSON/CSV are in
scope, raw captures and datasets are not.

`.github/cd/` owns non-interactive deployment and acceptance scripts used by
continuous delivery. Example-local `deploy_release.ps1` files only copy a
release for later human validation; they must not execute acceptance, request
`sudo`, claim a physical PASS, or collect evidence.

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

`.pi/` and `.mcp.json` attach those tools to specific agents: `.pi/extensions/`
registers them as native pi tools and `.mcp.json` registers the MCP server.
They are configuration only. The skill that tells a model when to use the
tools lives with the other skills, under
`.codex/skills/custom/ic_design/ic-design-tools/`, so one copy serves pi,
Claude Code and Codex alike.

`docs/plans/` holds architecture and implementation plans. `docs/acceptance/`
holds reproduction procedures for delivered capabilities: each records the
exact commands and the output they actually produced, and states what was not
verified. `docs/<toolchain>/README.md` documents a toolchain under `tools/`.

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

`mount/` is empty in a clean checkout. It receives build products staged for
the board and nothing is authored there.

## Rules

One directory per design under `src/hw/rtl/`, `src/hw/tb/`, and
`src/hw/vivado_tcl/`. The directory name is the design name and must match
across all three.

A testbench is named `tb_<module>.sv` and lives in `src/hw/tb/<design>/`.
`make sim` discovers tests by that pattern, so a testbench outside it never
runs in CI.

`src/test/model/` defines the numeric contract: quantization, rounding
direction, saturation bounds, accumulator width and overflow behaviour. When
`src/export/` starts depending on it, promote it to `src/model/` rather than
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

## Not in this repository

Do not add a top-level `skills/`, `scripts/`, `sim/`, `sw/`, `configs/`,
`logs/`, `core/`, `test/`, or `tools/`. Simulation entry points belong in
`src/test/`, board software in `src/runtime/`, project-generating Tcl in
`src/hw/vivado_tcl/`, and agent skills in `.codex/skills/`.

Do not add `vivado_projects/`, `results/`, a bitstream, or any Vivado project
directory. Regenerate them from `src/hw/vivado_tcl/`; bitstreams attach to a
GitHub Release.
