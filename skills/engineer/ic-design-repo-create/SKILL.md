---
name: ic-design-repo-create
description: Create or restructure an FPGA/ASIC RTL repository with a source/build separation, GitHub Actions CI (lint + simulation) and CD (synthesis on a self-hosted runner), and a branch model suited to hardware releases. Use when the user asks to build, scaffold, lay out, reorganize, or refactor an RTL, HDL, Verilog, SystemVerilog, FPGA, Vivado, or IC design repository, or asks where RTL, testbenches, constraints, Tcl, or Vivado projects should live.
---

# IC Design Repo Create

Lay out an RTL repository so that Git tracks only human-authored sources, and
every tool product is reproducible from them. The controlling rule: **if a tool
can regenerate it, it does not belong in Git.**

## Workflow

1. Read `references/rules/env.md` and `references/directory.md`. Read
   `references/rules/git/` before any commit, branch, pull request, or issue.
2. Inventory the target directory. Separate every existing path into `source`,
   `tool product`, or `deploy payload`. Run
   `git ls-files` and `git status --short` together — untracked design sources
   are the most common defect and must be reported before any move.
3. Show the user the inventory and the proposed destination for each path.
   Do not move anything until they confirm.
4. Create the directory structure in `references/directory.md`. Move tracked
   sources with `git mv` and untracked ones with a plain move, so history
   survives where it exists.
5. Install `.gitignore` from `references/gitignore.md`. Verify no tool product
   is tracked: `git ls-files | grep -E '\.(bit|dcp|jou|log|pb|rpx|wdb)$'`
   must print nothing.
6. Install CI from `references/workflows/ci.yml` and, only if a self-hosted
   runner exists, `references/workflows/build.yml`. Read
   `references/ci-cd.md` first.
7. Create `sim/Makefile` from `references/simulation.md` and confirm
   `make -C sim lint` passes locally before committing.
8. Write `AGENTS.md`, and a `CLAUDE.md` whose entire body is `See @AGENTS.md`.
9. Set up branches per `references/rules/git/git-branch.md`.
10. Report what moved, what is now ignored, and what CI cannot run.

## Guardrails

- Never `git rm`, delete, or overwrite a design source, bitstream, or Vivado
  project during a restructure. Move only; let `.gitignore` handle exclusion.
- Never commit a Vivado project directory (`*.cache/`, `*.gen/`, `*.runs/`,
  `*.hw/`, `*.sim/`, `*.xpr`). Commit the regenerating Tcl instead.
- Never claim CI runs synthesis on a GitHub-hosted runner. Vivado is not
  installable there. Say so explicitly and gate synthesis behind
  `runs-on: self-hosted`.
- Do not rename a directory that another script reads without updating that
  script in the same change. Grep for the old name first.
- Stop and ask when a directory serves both as a source location and as a
  deploy payload staging area; splitting it changes runtime behaviour.
- Never merge `dev` into `main`. That is a deploy decision a person makes; an
  agent may prepare and report it, never perform it.
- Do not add a `scripts/` directory at the repository root, a changelog, or a
  contributor guide unless the user asks.
