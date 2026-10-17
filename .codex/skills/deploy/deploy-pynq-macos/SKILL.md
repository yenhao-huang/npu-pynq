---
name: deploy-pynq-macos
description: Deploy and run NPU overlays and MLIR-compiled models (SmolLM2, Qwen3, ResNet-18) on the PYNQ-Z1 directly from macOS, without the Windows self-hosted runner. Use when the user wants to send overlays or packages to the board from a Mac, run the board acceptance from a Mac, or asks why the board is unreachable from the Mac.
---

# Deploy to the PYNQ-Z1 from macOS

One bash script, `references/scripts/pynq_deploy.sh`, does what the exp-board
workflow's board job does, from a Mac on the board's direct Ethernet link.
Vivado is not available on macOS: the overlay comes from a self-hosted build
(an exp-board run artifact), a GitHub Release, or a local directory.

## Workflow

1. Read `references/rules/env.md`.
2. `pynq_deploy.sh preflight`. It names the problem when the Mac has no
   interface on the board's subnet, key login fails, or the board lacks the
   PYNQ venv or the passwordless sudo rule. Fixing the network or installing
   an SSH key is the user's action: tell them what the preflight printed.
3. `pynq_deploy.sh overlay --run <exp-board run id>` for an overlay with the
   ISA front end (ABI 1.1), or `--release vX.Y.Z` / `--dir <path>`.
4. `pynq_deploy.sh export` builds the three packages on this Mac
   (`exp/1007_sw_stack/export_all.sh`; LLVM 22 + the repository venv).
5. `pynq_deploy.sh run` copies overlay, packages, `src/` and `exp/` to
   `/home/xilinx/npu_exp/mac-<time>/`, runs
   `exp/1007_sw_stack/board/run_models.sh` detached, polls, and copies
   `results/` back to `build/deploy/results/mac-<time>/`.

`pynq_deploy.sh all --run <id>` chains the four steps. Add `--dry-run` before
the command to print every action without touching the network.

## Guardrails

- Never change the Mac's network settings, the board image, or sudoers; report
  what preflight found.
- Never store a password, key or token in the repository or the skill.
- A Release overlay without capability bit 6 (`ISA_FRONTEND`, ABI 1.1) cannot
  run compiled packages; the board run fails at `IsaDevice`/runtime start.
- Report results from `build/deploy/results/<id>/results/` only; do not claim
  a board pass without them.
