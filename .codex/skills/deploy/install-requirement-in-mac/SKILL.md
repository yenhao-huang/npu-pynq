---
name: install-requirement-in-mac
description: Install and verify everything a Mac needs to work on the PYNQ-Z1 NPU repository — RTL lint/simulation, the MLIR/LLVM compiler and model export, and deployment to the board — then run those hardware scripts, software scripts and deployments from macOS. Use when setting up a new Mac, when a tool is missing, or when the user wants to run the hardware/software/deploy flow on macOS.
---

# NPU development on macOS

`references/scripts/mac_npu.sh` is the single entry point.

```text
mac_npu.sh doctor              what is installed and what is missing (changes nothing)
mac_npu.sh install [--dry-run] Homebrew tools + Python 3.12 venv at <repo>/.venv
mac_npu.sh hw                  make -C src/test lint sim   (14 testbenches)
mac_npu.sh sw [--full]         ISA/compiler/operator tests; --full exports and runs
                               SmolLM2, Qwen3, ResNet-18 on the host (simulated NPU)
mac_npu.sh export [DIR]        build the three PYNQ packages (Cortex-A9 libraries)
mac_npu.sh deploy ARGS         deploy-pynq-macos: preflight | overlay | run | all
```

## Workflow

1. Run `mac_npu.sh doctor`. Report what is missing.
2. Installing changes the machine: show `mac_npu.sh install --dry-run` and get
   the user's go-ahead, then run `mac_npu.sh install` (it ends with `doctor`).
3. Hardware scripts: `mac_npu.sh hw`. Software scripts: `mac_npu.sh sw`
   (`--full` for the host acceptance, a few minutes and ~1 GB of temporary
   packages).
4. Deployment: `mac_npu.sh deploy preflight`, then
   `mac_npu.sh deploy all --run <exp-board run id>` (or `--release vX.Y.Z`).
   The board must be on a direct Ethernet link to the Mac; preflight explains
   the adapter settings and the one-time `ssh-copy-id`. See the
   `deploy-pynq-macos` skill.

## What gets installed

| Purpose | Tools | Source |
| --- | --- | --- |
| RTL lint and simulation | Verilator, Icarus Verilog, make | Homebrew (`verilator`, `icarus-verilog`), Xcode CLT |
| Compiler | LLVM 22 with MLIR (`mlir-opt`, `mlir-translate`, `llc`, `clang`, MLIR CMake package), CMake | Homebrew (`llvm`, `cmake`) |
| ARM linker | `ld.lld` from the `ziglang` wheel | venv |
| Models, AWQ, references | numpy, torch, torchvision, transformers, safetensors, huggingface_hub, pillow, regex, pyyaml | venv (`references/requirements-mac.txt`) |
| Deployment | ssh, scp, tar, `gh` (artifacts and Releases) | macOS, Homebrew (`gh`) |

Vivado does not run on macOS. Bitstreams come from the self-hosted Windows
runner (`DESKTOP-54U632L`, Vivado 2026.1) as exp-board artifacts, or from a
GitHub Release.

## Guardrails

- `doctor` never changes anything; `install` only adds missing Homebrew
  packages and the repository `.venv` (ignored by Git).
- Never change network settings, SSH keys on the board, or sudoers yourself;
  report the command for the user to run (e.g. `! ssh-copy-id xilinx@192.168.2.99`).
- Report the output of `hw`, `sw` and `deploy` as is; a board result exists
  only when `build/deploy/results/<id>/results/` does.
