# Environment Rules

Primary language: bash (macOS system bash 3.2 compatible)
Runtime version: macOS 13+ on Apple silicon (arm64)
Package manager: Homebrew; pip inside <repo>/.venv
Required services: none locally; `gh` authenticated for overlay downloads

- LLVM is found at `/opt/homebrew/opt/llvm/bin`; set `NPU_LLVM_BIN` to use
  another LLVM 22 build with MLIR.
- The venv lives at `<repo>/.venv` (per worktree); it is never committed.
- `npu-opt` builds itself into `<repo>/build/npu-opt` on first use.
- The PYNQ-Z1 is `xilinx@192.168.2.99` on a direct Ethernet link; see
  `deploy-pynq-macos/references/rules/env.md`.
