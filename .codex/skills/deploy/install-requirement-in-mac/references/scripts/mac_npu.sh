#!/bin/bash
# Set up a Mac for the NPU repository and run its hardware scripts, software
# scripts and PYNQ-Z1 deployment.
#
#   mac_npu.sh doctor            what is installed, what is missing (changes nothing)
#   mac_npu.sh install [--dry-run]
#                                Homebrew tools + Python 3.12 venv at <repo>/.venv
#   mac_npu.sh hw                RTL lint + simulation (make -C src/test lint sim)
#   mac_npu.sh sw [--full]       ISA/compiler/operator tests; --full adds the host
#                                acceptance (exports and runs SmolLM2, Qwen3, ResNet-18)
#   mac_npu.sh export [DIR]      build the three PYNQ packages (default build/deploy/export)
#   mac_npu.sh deploy ARGS...    pynq_deploy.sh ARGS (preflight | overlay | run | all)
#   mac_npu.sh vivado ARGS...    vivado_mac.sh ARGS: Linux Vivado in Docker (Rosetta),
#                                GUI through Screen Sharing (check | install-gui | gui | build | deploy)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../../../../.." && pwd)
SKILL=$(cd "$(dirname "$0")/.." && pwd)
VENV="$REPO/.venv"
BREW_PACKAGES="llvm cmake verilator icarus-verilog gh python@3.12"
LLVM_BIN=${NPU_LLVM_BIN:-/opt/homebrew/opt/llvm/bin}
DEPLOY="$REPO/.codex/skills/deploy/deploy-pynq-macos/references/scripts/pynq_deploy.sh"

ok() { printf '  ok       %-22s %s\n' "$1" "$2"; }
miss() { printf '  MISSING  %-22s %s\n' "$1" "$2"; MISSING=$((MISSING + 1)); }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

python312() {
    local candidate
    for candidate in /opt/homebrew/opt/python@3.12/bin/python3.12 /opt/homebrew/bin/python3.12 python3.12; do
        if command -v "$candidate" >/dev/null 2>&1; then command -v "$candidate"; return 0; fi
    done
    return 1
}

venv_python() {
    [ -x "$VENV/bin/python" ] || die "no venv at $VENV; run: mac_npu.sh install"
    printf '%s' "$VENV/bin/python"
}

cmd_doctor() {
    MISSING=0
    printf 'macOS %s (%s), repo %s\n' "$(sw_vers -productVersion)" "$(uname -m)" "$REPO"
    echo "hardware scripts"
    if command -v make >/dev/null; then ok make "$(make --version | head -1)"; else miss make "xcode-select --install"; fi
    if command -v verilator >/dev/null; then ok verilator "$(verilator --version | cut -c1-40)"; else miss verilator "brew install verilator"; fi
    if command -v iverilog >/dev/null; then ok iverilog "$(iverilog -V 2>&1 | head -1 | cut -c1-40)"; else miss iverilog "brew install icarus-verilog"; fi
    echo "compiler (MLIR + LLVM 22)"
    for t in mlir-opt mlir-translate llc clang mlir-tblgen; do
        if [ -x "$LLVM_BIN/$t" ]; then ok $t "$LLVM_BIN/$t"; else miss $t "brew install llvm"; fi
    done
    if [ -f "$LLVM_BIN/../lib/cmake/mlir/MLIRConfig.cmake" ]; then ok "MLIR CMake package" "$LLVM_BIN/../lib/cmake/mlir"; else miss "MLIR CMake package" "brew install llvm"; fi
    if command -v cmake >/dev/null; then ok cmake "$(cmake --version | head -1)"; else miss cmake "brew install cmake"; fi
    echo "python"
    if p=$(python312); then ok python3.12 "$p"; else miss python3.12 "brew install python@3.12"; fi
    if [ -x "$VENV/bin/python" ]; then
        ok venv "$VENV"
        local mods
        mods=$("$VENV/bin/python" - <<'PY'
import importlib
for m in ("numpy", "torch", "torchvision", "transformers", "safetensors", "huggingface_hub", "PIL", "yaml", "regex", "ziglang"):
    try:
        mod = importlib.import_module(m)
        print(f"ok {m} {getattr(mod, '__version__', '')}")
    except Exception:
        print(f"missing {m}")
PY
)
        while read -r state name version; do
            if [ "$state" = ok ]; then ok "$name" "$version"; else miss "$name" "mac_npu.sh install"; fi
        done <<< "$mods"
    else
        miss venv "mac_npu.sh install"
    fi
    echo "deployment"
    for t in ssh scp tar gh; do
        if command -v $t >/dev/null; then ok $t "$(command -v $t)"; else miss $t "brew install gh"; fi
    done
    if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then ok "gh auth" "logged in"; else miss "gh auth" "gh auth login"; fi
    if ls ~/.ssh/id_* >/dev/null 2>&1; then ok "ssh key" "$(ls ~/.ssh/id_*.pub 2>/dev/null | head -1)"; else miss "ssh key" "ssh-keygen -t ed25519; ssh-copy-id xilinx@192.168.2.99"; fi
    echo "Vivado: no macOS build; run it in Docker with: mac_npu.sh vivado check"
    if [ "$MISSING" -eq 0 ]; then echo "PASS: doctor"; else echo "$MISSING missing; run: mac_npu.sh install"; return 1; fi
}

cmd_install() {
    local dry=0
    [ "${1:-}" = --dry-run ] && dry=1
    run() { if [ "$dry" = 1 ]; then echo "dry-run: $*"; else "$@"; fi; }
    command -v brew >/dev/null || die "Homebrew is required: https://brew.sh"
    local missing=""
    for pkg in $BREW_PACKAGES; do
        brew list --versions "$pkg" >/dev/null 2>&1 || missing="$missing $pkg"
    done
    if [ -n "$missing" ]; then run brew install $missing; else echo "Homebrew packages present: $BREW_PACKAGES"; fi
    local py
    py=$(python312 || echo /opt/homebrew/opt/python@3.12/bin/python3.12)
    [ -x "$VENV/bin/python" ] || run "$py" -m venv "$VENV"
    run "$VENV/bin/python" -m pip install --upgrade pip
    run "$VENV/bin/python" -m pip install -r "$SKILL/requirements-mac.txt"
    [ "$dry" = 1 ] || cmd_doctor
}

cmd_hw() {
    make -C "$REPO/src/test" lint sim PYTHON="$(venv_python)"
}

cmd_sw() {
    local py
    py=$(venv_python)
    export NPU_LLVM_BIN="$LLVM_BIN"
    (cd "$REPO" && PYTHONPATH=. "$py" -m unittest src.test.tests.test_isa src.test.tests.test_compiler src.test.tests.test_ops -v)
    if [ "${1:-}" = --full ]; then
        (cd "$REPO" && PYTHONPATH=. "$py" exp/1007_sw_stack/host/acceptance.py --json build/host_acceptance.json)
    fi
}

cmd_export() {
    local out=${1:-$REPO/build/deploy/export} py
    py=$(venv_python)
    export NPU_LLVM_BIN="$LLVM_BIN"
    (cd "$REPO" && PATH="$(dirname "$py"):$PATH" bash exp/1007_sw_stack/export_all.sh "$out")
}

cmd_vivado() {
    "$REPO/.codex/skills/deploy/vivado-on-mac/references/scripts/vivado_mac.sh" "$@"
}

cmd_deploy() {
    [ -x "$DEPLOY" ] || die "missing $DEPLOY"
    PATH="$VENV/bin:$PATH" NPU_LLVM_BIN="$LLVM_BIN" "$DEPLOY" "$@"
}

[ $# -gt 0 ] || { sed -n '2,18p' "$0"; exit 2; }
command=$1; shift
case "$command" in
    doctor|install|hw|sw|export|deploy|vivado) "cmd_$command" "$@" ;;
    -h|--help) sed -n '2,18p' "$0" ;;
    *) die "unknown command $command" ;;
esac
