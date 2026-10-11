#!/usr/bin/env bash
# Rebuild the environment the exp/ab_test/ A/B experiments ran in (macOS arm64).
# Idempotent: every step skips itself when its result already exists.
#
# Layout it creates or uses:
#   $REPO/exp/ab_test         A/B experiment in any npu-pynq checkout/worktree
#   $AB_TOOL_REPO             detached worktree at the pinned IC tools commit
#   ~/.ab_private/RTLLM         task set, pinned, kept out of the agent's cwd
#   ~/pdk                       Nangate45 liberty + LEF
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
COMMON_GIT="$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)"
NPU="$(dirname "$(dirname "$COMMON_GIT")")"
HERE="$REPO/exp/ab_test/setup"
WT="${AB_TOOL_REPO:-$NPU/worktrees/npu-ab-tools-cdb514f}"
TOOL_PIN="cdb514f7f1c2e5b15a0890d6ac4a3353d79a3e90"
RTLLM_PIN="9d5f61a"          # RTLLM v2.0
LIBFST_PIN="c716c1f"
IMG="openroad/orfs@sha256:c950756157f7cd1b99229e1258586a270758d3f1778e0d4478e7287d4b450445"
DOCKER_BIN="/Applications/Docker.app/Contents/Resources/bin"
export PATH="$HOME/.local/bin:$DOCKER_BIN:/opt/homebrew/bin:$PATH"
step() { printf '\n== %s\n' "$*"; }

step "1. Homebrew toolchain"
brew list verilator >/dev/null 2>&1 || brew install verilator
brew list yosys >/dev/null 2>&1 || brew install yosys
brew list icarus-verilog >/dev/null 2>&1 || brew install icarus-verilog
brew list lz4 >/dev/null 2>&1 || brew install lz4   # Verilator's FST writer needs lz4.h
command -v uv >/dev/null || brew install uv

step "2. Worktree with the pinned IC tools ($TOOL_PIN)"
git -C "$REPO" fetch -q origin
if [ ! -d "$WT" ]; then
  git -C "$REPO" worktree add --detach "$WT" "$TOOL_PIN"
fi
if [ "$(git -C "$WT" rev-parse HEAD)" != "$TOOL_PIN" ]; then
  echo "$WT does not point to the experiment's pinned IC tools commit $TOOL_PIN" >&2
  exit 1
fi
if [ ! -x "$WT/.venv/bin/ic" ]; then
  uv venv -q "$WT/.venv" --python 3.11
  uv pip install -q --python "$WT/.venv/bin/python" -e "$WT/tools/ic[all]" pytest pytest-asyncio
  git -C "$WT" checkout -q -- tools/ic/ic_design_tools.egg-info 2>/dev/null || true
fi

step "3. RTLLM v2.0 ($RTLLM_PIN)"
if [ ! -d "$HOME/.ab_private/RTLLM" ]; then
  mkdir -p "$HOME/.ab_private"
  git clone -q https://github.com/hkust-zhiyao/RTLLM "$HOME/.ab_private/RTLLM"
fi
git -C "$HOME/.ab_private/RTLLM" checkout -q "$RTLLM_PIN"
git -C "$HOME/.ab_private/RTLLM" log --oneline -1

step "4. fst2vcd (native arm64; the GTKWave cask binary is quarantined x86_64 and hangs)"
mkdir -p "$HOME/.local/bin"
if [ ! -x "$HOME/.local/bin/fst2vcd" ]; then
  tmp="$(mktemp -d)"
  git clone -q https://github.com/gtkwave/libfst "$tmp/libfst"
  git -C "$tmp/libfst" checkout -q "$LIBFST_PIN"
  cc -O2 -I"$tmp/libfst/src" "$HERE/fst2vcd_min.c" \
     "$tmp/libfst/src/fstapi.c" "$tmp/libfst/src/fastlz.c" "$tmp/libfst/src/lz4.c" -lz \
     -o "$HOME/.local/bin/fst2vcd"
fi

step "5. OpenROAD: pinned ORFS image in one long-lived container with no host mounts"
"$DOCKER_BIN/docker" version --format 'docker {{.Server.Version}}' \
  || { echo "Start Docker Desktop first (open -a Docker)"; exit 1; }
"$DOCKER_BIN/docker" image inspect "$IMG" >/dev/null 2>&1 \
  || "$DOCKER_BIN/docker" pull --platform linux/amd64 "$IMG"
if [ ! -f "$HOME/pdk/NangateOpenCellLibrary_typical.lib" ]; then
  mkdir -p "$HOME/pdk"
  "$DOCKER_BIN/docker" run --rm --platform linux/amd64 -v "$HOME/pdk:/out" "$IMG" sh -c \
    'cp /OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib /out/ &&
     cp -r /OpenROAD-flow-scripts/flow/platforms/nangate45/lef /out/' </dev/null
fi
if ! "$DOCKER_BIN/docker" ps --format '{{.Names}}' | grep -qx ic-openroad; then
  "$DOCKER_BIN/docker" rm -f ic-openroad >/dev/null 2>&1 || true
  "$DOCKER_BIN/docker" run -d --name ic-openroad --platform linux/amd64 "$IMG" sleep infinity </dev/null
  "$DOCKER_BIN/docker" cp "$HOME/pdk" ic-openroad:/pdk
fi
install -m 755 "$HERE/openroad_shim.py" "$HOME/.local/bin/openroad"
openroad -version

step "6. Environment file"
cat > "$REPO/exp/ab_test/.ab_env" <<EOF
export AB_TOOL_REPO=$WT
export PATH=$HOME/.local/bin:$WT/.venv/bin:$DOCKER_BIN:\$PATH
export IC_PDK_LIBERTY=$HOME/pdk/NangateOpenCellLibrary_typical.lib
export IC_PDK_TECH_LEF=$HOME/pdk/lef/NangateOpenCellLibrary.tech.lef
export IC_PDK_LEF=$HOME/pdk/lef/NangateOpenCellLibrary.macro.mod.lef
export CPATH=/opt/homebrew/include
export LIBRARY_PATH=/opt/homebrew/lib
EOF
echo "wrote $REPO/exp/ab_test/.ab_env"

step "7. Smoke test"
# shellcheck disable=SC1091
source "$REPO/exp/ab_test/.ab_env"
export IC_ROOT="$(mktemp -d)"
cd "$WT"
ic --compact ppa --files src/hw/rtl/systolic_array/npu_pe.sv --top npu_pe --clock-port clk \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("ppa ok", d["ok"], "adp", d["adp"], "fmax", d["timing"]["fmax_mhz"]); assert d["ok"] and d["adp"]'
# Check auth the way the harnesses call claude: without the parent session's CLAUDE_CODE_* variables.
unset_args=(); for v in $(env | grep -o '^CLAUDE_CODE_[A-Z_]*\|^CLAUDECODE'); do unset_args+=(-u "$v"); done
env "${unset_args[@]}" claude auth status | grep -q '"loggedIn": true' \
  || echo "WARNING: claude is not logged in; run 'claude auth login' in a normal terminal before the experiments"
echo "setup complete"
