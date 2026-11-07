#!/usr/bin/env bash
# Send an export directory to the PYNQ-Z1, with the runtime it needs.
#
#   src/deploy/deploy.sh dist/smollm [files to add, such as a notebook]
#
# <dir> holds package/ (the export) and overlay/ (the bitstream). This copies
# the runtime and the NPU driver sources into <dir>/src, adds the files, and
# copies <dir> to ~/jupyter_notebooks/ on the board. BOARD overrides the
# target (default xilinx@192.168.2.99).
set -euo pipefail

dir=${1:?usage: deploy.sh <dir> [files...]}
shift
repo=$(cd "$(dirname "$0")/../.." && pwd)
board=${BOARD:-xilinx@192.168.2.99}

rm -rf "$dir/src"
mkdir -p "$dir/src/vendors/npu_pynq"
cp -r "$repo/src/runtime" "$dir/src/"
cp "$repo/src/vendors/__init__.py" "$dir/src/vendors/"
cp -r "$repo/src/vendors/npu_pynq/__init__.py" "$repo/src/vendors/npu_pynq/driver" \
      "$repo/src/vendors/npu_pynq/spec" "$dir/src/vendors/npu_pynq/"
find "$dir/src" -name __pycache__ -prune -exec rm -rf {} +
if [ $# -gt 0 ]; then cp "$@" "$dir/"; fi

scp -r "$dir" "$board:jupyter_notebooks/"
