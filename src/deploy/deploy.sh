#!/usr/bin/env bash
# Send an export directory to the PYNQ-Z1, with the runtime it needs.
#
#   src/deploy/deploy.sh dist/smollm [files to add, such as a notebook]
#
# <dir> holds package/ (the export) and overlay/ (the bitstream). This copies
# the runtime and the NPU driver sources into <dir>/src, adds the files, and
# copies <dir> to ~/jupyter_notebooks/ on the board. What only the host uses
# stays behind: other CPUs' libraries, ir/, compile.json and *.pth weights.
# BOARD overrides the target (default xilinx@192.168.2.99). ARCHIVE=<file.tar.gz>
# writes the same tree to that file instead of sending it; a release ships it.
set -euo pipefail

dir=${1:?usage: deploy.sh <dir> [files...]}
shift
repo=$(cd "$(dirname "$0")/../.." && pwd)
board=${BOARD:-xilinx@192.168.2.99}

rm -rf "$dir/src"
mkdir -p "$dir/src/vendors/npu_pynq" "$dir/src/vendors/risc_v_npu"
cp -r "$repo/src/runtime" "$dir/src/"
cp "$repo/src/vendors/__init__.py" "$dir/src/vendors/"
for vendor in npu_pynq risc_v_npu; do
  cp -r "$repo/src/vendors/$vendor/__init__.py" "$repo/src/vendors/$vendor/driver" \
        "$repo/src/vendors/$vendor/spec" "$dir/src/vendors/$vendor/"
done
find "$dir/src" -name __pycache__ -prune -exec rm -rf {} +
if [ $# -gt 0 ]; then cp "$@" "$dir/"; fi

name=$(basename "$dir")
skip=(--exclude="$name/package/ir" --exclude="$name/package/compile.json" --exclude="*.pth")
for lib in "$dir"/package/*/model.so; do
  cpu=$(basename "$(dirname "$lib")")
  if [ -f "$lib" ] && [ "$cpu" != arm-cortex-a9 ]; then skip+=(--exclude="$name/package/$cpu"); fi
done

if [ -n "${ARCHIVE:-}" ]; then
  mkdir -p "$(dirname "$ARCHIVE")"
  archive="$(cd "$(dirname "$ARCHIVE")" && pwd)/$(basename "$ARCHIVE")"
  tar -C "$(dirname "$dir")" "${skip[@]}" -czf "$archive" "$name"
else
  tar -C "$(dirname "$dir")" "${skip[@]}" -cf - "$name" |
    ssh "$board" 'mkdir -p jupyter_notebooks && tar -xmf - -C jupyter_notebooks'
fi
