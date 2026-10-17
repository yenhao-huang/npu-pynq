#!/usr/bin/env bash
set -euo pipefail
TASK_MODEL=${1:-${NPU_ONNX_MODEL:-}}
if [ -z "$TASK_MODEL" ]; then
  echo "usage: bash src/runtime/build_onnxruntime_bridge.sh MODEL_DIR" >&2
  exit 2
fi
TASK_SOURCE=$(cd "$(dirname "$0")" && pwd)
TASK_ORT="$TASK_MODEL/onnx-runtime"
test -f "$TASK_ORT/libonnxruntime.so.1.20.1"
test -f "$TASK_ORT/onnxruntime_c_api.h"
ln -sfn libonnxruntime.so.1.20.1 "$TASK_ORT/libonnxruntime.so.1"
gcc -std=c11 -O2 -shared -fPIC -Wall -Wextra \
  -I "$TASK_ORT" "$TASK_SOURCE/c/onnx_ort_bridge.c" \
  -L "$TASK_ORT" -l:libonnxruntime.so.1.20.1 -Wl,-rpath,'$ORIGIN' \
  -o "$TASK_ORT/libonnx_ort_bridge.so"
ldd "$TASK_ORT/libonnx_ort_bridge.so"
