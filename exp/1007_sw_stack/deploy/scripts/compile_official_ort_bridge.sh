#!/usr/bin/env bash
set -euo pipefail
TASK_DEPLOY=$(cd "$(dirname "$0")/.." && pwd)
TASK_REPO=$(cd "$TASK_DEPLOY/../../.." && pwd)
bash "$TASK_REPO/src/runtime/build_onnxruntime_bridge.sh" "$TASK_DEPLOY/packages"
