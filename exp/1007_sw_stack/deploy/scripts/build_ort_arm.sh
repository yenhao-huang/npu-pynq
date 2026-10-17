#!/usr/bin/env bash
set -euo pipefail
TASK_REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
bash "$TASK_REPO/src/runtime/build_onnxruntime_arm.sh"
