#!/usr/bin/env bash
set -euo pipefail
TASK_LIB=/opt/ort-build/MinSizeRel
test -f "$TASK_LIB/libonnxruntime.so.1.20.1"
arm-linux-gnueabihf-gcc -std=c11 -O2 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard \
  -I /opt/ort-linux-source/include/onnxruntime/core/session \
  /work/onnx_ort_full_reference.c -L "$TASK_LIB" -l:libonnxruntime.so.1.20.1 \
  -Wl,-rpath,"$TASK_LIB" -o /work/official_arm_oracle
qemu-arm -cpu cortex-a9 -L /usr/arm-linux-gnueabihf \
  /work/official_arm_oracle /work/model_int8.onnx /work/prompt_ids.bin /work/oracle-arm \
  | tee /work/arm-oracle.log
echo ARM_ORACLE_COMPLETE
