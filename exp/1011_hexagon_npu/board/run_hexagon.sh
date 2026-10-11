#!/bin/bash
# exp-board entry point for #136: the Hexagon subsystem self-test (scalar,
# vector, tensor GEMM), then SmolLM2 on the same overlay through the tensor
# accelerator. Expects overlay/artifacts/npu_matrix.bit (built with
# --hexagon) and export/smollm (exp/1011_hexagon_npu/export_smollm.sh).
set -u
cd "$1"
mkdir -p results
code=0
bash exp/1011_hexagon_npu/board/hx_selftest.sh "$1" || code=1
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
sudo -n XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -B exp/1007_sw_stack/board/run_models.py \
    --overlay overlay/artifacts/npu_matrix.bit --packages export --models smollm \
    --out results/smollm.json || code=1
exit $code
