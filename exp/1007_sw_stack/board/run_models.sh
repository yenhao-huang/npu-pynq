#!/bin/bash
# Board acceptance for #119: the ISA GEMM check, then SmolLM2, Qwen3 and
# ResNet-18 compiled packages on the NPU overlay.
set -u
cd "$1"
code=0
bash exp/1007_sw_stack/board/isa_gemm.sh "$1" || code=1
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
sudo -n XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -B exp/1007_sw_stack/board/run_models.py \
    --overlay overlay/artifacts/npu_matrix.bit --packages export --out results/models.json || code=1
exit $code
