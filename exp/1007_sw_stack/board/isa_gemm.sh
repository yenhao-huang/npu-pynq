#!/bin/bash
# Board check for #120: the ABI 1.0 DMA path still works and instruction-stream
# GEMMs match NumPy on the physical NPU.
set -u
cd "$1"
bash exp/1007_sw_stack/board/probe.sh "$1" || true
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
sudo -n XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -B exp/1007_sw_stack/board/isa_gemm.py \
    --overlay overlay/artifacts/npu_matrix.bit --out results/isa_gemm.json
