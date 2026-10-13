#!/bin/bash
# Board acceptance for #136: identify npu_hx_0, run a scalar DDR program and
# the INT8 / FP16 vector kernels against the reference models, then the
# tensor ISA GEMM check, all on the --hexagon overlay. Writes
# results/hx_selftest.json (and results/isa_gemm.json).
set -u
cd "$1"
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
sudo -n XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -B exp/1011_hexagon_npu/board/hx_selftest.py \
    --overlay overlay/artifacts/npu_matrix.bit --out results/hx_selftest.json \
    --tensor-out results/isa_gemm.json
