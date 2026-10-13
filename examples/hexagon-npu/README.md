# Hexagon-style heterogeneous NPU (#136)

[hexagon_npu.ipynb](hexagon_npu.ipynb) is the hands-on check of the units
added by issue #136, on the PYNQ-Z1:

1. load the `--hexagon` overlay and identify `npu_hx_0` (MAGIC, VERSION, CAPS);
2. run an RV32I sort program on the scalar core through the L1/L2 caches,
   checked against numpy and the reference ISS (cycles, CPI, hit rates);
3. run INT8 saturating add and FP16 a*x+y on the vector unit through the TCM
   DMA, bit-exact against numpy, with NPU kernel time vs numpy on the A9;
4. stop a program that never ends (CONTROL.STOP) and resume it;
5. run SmolLM2 on the same overlay.

Design, register map and results: `docs/goals/hexagon_npu/`
(`architecture.md`, `Report.md`, `board.md`).

## What the notebook needs

The notebook looks for these next to itself (or up the directory tree):

```text
hexagon_npu.ipynb
src/                         the repository's src/ (runtime, assembler, ISS, inference)
overlay/npu_matrix.bit       built with build_overlay.tcl --hexagon
overlay/npu_matrix.hwh
export/smollm/               optional: exp/1011_hexagon_npu/export_smollm.sh output
```

In a repository checkout it also finds
`build/vivado/npu_matrix_16x16_hx/artifacts/npu_matrix.bit`.

## 1. Build the overlay and the SmolLM2 package

```bash
vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl -tclargs --hexagon
bash exp/1011_hexagon_npu/export_smollm.sh build/exp-payload
```

or reuse the artifacts of an exp-board run of `npu/issue136-a`
(`exp-overlay-<id>` and `exp-export-<id>`; see
`docs/goals/hexagon_npu/Reproduce.md`). Bitstreams are never committed.

## 2. Put it on the board

```bash
board=xilinx@192.168.2.99
dir=/home/xilinx/jupyter_notebooks/hexagon_npu
ssh $board "mkdir -p $dir/overlay $dir/export"
scp examples/hexagon-npu/hexagon_npu.ipynb $board:$dir/
scp build/vivado/npu_matrix_16x16_hx/artifacts/npu_matrix.{bit,hwh} $board:$dir/overlay/
scp -r src $board:$dir/
scp -r build/exp-payload/smollm $board:$dir/export/
```

## 3. Run

Open `http://192.168.2.99:9090/tree`, go to `hexagon_npu/` and run
`hexagon_npu.ipynb` top to bottom. Every check cell asserts its result and
prints `PASS`; the vector timing cell prints the NPU kernel time next to the
whole Python call and numpy on the Cortex-A9.

On the reference run (exp-board run 38329269229) the sort took 57,311 cycles
(CPI 3.87), the 512-element INT8 add kernel 1,447 cycles (14.5 us) and the
FP16 a*x+y kernel 3,231 cycles, and SmolLM2 answered "The capital of France is
Paris..." at 1.45 tok/s decode.
