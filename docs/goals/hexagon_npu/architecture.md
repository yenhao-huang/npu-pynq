# Hexagon-style heterogeneous NPU: architecture and interfaces

Issue #136. Goal: [goal.md](goal.md). Reference: Qualcomm Hexagon NPU, Hot
Chips 2023 (E. Mahurin), summarised below; block diagram
[hexagon_npu.png](hexagon_npu.png).

## What Hexagon does that the current NPU does not

| Hexagon (HC2023) | Current PYNQ NPU (`npu/issue119-a`) |
| --- | --- |
| Scalar ISA for control flow, with I$ and D$ | none: the ARM sequences jobs; the ISA front end only decodes GEMM programs |
| Vector SIMD (8/16/32-bit int, 16/32-bit float), load/store to L2 or TCM | none: elementwise ops, requantisation, activations run on the Cortex-A9 |
| Tensor unit, int 8/16 x 4/8/16, FP16 x FP16 | 16 x 16 INT8 systolic array, 32-bit saturating accumulators |
| TCM shared by vector and tensor | per-row/column operand banks inside the matrix datapath |
| L2 cache in front of DDR | none |
| DMA/BUS | `dma_axi` register file + AXI DMA streams + the ISA front end's AXI4 master |

## Target organisation

```text
npu_hexagon (new top, replaces npu_accelerator as the block-design module)
 |-- npu_accelerator            existing, unchanged interface: AXI-lite regs (dma_axi),
 |                              ISA front end, operand TCM, 16 x 16 INT8 tensor array
 `-- npu_hx_subsystem           new, its own AXI-lite window and AXI4 master
      |-- npu_hx_regs           control/status: scalar start PC, halt, perf counters
      |-- npu_scalar_core       RV32I subset, multi-cycle; custom-0 dispatches vector ops
      |     |-- I-port -> npu_cache (L1 I$, read-only)
      |     `-- D-port -> address decode: DDR -> npu_cache (L1 D$), TCM window, MMIO window
      |-- npu_mem_arbiter       L1 I$ + L1 D$ line requests -> L2
      |-- npu_cache (L2)        unified, same cache core with line-wide CPU side
      |-- npu_axi4_line_bridge  L2 line refill/write-back -> AXI4 bursts
      |-- npu_tcm               dual-port BRAM scratchpad, VLEN-bit words
      |-- npu_tcm_dma           DDR <-> TCM block copies, its own AXI4 read/write
      |-- npu_axi4_arbiter      bridge + DMA -> one AXI4 master
      `-- npu_vector_unit       VRF + SIMD lanes (INT8/16/32, FP16), load/store to TCM
           `-- npu_fp16_add / npu_fp16_mul
```

Tensor-side data widths (INT16, FP16) are a separate, parameterised tensor
unit: `npu_pe_mp` and `npu_systolic_array_mp` (multi-precision PE and array).
They are verified and PPA-measured stand-alone. The deployed 16 x 16 overlay
keeps the INT8 array: on the Zynq-7020 the INT8 array already uses 194/220
DSPs and 84% of the slices, and an FP16 PE costs several times an INT8 PE.

### Modularity rules

- Every unit is a separate module with a valid/ready (or req/gnt) boundary.
  No unit reaches into another unit's internals.
- Each unit has its own testbench `src/hw/tb/npu_matrix/tb_<module>.sv`.
- The subsystem top is the only place where units are wired together, and
  each unit can be removed by a parameter (`HX_ENABLE_*`) for PPA deltas.
- New RTL lives under `src/hw/rtl/npu_matrix/npu_hexagon/<unit>/`.

## Numeric contract

| Type | Code | Lanes in 64-bit | Semantics |
| --- | --- | --- | --- |
| INT8 | 0 | 8 | two's complement |
| INT16 | 1 | 4 | two's complement |
| INT32 | 2 | 2 | two's complement |
| FP16 | 3 | 4 | IEEE 754 binary16 |

- Integer ADD/SUB saturate to the lane type. MUL keeps the low half (wraps).
  SRA is an arithmetic right shift with round-half-up: `(x + (1 << (s-1))) >>> s`
  for `s > 0`, then saturate (no-op for `s = 0`).
- FP16 ADD/SUB/MUL: IEEE 754 binary16, round-to-nearest-even, subnormal inputs
  and outputs supported (no flush to zero), overflow to +-Inf, any NaN result
  is the canonical quiet NaN `0x7E00`. MAX/MIN: if either operand is NaN the
  result is `0x7E00`; `max(-0, +0) = +0`, `min(-0, +0) = -0`.
- FP32 (tensor FP16 accumulation only): IEEE 754 binary32, RNE, subnormals
  supported, canonical NaN `0x7FC00000`.
- Golden model: `src/test/model/hx_numeric.py` (numpy `float16`/`float32` give
  correctly rounded results for these cases: an FP16 x FP16 product is exact in
  FP32, an FP16 + FP16 sum is exact in FP64 before the final rounding).

## Interfaces

All modules: `clk`, active-low asynchronous reset `rst_n` (repository style,
see `npu_pe.sv`). Storage arrays have no reset so Vivado can infer BRAM.

### Numerics (`npu_hexagon/numeric/`)

```systemverilog
module npu_fp16_mul (input clk, rst_n, in_valid, input [15:0] a, b,
                     output out_valid, output [15:0] y);   // fixed latency LAT_MUL = 2
module npu_fp16_add (input clk, rst_n, in_valid, input [15:0] a, b, input sub,
                     output out_valid, output [15:0] y);   // fixed latency LAT_ADD = 3
module npu_fp32_add (same shape, 32-bit operands, LAT = 3)
```

### Multi-precision tensor PE and array (`npu_hexagon/tensor_mp/`)

`npu_pe_mp`: same systolic dataflow and handshake as `npu_pe`, 16-bit operand
lanes, runtime `prec[1:0]` (0 = INT8 operands in the low byte, 32-bit
saturating accumulator exactly as `npu_pe`; 1 = INT16, 48-bit saturating
accumulator; 2 = FP16 x FP16 with FP32 accumulation). `accumulator[47:0]`
(FP32 result in bits [31:0] in FP16 mode). `npu_systolic_array_mp`: ROWS x
COLUMNS of `npu_pe_mp`, same edge wiring as `npu_systolic_array`.

### Vector unit (`npu_hexagon/vector/`)

Parameters: `VLEN = 64` (bits), `NREGS = 8`, `TCM_AW` (TCM word-address width).

```systemverilog
// command from the scalar core (custom-0), one command per handshake
input  cmd_valid, output cmd_ready,
input  [4:0] cmd_op, input [1:0] cmd_type,
input  [2:0] cmd_vd, cmd_vs1, cmd_vs2,
input  [TCM_AW-1:0] cmd_addr, input [31:0] cmd_scalar,
output busy,                                   // commands in flight
// TCM port A (1-cycle read latency after a granted read)
output tcm_req, output tcm_we, output [TCM_AW-1:0] tcm_addr,
output [VLEN-1:0] tcm_wdata, input tcm_gnt, input [VLEN-1:0] tcm_rdata
```

Opcodes: `VLD=0` (vd <- TCM[addr]), `VST=1` (TCM[addr] <- vs1), `VSPLAT=2`
(every lane of vd <- scalar, truncated to the lane), `VADD=3`, `VSUB=4`,
`VMUL=5`, `VMAX=6`, `VMIN=7`, `VRELU=8` (vd <- max(vs1, 0)), `VSRA=9` (int
only, shift = scalar[4:0]), `VMOV=10` (vd <- vs1). Others are illegal and set
a sticky `illegal` output. Commands complete in order; a later command reads
the result of an earlier one (the unit stalls on hazards).

### TCM and TCM DMA (`npu_hexagon/memory/`)

`npu_tcm #(WIDTH=64, DEPTH=1024)`: true dual port; each port `req, we,
addr, wdata, wstrb[WIDTH/8], gnt (= req), rdata` valid the cycle after a read.
`npu_tcm_dma`: descriptor `start, dir (0 = DDR->TCM, 1 = TCM->DDR), ddr_addr
[31:0] (8-byte aligned), tcm_addr, words[15:0]`, `busy, done (pulse), error`;
AXI4 master read and write channels, 64-bit data, INCR bursts of up to 16
beats, never crossing a 4 KB boundary.

### Caches (`npu_hexagon/cache/`)

`npu_cache #(SIZE_BYTES, LINE_BYTES, WAYS = 1 or 2, CPU_W = 32, WRITE_BACK = 1,
READ_ONLY = 0)`.

```systemverilog
// CPU side
input  req_valid, output req_ready, input [31:0] req_addr, input req_we,
input  [CPU_W/8-1:0] req_wstrb, input [CPU_W-1:0] req_wdata,
output resp_valid, output [CPU_W-1:0] resp_rdata,
input  flush,  output flush_done,   // write back dirty lines, then invalidate
// memory side, whole lines
output mem_req_valid, input mem_req_ready, output [31:0] mem_req_addr,
output mem_req_we, output [LINE_BYTES*8-1:0] mem_req_wdata,
input  mem_resp_valid, input [LINE_BYTES*8-1:0] mem_resp_rdata
// counters: hits, misses (32-bit, saturating)
```

L1 I$ = `npu_cache #(.SIZE_BYTES(1024), .LINE_BYTES(32), .READ_ONLY(1))`;
L1 D$ = `npu_cache #(.SIZE_BYTES(1024), .LINE_BYTES(32), .WRITE_BACK(1))`;
L2 = `npu_cache #(.SIZE_BYTES(8192), .LINE_BYTES(32), .WAYS(2), .CPU_W(256))`.
`npu_mem_arbiter`: two line-request masters, round-robin, responses routed
back in order. `npu_axi4_line_bridge`: one line request -> one AXI4 burst of
LINE_BYTES/8 beats (read or write, waits for B).

### Scalar core (`npu_hexagon/scalar/`)

`npu_scalar_core`: RV32I minus FENCE/CSR, multi-cycle (no pipeline). `start`
with `start_pc`; `halted` after ECALL or EBREAK, `fault` on an illegal or
misaligned access. Instruction port and data port use the cache CPU-side
protocol (`req_valid/req_ready/req_addr/req_we/req_wstrb/req_wdata`,
`resp_valid/resp_rdata`). Custom-0 (`opcode = 7'b0001011`, R-type fields) sends a vector
command on a `vcmd_*` port shaped like the vector unit's `cmd_*` port:

| Vector command field | Instruction bits |
| --- | --- |
| `op` | `funct7[4:0]` (`inst[29:25]`) |
| `type` | `funct3[1:0]` (`inst[13:12]`) |
| `vd` | `rd[2:0]` (`inst[9:7]`) |
| `vs1` | `rs2[2:0]` (`inst[22:20]`) |
| `vs2` | `{funct7[6:5], funct3[2]}` (`{inst[31:30], inst[14]}`) |
| `scalar` | value of register `rs1` |
| `addr` | value of register `rs1`, low `TCM_AW` bits (TCM word address) |

`op = 31` (`VFENCE`) is not sent: the core waits until `vec_busy` is low.
The core stalls while `vcmd_ready` is low.

Address map seen by the scalar core:

| Range | Target |
| --- | --- |
| `0x0000_0000`-`0x3FFF_FFFF` | DDR through L1 D$ / L1 I$ and L2 |
| `0x4000_0000`-`0x4000_FFFF` | TCM port B, uncached, 32-bit accesses |
| `0x8000_0000`-`0x8000_00FF` | MMIO: TCM DMA descriptor, vector status, cache flush |

## PPA method

| Metric | Source |
| --- | --- |
| Area | Vivado 2026.1 out-of-context synthesis + implementation, `xc7z020clg400-1`: LUT, FF, DSP, BRAM |
| Performance | Fmax from the routed OOC timing summary (10 ns constraint), and cycles per operation from the testbench; throughput = ops/cycle x Fmax |
| Power | Vivado `report_power`, vectorless, default toggle rates, OOC routed design |

Local inner loop: `ic synth --mode estimate` (Yosys `synth_xilinx`) for rough
area. Every PPA table records which source produced it. "Before" for a unit
is the subsystem configuration without it (`HX_ENABLE_*`), so each table
shows the delta that unit adds.
