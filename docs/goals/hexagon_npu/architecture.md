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
npu_matrix block design (src/hw/vivado_tcl/npu_matrix/build_overlay.tcl --hexagon)
 |-- npu_accelerator_0          existing, unchanged: AXI-lite regs (dma_axi), ISA front
 |                              end, operand TCM, 16 x 16 INT8 tensor array
 `-- npu_hx_0 = npu_hx_subsystem  new cell: its own AXI-lite window and AXI4 master (HP0)
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

The block design is the only place the two cells meet: both are AXI-lite
slaves of the PS GP0 interconnect and AXI4 masters of the HP0 interconnect, so
the tensor path, its register map and the runtime that drives it stay
unchanged, and the overlay without `--hexagon` is the PPA "before" design.

Tensor-side data widths (INT16, FP16) are a separate, parameterised tensor
unit: `npu_pe_mp` and `npu_systolic_array_mp` (multi-precision PE and array).
They are verified and PPA-measured stand-alone. The deployed 16 x 16 overlay
keeps the INT8 array: on the Zynq-7020 the INT8 array already uses 194/220
DSPs and 84% of the slices, and an FP16 PE costs several times an INT8 PE.

### Modularity rules

- Every unit is a separate module with a valid/ready (or req/gnt) boundary.
  No unit reaches into another unit's internals.
- Each unit has its own testbench `src/hw/tb/npu_matrix/tb_<module>.sv`.
- `npu_hx_subsystem` is the only place where the new units are wired
  together, and each unit can be removed by a parameter (`HX_ENABLE_*`) for
  PPA deltas.
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
READ_ONLY = 0, CPU_HOLD = 0, MEM_W = CPU_W, RESP_W = CPU_W)`.

```systemverilog
// CPU side (a read answers with CPU_W/RESP_W resp_valid beats, a write with one)
input  req_valid, output req_ready, input [31:0] req_addr, input req_we,
input  [CPU_W/8-1:0] req_wstrb, input [CPU_W-1:0] req_wdata,
output resp_valid, output [RESP_W-1:0] resp_rdata,
input  flush,  output flush_done,   // write back dirty lines, then invalidate
// memory side: whole-line requests, read data in MEM_W-bit beats
output mem_req_valid, input mem_req_ready, output [31:0] mem_req_addr,
output mem_req_we, output [LINE_BYTES*8-1:0] mem_req_wdata,
input  mem_resp_valid, input [MEM_W-1:0] mem_resp_rdata
// counters: hits, misses (32-bit, saturating)
```

Line protocol (cache memory side, both `npu_mem_arbiter` sides, the L2 CPU
side, `npu_axi4_line_bridge`): one request outstanding; the requester holds
addr, we and the whole-line wdata through its last response pulse. A write
completes with one `mem_resp_valid` pulse; a read with LINE_BYTES*8/W pulses
(beats, lowest address first, gaps allowed, no ready), W = 32 between the
L1s and the L2, W = 64 (the AXI beat) between the L2 and the bridge. Refill
beats go straight into the data RAM, so the L1s need no line-wide refill
register.

L1 I$ = `npu_cache #(.SIZE_BYTES(1024), .LINE_BYTES(32), .READ_ONLY(1))`;
L1 D$ = `npu_cache #(.SIZE_BYTES(1024), .LINE_BYTES(32), .WRITE_BACK(1))`
(`.CPU_HOLD(1)` is allowed: the scalar core holds its data request through
the response);
L2 = `npu_cache #(.SIZE_BYTES(8192), .LINE_BYTES(32), .WAYS(2), .CPU_W(256),
.CPU_HOLD(1), .MEM_W(64), .RESP_W(32))`.
`npu_mem_arbiter #(.BEAT_W(32), .P0_READ_ONLY(1))`: two line-request masters,
round-robin, responses routed back in order. `npu_axi4_line_bridge`: one line
request -> one AXI4 burst of LINE_BYTES/8 beats (read or write, waits for B);
its `error` output pulses once per RRESP/BRESP error, and `npu_hx_regs` keeps
the sticky STATUS.bus_error.

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

## Subsystem integration (`npu_hexagon/npu_hx_subsystem.sv`)

### Ports

- `s_axi_*`: AXI4-Lite slave, 32-bit data, 16-bit address (64 KB window), same
  signal names and `X_INTERFACE` attributes style as `npu_accelerator`, clock
  `s_axi_aclk`, reset `s_axi_aresetn`.
- `m_axi_*`: AXI4 master, 32-bit address, 64-bit data, no ID, same names as
  `npu_accelerator`'s `m_axi_*`.
- `irq`: registered level of the sticky halt event, fault event and DMA-done
  bits gated by `IRQ_ENABLE`; CONTROL.CLEAR acknowledges it (a raw
  halted/fault level could only be dropped by restarting the core).

### Host register map (`npu_hx_regs`)

| Offset | Name | Access | Meaning |
| --- | --- | --- | --- |
| 0x000 | MAGIC | R | `0x314E5848` ("HXN1") |
| 0x004 | VERSION | R | `0x00010001` (minor 1 adds CONTROL.STOP and CAUSE[9:8]; the first build reads `0x00010000`) |
| 0x008 | CAPS | R | bit0 scalar, 1 vector, 2 L1I, 3 L1D, 4 L2, 5 TCM, 6 DMA; [15:8] VLEN/8; [31:16] TCM words/64 |
| 0x00C | CONTROL | W | bit0 START (scalar core at START_PC), bit1 FLUSH (L1D then L2 write-back + invalidate of all caches), bit2 DMA_START, bit3 CLEAR (sticky status bits), bit4 STOP (stop a running scalar core at the next instruction boundary, see below) |
| 0x010 | STATUS | R | bit0 running, 1 halted (ECALL, EBREAK or STOP), 2 fault, 3 vec_busy, 4 dma_busy, 5 flush_busy, 6 vec_illegal, 7 dma_error, 8 dma_done (sticky), 9 bus_error (sticky), 10 halt event (sticky), 11 fault event (sticky) |
| 0x014 | START_PC | RW | |
| 0x018 | CAUSE | R | [2:0] fault_cause, [9:8] halt_cause: 0 ECALL, 1 EBREAK, 2 STOP |
| 0x01C | FAULT_PC | R | PC of the ECALL / EBREAK / faulting instruction; after a STOP the first instruction not executed (the resume address) |
| 0x020 | CYCLES | R | scalar core cycles |
| 0x024 | INSTRET | R | |
| 0x028/0x02C | L1I_HITS / L1I_MISSES | R | |
| 0x030/0x034 | L1D_HITS / L1D_MISSES | R | |
| 0x038/0x03C | L2_HITS / L2_MISSES | R | |
| 0x040 | DMA_DDR_ADDR | RW | |
| 0x044 | DMA_TCM_ADDR | RW | word address |
| 0x048 | DMA_LEN | RW | [15:0] words, [31] dir (1 = TCM->DDR) |
| 0x04C | IRQ_ENABLE | RW | bit0 halt, bit1 fault, bit2 DMA done |
| 0x050-0x05C | MAILBOX0-3 | RW | arguments/results shared with scalar programs |
| 0x060 | VEC_CYCLES | R | cycles with vec_busy high, cleared by START |
| 0x064-0x7FFF | - | - | reads 0, writes ignored |
| 0x8000-0xFFFF | TCM | RW | 32-bit window onto TCM port B: word = offset[14:3], lane = offset[2]; past the TCM end reads 0 / ignores writes (OKAY) |

CAPS for the four PPA configurations: all units `0x0010087F`, no vector
`0x0010007D`, no L2 `0x0010086F`, no DMA `0x0010083F`. Cache counters are the
caches' own (cleared only by reset).

### Scalar data-port address map

| Range | Target |
| --- | --- |
| `0x0000_0000`-`0x3FFF_FFFF` | DDR through L1 D$ and L2 |
| `0x4000_0000`-`0x4000_FFFF` | TCM port B, uncached, 32-bit lane of the 64-bit word |
| `0x8000_0000`-`0x8000_0FFF` | the same `npu_hx_regs` registers as the host map (offset = addr[11:0]) |
| anything else, and the TCM window past its end | `d_resp_err`: the core faults with cause 6 |

TCM and MMIO accesses answer one cycle after the handshake (integration
review round 2; it was two), so they take as long as an L1 D$ hit (a load or
store is 5 core cycles). The TCM window request goes to port B in the
handshake cycle when the DMA is not using the port; an MMIO access executes
in the cycle after the handshake and answers in it. AXI errors on cached
traffic cannot reach the core (the caches have no error path): the bridge's
error pulse sets the sticky STATUS.bus_error instead.

CONTROL.STOP recovers a hung program without reloading the bitstream and
without changes to the scalar core: while a STOP is pending (armed only while
the core runs) the subsystem raises the core's `i_resp_err`, so the next
instruction-fetch response the core consumes is an instruction bus error. The
core stops at that instruction boundary (the previous instruction retired
with its memory access complete, the fetched one has no effect) with fault
cause 5, which nothing else raises in this subsystem; `npu_hx_regs` reports
it as a halt with halt cause 2 (halt event and IRQ like ECALL) and FAULT_PC
is the resume address: START at FAULT_PC continues the program (the register
file keeps its contents, CYCLES / INSTRET restart). A STOP while the core is
not running does nothing; START together with STOP on an idle core is a
START. It acts within about two instructions or when the current one ends: a
VFENCE waits for the vector unit and a cached access for its AXI traffic, so
an AXI slave that never answers is not recoverable this way. The vector unit
and a DMA in flight finish their work.

FLUSH flushes L1D and invalidates L1I together, then flushes L2 once L1D is
done. With `HX_ENABLE_L2 = 0` the line bridge feeds the L1 arbiter through a
64-to-32-bit beat serializer. A DMA_TCM_ADDR beyond the TCM makes the DMA
reject the descriptor (error + done, nothing moved). With
`HX_ENABLE_VECTOR = 0`, vector commands are consumed and set
STATUS.vec_illegal.

TCM port B is shared by the TCM DMA, the scalar core's TCM window and the host
TCM window, with fixed priority DMA > scalar > host (the DMA tolerates `gnt`
low). Port A belongs to the vector unit. Caches are not coherent with the host
or the DMA: software flushes (CONTROL.FLUSH) before handing DDR data to the
host and before starting a new program.
