# NPU Matrix Overlay

The PYNQ-Z1 matrix overlay supports source-controlled 2x2 and 8x8 systolic
array configurations. The 8x8 target is the default. Explicit 2x2 builds remain supported.

## RTL boundaries

The RTL directory follows the instance hierarchy. Under the design directory
`src/hw/rtl/npu_matrix/`, `npu_accelerator/` contains its top-level
module, `dma_axi/`, and `npu_matrix_core/`. The core directory contains its
module, `controller/`, and `npu_matrix_datapath/`; the datapath directory
contains its module, `memory/`, and `systolic_array/`. The DMA engine remains
a Xilinx IP in the Vivado design.

- **DMA and AXI:** Vivado instantiates the Xilinx AXI DMA IP and connects its
  AXI-Stream ports to `npu_accelerator`. The separate
  `npu_axi_lite_regs` RTL module owns the accelerator's control/status register
  interface. The refactor does not replace the DMA IP or change the register map.
- **Controller:** `npu_matrix_controller` owns job validation, load/compute/output
  sequencing, stream handshakes, buffer addresses and enables, timeout/error
  handling, and the output selection. It does not store operands or instantiate
  processing elements.
- **Memory:** `npu_matrix_datapath` instantiates one `npu_operand_buffer` bank
  per array row for A and per column for B. A banks accept row-aligned words;
  B banks accept one byte per emitted row. Each bank has a synchronous read
  port, and disabled reads hold their prior output. This preserves the
  block-RAM-friendly schedule used by the 64-bit input stream. Each bank is
  two halves deep. The controller's `load_half` and `exec_half` select the
  half written by the next job and the half read by the current job (A/B
  ping-pong buffer, issue #64).
- **Systolic array:** `npu_systolic_array` owns the processing elements and
  accumulators. `npu_matrix_core` wires it and the operand memories to the
  controller; `npu_accelerator` wires that core to AXI-Lite and streams.

The external AXI-Stream interfaces and matrix results are unchanged by the
module split. Issue #64 adds a two-entry job queue: the next job's operands
load into one half of every bank while the current job computes and drains
from the other. STATUS bit 3 (ACCEPT) reports a free queue entry and gates the
configuration registers, and CAPABILITIES bit 5 (PIPELINED_JOBS) advertises
this. A driver that waits for `!BUSY` still runs one job at a time.

## Select a configuration

Run Vivado from the repository root. Select the 8x8 target with:

```text
vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl \
  -tclargs --array-size 8
```

Omit the option for the default 8x8 target; use `--array-size 2` for 2x2. Only `2` and
`8` are accepted. The explicit 2x2 build is written below
`build/vivado/npu_matrix/`; the 8x8 build uses
`build/vivado/npu_matrix_8x8/` so the two configurations cannot overwrite one
another.

Add `--elaborate-only` for block-design validation. Add `--allow-dirty` only
for an exploratory local implementation whose artifacts will not be
published.

## Verify the target

The repository simulation suite includes full 8x8 and masked 7x5 jobs with
deterministic pseudo-random signed INT8 operands. Run all open-source gates
with:

```text
make -C src/test lint sim
```

For a release candidate, run the 8x8 Vivado command without
`--elaborate-only`. The build must finish synthesis, implementation, routing,
DRC, and setup timing with nonnegative WNS before BIT/HWH artifacts are
published. The build evidence records the selected row and column counts.

Finally, deploy the matching BIT/HWH pair to a PYNQ-Z1 and run the matrix
example's board smoke test. Open-source simulation does not substitute for
Vivado timing/resource evidence or physical-board validation.

## Operand input stream

A and B reach the accelerator as two AXI4-Stream frames from AXI DMA MM2S,
eight INT8 operands per 64-bit beat in row-major byte order. A frame whose
length is not a multiple of eight ends with one partial beat marked by TKEEP.
Software still sends `M*K` and `K*N` byte transfers. A row aligner writes one
row-aligned word per cycle into the operand banks, so a stall-free physical job
takes `M*ceil(K/8) + K*ceil(N/8) + M*N + M + N + 3` cycles when it runs alone.
Back-to-back jobs issued on ACCEPT settle at
`max(M*ceil(K/8) + K*ceil(N/8) + 2, K + M + N + M*N + 1)` cycles each: the
operand load or the array's compute and output, whichever is longer.

## 8x8 implementation evidence

Vivado 2026.1 implemented and routed the 8x8 target for
`xc7z020clg400-1` at 100 MHz on 2026-09-05. The routed timing report met all
specified constraints with setup WNS 0.079 ns and hold WHS 0.017 ns. The
implementation used 17,302 slice LUTs (32.52%), 8,528 slice registers (8.02%),
2 block RAM tiles (1.43%), and 64 DSPs (29.09%). There were no unrouted nets,
routing errors, setup failures, hold failures, or DRC errors.

These figures predate the 64-bit packed operand stream (#93), which changes
the input buffers and the DMA MM2S width; they must be re-measured from the
next Vivado build.

These figures establish build feasibility for the PYNQ-Z1 target. A physical
matrix smoke test is still required for each release artifact because the
repository does not treat implementation evidence as board evidence.

## Instruction-stream front end (ABI 1.1)

ABI 1.1 adds capability bit 6, `ISA_FRONTEND`: the NPU fetches and executes a
program from DDR instead of having the ARM sequence every tile. The register
and DMA path of ABI 1.0 is unchanged and still works; while a program is in
flight (`ISA_STATUS.BUSY`), the front end owns the matrix controller and the DMA
streams are held off.

```text
            AXI-Lite (GP0)                         AXI4 master (HP0, 64-bit)
                 |                                        ^   ^   |
   PROG_ADDR/LEN, DATA_BASE, START                        |   |   |
                 v                                        |   |   v
   +-------------------+   +------------------+   +-------+---+-------+
   | IF  npu_isa_fetch | ->| ID npu_isa_decode| ->| EX  npu_isa_lsu   |
   | PC, 32-word queue |   | shape, A/B/C     |   | A,B tiles -> s_axis
   | 16-beat bursts    |   | offsets, REPEAT  |   | m_axis -> C tile   |
   +-------------------+   +------------------+   +---------+---------+
                                    | START, M/N/K          |
                                    v                       v
                           npu_matrix_controller + 16 x 16 systolic array
```

The instruction set is defined once, in `src/isa/isa.py`; `src/isa/sim.py` is
its bit-accurate simulator and `src/isa/layout.py` the tiled-GEMM layout that
the compiler, runtimes and testbench share. Each instruction is one 64-bit
word with the opcode in bits [63:56]:

| Opcode | Mnemonic | Operands | Effect |
| --- | --- | --- | --- |
| 0x00 | NOP | | |
| 0x01 | END | | wait until every job is written back, then DONE and IRQ |
| 0x02 | FENCE | | wait until every job is written back |
| 0x10 | SHAPE | m [31:24], n [23:16], k [15:0] | job shape |
| 0x11-0x13 | ADDR_A/B/C | offset [31:0], 8-byte aligned | tile offsets from DATA_BASE |
| 0x14 | INCR | a [15:0], b [31:16], c [47:32], in 8-byte units | post-increments |
| 0x15 | REPEAT | count [31:0] | execute the next word count times |
| 0x20 | GEMM | flags [2:0] = inc C, B, A | `C[m,n] = A[m,k] @ B[k,n]`, then advance the flagged offsets |

A GEMM reads the dense row-major INT8 tiles at `DATA_BASE + a` and
`DATA_BASE + b` and writes the dense INT32 result at `DATA_BASE + c`. `REPEAT n`
followed by a post-incrementing GEMM walks a row of `n` tiles in two words.
Jobs pipeline through the controller's two-entry queue; the load/store unit
keeps up to eight read and eight write bursts outstanding, each at most 16
beats and never crossing a 128-byte line.

| Offset | Register | Access | Meaning |
| --- | --- | --- | --- |
| 0x40 | ISA_CONTROL | W | bit 0 START (ignored while BUSY) |
| 0x44 | ISA_STATUS | R | bit 0 RUNNING, 1 DONE, 2 ERROR, 3 BUSY |
| 0x48 | PROG_ADDR | RW | physical address of word 0, 8-byte aligned |
| 0x4C | PROG_LEN | RW | program length in words; fetch never reads past it |
| 0x50 | DATA_BASE | RW | physical address that tile offsets are relative to |
| 0x54 | ISA_ERROR | R | 0x10 illegal opcode, 0x11 bad shape, 0x12 misaligned, 0x13 AXI read, 0x14 AXI write, 0x15 fetch; 1-6 forward the controller's code |
| 0x58 | ISA_PC | R | words consumed by decode |
| 0x5C | ISA_JOBS | R | jobs written back by the last program |
| 0x60 | ISA_CYCLES | R | cycles the last program ran |
| 0x64 | ISA_INSTRUCTIONS | R | words decoded by the last program |
| 0x68 | ISA_VERSION | R | 1 |

`tb_npu_isa_frontend` runs encoder-generated programs on the full 16 x 16
accelerator against an AXI memory with random back-pressure, and compares all
of memory with the simulator's result. An AXI error response or a controller
error stops decode; the jobs already issued drain before BUSY falls. Recovery
from an AXI error is a bitstream reload.
