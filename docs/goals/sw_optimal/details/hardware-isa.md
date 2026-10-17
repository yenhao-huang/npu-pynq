# NPU ISA and the IF/ID front end (#120)

## Instruction set

One definition, `src/isa/isa.py`; the RTL decoder, the C++ encoder
(`src/compiler/mlir/lib/NpuEncoder.cpp`) and the C runtime's simulator are
checked against it (`src/test/tests/test_isa.py`, `test_compiler.py`).

| Opcode | Mnemonic | Operands | Effect |
| --- | --- | --- | --- |
| 0x00 | NOP | | |
| 0x01 | END | | wait for every job to be written back; DONE + IRQ |
| 0x02 | FENCE | | wait for every job to be written back |
| 0x10 | SHAPE | m, n, k | job shape (m, n <= 16, k <= 256) |
| 0x11-0x13 | ADDR_A/B/C | 32-bit offset from DATA_BASE | tile addresses |
| 0x14 | INCR | a, b, c (8-byte units) | post-increments |
| 0x15 | REPEAT | count | run the next word `count` times |
| 0x20 | GEMM | inc flags C/B/A | `C[m,n] = A[m,k] @ B[k,n]` (INT8 -> INT32) |

`REPEAT n` + a post-incrementing `GEMM` walks a row of `n` tiles in two words,
so one token through a 576 x 1536 layer (288 jobs) is a 22-word program.

## Hardware

```text
 AXI-Lite: PROG_ADDR, PROG_LEN, DATA_BASE, START             AXI4 master -> HP0 (64-bit)
                     |                                         ^      ^        |
   IF npu_isa_fetch  |  PC, 32-word queue, 16-beat bursts -----+      |        |
   ID npu_isa_decode |  shape/offset/increment state, REPEAT          |        |
   EX npu_isa_lsu    |  A,B tiles DDR -> s_axis ----------------------+        |
                     |  m_axis -> C tile in DDR <------------------------------+
                     v
     npu_matrix_controller (2-entry job queue) + 16 x 16 systolic array
```

- IF and the LSU share one AXI4 master; an in-order tag FIFO routes R beats.
  Up to 8 read and 8 write bursts are outstanding; bursts never cross a
  128-byte line.
- The ABI 1.0 register/DMA path is untouched: ABI 1.1 adds capability bit 6
  and registers 0x40-0x68 (`docs/npu-matrix-spec.md`).
- Files: `src/hw/rtl/npu_matrix/npu_accelerator/npu_isa_frontend/`,
  `npu_accelerator.sv` (mux), `npu_axi_lite_regs.sv`, Vivado
  `build_overlay.tcl` (`m_axi` -> `memory_ic/S02_AXI`).

## Verification

`tb_npu_isa_frontend` runs encoder-generated programs on the full 16 x 16
accelerator against a behavioural AXI memory with pseudo-random ready/valid
gaps, checks burst rules and WLAST, and compares *every* memory word with
the ISA simulator (results region poisoned beforehand):

| Program | Jobs | Cycles (random back-pressure) |
| --- | --- | --- |
| 16 x 16 x 256 | 1 | 1518 |
| 5 x 37 x 300 | 6 | 2455 |
| 1 x 49 x 40 | 4 | 399 |
| 33 x 20 x 320 (tk = 128) | 18 | 6617 |
| 3 x 3 x 7 | 1 | 46 |
| illegal opcode / bad shape | 1 / 0 | error 0x10 / 0x11 reported |

`make -C src/test lint sim`: Verilator lint clean, 14/14 testbenches pass.
