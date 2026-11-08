# Goal: A/B ping-pong buffer (issue #64)

## Prompt

實做 ping-pong buffer

實驗: 跑 16 個矩陣乘法 job 的 testbench 看 cycles 數是不是有下降
應該不會下降很多，因為目前 load data 的時間很重，但先實作好

驗收條件
得到 16 個矩陣乘法 job 的實驗數據

## Implementation

Branch `npu/issue64-b`, merged with `dev` at bd63070. That `dev` includes:
- #59: per-edge operand banks and in-job B-row overlap;
- #94: eight INT8 operands per 64-bit input beat;
- #62: the module split into controller, datapath and `npu_operand_buffer`.

The ping-pong buffer is built on top of all three:

- **Datapath:** every `npu_operand_buffer` is two halves deep. The controller's
  `load_half` / `exec_half` outputs drive the address MSB of writes and reads.
  The number of buffers is unchanged.
- **Controller:** split into a load engine (row aligner, frame checks, writes
  the half at `load_ptr`) and an exec engine (clear, wavefront, output, reads
  the half at `exec_ptr`). The engines share a two-entry job queue.
- **Exec start:** the exec engine starts once a job's A frame lands, so the
  wavefront still advances as that job's B rows arrive (#59).
- **Software interface:** STATUS bit 3 (ACCEPT) means a queue entry is free.
  CAPABILITIES bit 5 advertises PIPELINED_JOBS.

A driver that waits for `!BUSY` runs one job at a time, one cycle faster than
before: the idle `STATE_CLEAR` cycle between `LOAD_B` and `COMPUTE` is gone.

## Experiment: 16 matrix-multiply jobs

Testbench: `src/hw/tb/npu_matrix/tb_npu_matrix_controller_16jobs.sv`.
It drives `npu_matrix_core` with 16 stall-free jobs: random INT8 operands, 64-bit
packed beats with TKEEP, and every output checked against an exact INT32 model.
The output sink takes one INT32 per cycle with no backpressure. `span` is the
wall-clock length of all 16 jobs. `steady` is the mean `CYCLES` of jobs 1..15.

- **dev**: `dev` at bd63070 (packed stream, single buffer), serialized
  (`-DNPU_SERIAL_BASELINE`). This is the hardware this change replaces.
- **ping-pong serial**: this branch, START on `!BUSY`.
- **ping-pong pipelined**: this branch, START on `ACCEPT`.

| shape (M x N x K) | dev span | ping-pong serial | **ping-pong pipelined** | saved vs dev | steady/job dev -> pipelined | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4x4x8     |  704  |  688  |   **538** | 23.6 % |   42 -> 33   | 1.31x |
| 4x4x64    |  2048 |  2032 |  **1597** | 22.0 % |  126 -> 98   | 1.28x |
| 4x4x256   |  6656 |  6640 |  **6205** |  6.8 % |  414 -> 386  | 1.07x |
| 8x8x8     |  1664 |  1648 |  **1438** | 13.6 % |  102 -> 89   | 1.16x |
| **8x8x64** (default) | **3456** | **3440** | **2390** | **30.8 %** | **214 -> 145** | **1.45x** |
| 8x8x256   |  9600 |  9584 |  **8309** | 13.4 % |  598 -> 514  | 1.16x |
| 16x16x64  |  8832 |  8816 |  **5846** | 33.8 % |  550 -> 353  | 1.51x |
| 16x16x256 | 21120 | 21104 | **16709** | 20.9 % | 1318 -> 1026 | 1.26x |

### Reading the numbers

The first experiment (byte-wide stream, before #94) saved only 7 % at 8x8x64,
because the 1024-cycle load dwarfed everything else. With 8 operands per beat
the load at 8x8x64 is 130 cycles, so compute and output now matter. Ping-pong
saves 30.8 % there and 33.8 % at 16x16x64.

The steady-state cost per job is whichever engine is slower. It matches every
row above:

```
load = M*ceil(K/8) + K*ceil(N/8) + 2          (aligned words + 1 fill per frame)
exec = 1 + (K + M + N - 1) + M*N              (clear, wavefront, output)
cost = load if load >= exec, else exec + 1
```

- **Exec-bound** (8x8x64: exec 144 vs load 130 -> 145; 16x16x64: 352 vs 258 ->
  353): the array is now the bottleneck. The next job cannot start its
  wavefront until the `M*N` results have drained from the accumulators.
- **Load-bound** (8x8x256: load 514; 16x16x256: load 1026): ping-pong hides
  all of compute and output, and the 64-bit stream is the bound.

The next steps differ by regime:
- exec-bound shapes need output that does not stall the array (a result buffer
  or double accumulators);
- load-bound shapes need operand reuse across tiles.

### Reproduce

```bash
make -C src/test build/npu_matrix_controller_16jobs.ok   # default 8x8x64
# other shapes
iverilog -g2012 -s tb_npu_matrix_controller_16jobs \
  -Ptb_npu_matrix_controller_16jobs.ROWS=16 -Ptb_npu_matrix_controller_16jobs.COLUMNS=16 \
  -Ptb_npu_matrix_controller_16jobs.JOB_K=64 -o /tmp/16jobs.vvp \
  $(find src/hw/rtl -name '*.sv') \
  src/hw/tb/npu_matrix/tb_npu_matrix_controller_16jobs.sv && vvp /tmp/16jobs.vvp
# dev baseline: same command with -DNPU_SERIAL_BASELINE and the RTL files taken
# from `git show bd63070:<path>`
```

### History

The first run of this experiment used the byte-wide stream. At 8x8x64, 16 jobs
took:

| version | cycles | saving |
| --- | ---: | --- |
| pre-#59, serial | 18768 | baseline |
| ping-pong only | 16548 | -11.8 % |
| #59 only | 17760 | |
| #59 + ping-pong | 16484 | -7.2 % vs #59 |

### Validation and gaps

- `make -C src/test lint sim` passes, with all 13 testbenches green.
- `tb_npu_matrix_scaling` also passes at SIZE = 4 and 16. Its exact-latency
  constant moved from +4 to +3.
- `python -m unittest discover -s src/test/tests` passes (148 tests).
- Not yet run: Vivado synthesis, timing and BRAM/LUT usage. Board validation is
  also outstanding. The doubled-depth buffers (512 B at MAX_K = 256) should
  still map to one BRAM18 each, but this is unverified.
- `src/runtime/` still issues one job at a time. Software sees the speedup only
  after the runtime stages the next job on `ACCEPT`.
