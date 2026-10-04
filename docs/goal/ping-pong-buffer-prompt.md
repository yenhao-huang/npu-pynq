# Goal: A/B ping-pong buffer (issue #64)

## Prompt

實做 ping-pong buffer

實驗: 跑 16 個矩陣乘法 job 的 testbench 看 cycles 數是不是有下降
應該不會下降很多，因為目前 load data 的時間很重，但先實作好

驗收條件
得到 16 個矩陣乘法 job 的實驗數據

## Implementation

Branch `npu/issue64-b`, cut from `origin/dev` (667e072). Commit bfa3637
replays the issue #64 change prepared in `worktrees/npu-issue64-a/issue64.patch`:
`npu_matrix_controller` is split into a load engine and an exec engine that
share a two-entry job queue, one operand bank (A + B) per entry. While the exec
engine computes and drains job *n* from bank `exec_ptr`, the load engine fills
bank `load_ptr` with job *n+1*. STATUS bit 3 (ACCEPT) tells software a bank is
free; CAPABILITIES bit 5 advertises PIPELINED_JOBS. A driver that waits for
`!BUSY` sees the old behaviour cycle for cycle.

## Experiment: 16 matrix-multiply jobs

Testbench: `src/hw/tb/npu_matrix/tb_npu_matrix_controller_16jobs.sv`.
It runs 16 stall-free jobs (random INT8 operands, every output checked against
an exact INT32 golden model) twice:

- **serial**: each START waits for `!BUSY`, which is the pre-ping-pong behaviour.
- **pipelined**: each START waits only for `ACCEPT`, which keeps both banks busy.

The input stream is one byte per cycle and the output is one INT32 per cycle
with no backpressure. `span` is the wall-clock length of all 16 jobs.
`steady` is the mean controller-reported `CYCLES` of jobs 1..15.

The **baseline** column runs the same serialized pass against the controller
from `origin/dev`, before this change. It matches the serial pass cycle for
cycle in every shape, so the serial pass is a valid stand-in for the old
hardware.

| shape (M x N x K) | baseline span | serial span | pipelined span | saved | span reduction | steady/job serial -> pipelined | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4x4x8   |  1616 |  1616 |  1076 |  540 | 33.4 % |   99 -> 65   | 1.50x |
| 4x4x64  |  9680 |  9680 |  8300 | 1380 | 14.3 % |  603 -> 513  | 1.17x |
| 4x4x256 | 37328 | 37328 | 33068 | 4260 | 11.4 % | 2331 -> 2049 | 1.13x |
| 8x8x8   |  3536 |  3536 |  2156 | 1380 | 39.0 % |  219 -> 129  | 1.64x |
| **8x8x64** (default, #63 shape) | **18768** | **18768** | **16548** | **2220** | **11.8 %** | **1171 -> 1025** | **1.13x** |
| 8x8x256 | 70992 | 70992 | 65892 | 5100 |  7.2 % | 4435 -> 4097 | 1.08x |

In the pipelined run, the first job still costs the full serial latency
(1171 cycles at 8x8x64), because nothing is in flight yet to overlap with it.

### Reading the numbers

As expected, the drop is modest for realistic K. Each pipelined job settles
at exactly `ROWS*K + K*COLUMNS + 1` cycles (8x8x64: 512 + 512 + 1 = 1025). That
is the operand load time over the 8-bit `s_axis`. The ping-pong buffer fully
hides the compute and output phases (about `K + ROWS + COLUMNS` plus
`ROWS*COLUMNS` cycles) behind the next load, but it cannot hide the load
itself. So the saving is roughly `(compute + output) / (load + compute + output)`:

- It is large when K is small and the 64-beat output drain is a big share
  (8x8x8: 39 %).
- It is about 12 % at the #63 shape 8x8x64.
- It is about 7 % at K = 256, where load dominates.

Further speedup needs a wider or faster load path (for example a 32/64-bit
`s_axis`, or reusing B across row tiles). Banking alone cannot provide it.

### Reproduce

```bash
make -C src/test build/npu_matrix_controller_16jobs.ok   # default 8x8x64
# other shapes
iverilog -g2012 -s tb_npu_matrix_controller_16jobs \
  -Ptb_npu_matrix_controller_16jobs.ROWS=4 -Ptb_npu_matrix_controller_16jobs.COLUMNS=4 \
  -Ptb_npu_matrix_controller_16jobs.JOB_K=256 -o /tmp/16jobs.vvp \
  src/hw/rtl/npu_matrix/npu_matrix_controller.sv src/hw/rtl/systolic_array/*.sv \
  src/hw/tb/npu_matrix/tb_npu_matrix_controller_16jobs.sv && vvp /tmp/16jobs.vvp
# baseline: same command with -DNPU_SERIAL_BASELINE and the three RTL files
# taken from `git show origin/dev:<path>`
```

### Validation and gaps

- `make -C src/test lint sim` passes on this branch: all 11 testbenches, including
  the new 16-job bench and the unchanged 34-case 8x8 controller regression.
- Not yet run: Vivado synthesis, timing, and BRAM/LUT usage. Board validation
  is also outstanding.
- `src/runtime/` still issues one job at a time. Software sees this speedup only
  after the runtime stages the next job on `ACCEPT`.
