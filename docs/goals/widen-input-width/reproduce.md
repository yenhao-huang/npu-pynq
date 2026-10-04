# Reproduce the packed-stream results

Every number in [Report.md](Report.md) comes from one of the commands below,
run on `dev` at `3845645`. The recorded run used Icarus Verilog 13.0,
Verilator 5.052 and Yosys 0.69+post on macOS.

## Lint and the testbenches

```bash
make -C src/test lint sim
```

Expect a clean Verilator report and nine `PASS tb_...` lines. Among them:

- `tb_npu_matrix_controller` covers the TKEEP and TLAST error paths and a
  K = 13 job whose frames end in partial beats.
- `tb_npu_matrix_controller_8x8` checks 34 shapes, each with its exact
  stall-free cycle count.
- `tb_npu_matrix_scaling` covers stalls, backpressure, and aborts during
  overlapped B loading.

## Cycle counts

```bash
python3 src/test/benchmark_scaling.py --output build/scaling --sizes 4 8 16
```

This prints one `METRIC` line per case and writes `build/scaling/metrics.json`.
The stall-free lines (`stalls=0`) are the Report's results table;
[metrics.json](metrics.json) is the recorded copy, with paths made
repository-relative.

Check any stall-free row against the formula
`M*ceil(K/8) + K*ceil(N/8) + M*N + M + N + 4`. For example, 8x8 with K = 256
gives `8*32 + 256*1 + 64 + 8 + 8 + 4 = 596`.

`--baseline` no longer works against a pre-#94 RTL root, because the
testbench now drives the 64-bit stream. The baseline and #60 columns in the
Report are taken from
[`docs/exp/2026-09-11-hardware-scaling.md`](../../exp/2026-09-11-hardware-scaling.md).

## Mutation checks

Each mutation below must make a testbench fail. Restore the file after each
one.

1. Disable the TKEEP check in `src/hw/rtl/npu_matrix/npu_matrix_controller.sv`.
   Change `(s_axis_tkeep == beat_expected_keep)` to `1'b1`. Then
   `make -C src/test build/npu_matrix_controller.ok` fails at
   `short final A TKEEP`.
2. Make the aligner ignore buffer room. Change
   `(count_after_emit <= COUNT_BITS'(IN_BYTES))` to `1'b1`. Then
   `make -C src/test build/npu_matrix_scaling.ok` fails with a result
   mismatch.

   `tb_npu_matrix_controller_8x8` has no watchdog and hangs under this
   mutation, so stop it by hand.

## Resource estimate

Yosys does not accept the loop variables inside `always_comb`, so copy the five
RTL files to a scratch directory and replace `always_comb` with `always @*`.
Then run, for both the old and the new copies:

```bash
yosys -q -p "read_verilog -sv npu_pe.sv npu_systolic_array.sv npu_axi_lite_regs.sv \
  npu_matrix_controller.sv npu_matrix_accelerator.sv; \
  chparam -set ROWS 8 -set COLUMNS 8 npu_matrix_accelerator; \
  synth_xilinx -family xc7 -top npu_matrix_accelerator -flatten; stat"
```

For the "before" column, take the RTL from `git show 1b2db0d:src/hw/rtl/...`,
which is `dev` after #60 and before #94.

## Vivado and board

Vivado runs only on the self-hosted runner, and the board runs only through
`cd.yml` on a `release/vX.Y.Z` branch. See the `release-npu-pynq` skill
under `.codex/skills/deploy/`.
