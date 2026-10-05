# Earlier four-experiment phase

**Reopened at the user's request: complete at least 21 experiments, irrespective
of cycles. Experiments 01-04 remain valid; the previous stopping condition and
completion audit below describe the earlier phase only.**

## Earlier phase (historical)

**Stopped after experiment 04: 1,314 cycles, 84.51% fewer than 8,482
(6.46x cycle speedup).** Functional RTL simulation meets the user's <=2,000
cycle stop condition. The selected hardware fits the FPGA, but **does not meet
100 MHz setup timing**. No fifth experiment was started.

Objective: reduce the full cold 16x16, K=256 transaction from 8,482 to <=2,000
busy cycles, or stop after more than 20 designed/executed experiments (21).
Both operands (8,192 valid bytes) and all 256 INT32 outputs must be transferred;
no preload, smaller K, omitted output, clock change, or changed arithmetic qualifies.
Stalls are tested separately and the stop metric uses continuous valid/ready.

Each experiment stores its frozen source, configuration, hypothesis, exact
commands, metrics, correctness evidence and temporary files in its own directory.
This campaign varies ingress byte lanes while preserving the previous
compute overlap and 32-bit egress. A wider experimental interface is explicitly
allowed by the new optimization objective; it is not silently deployed through
the unchanged legacy 8-bit accelerator or DMA interface.

Packed protocol: one beat carries up to LANES contiguous bytes within one A/B
row, low byte first. TKEEP is the low contiguous valid-byte mask. Row tails are
padded; a beat never crosses a row boundary. A and B retain separate TLAST-ended
frames. A is row-major M by K; B is row-major K by N. Valid data size is unchanged.
A banks are additionally striped by K modulo LANES to permit one simultaneous
write per lane without synthesizing a multiwrite RAM. B's column banks each
receive at most one byte per beat.

Exploration stops immediately upon a functionally correct <=2,000-cycle result;
then verify that candidate with resource/timing analysis and consolidate results.
The previous campaign is archived under `hardware/`.

## Experiments and bottleneck evidence

All rows use M=N=16, K=256, signed INT8 operands, INT32 results and 65,536 MACs.
Load cycles include both complete operand frames. Output cycles include all
256 output handshakes. Compute overlaps B loading; the tail column counts
the remaining compute and state-transition cycles, not total compute activity.

| Experiment | Input width | Load cycles | Tail cycles | Output cycles | Total cycles | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [01: control](exp01_ingress8/README.md) | 8 bits | 8,192 | 34 | 256 | 8,482 | Reproduces prior baseline |
| [02: two lanes](exp02_ingress16/README.md) | 16 bits | 4,096 | 34 | 256 | 4,386 | Loading still dominates |
| [03: four lanes](exp03_ingress32/README.md) | 32 bits | 2,048 | 34 | 256 | 2,338 | Input alone exceeds target |
| [04: eight lanes](exp04_ingress64/README.md) | 64 bits | 1,024 | 34 | 256 | **1,314** | Stop; validate selected candidate |

The identical work and constant 290-cycle tail/output overhead isolate the
input bandwidth bottleneck. In the control, loading consumes 96.58% of latency;
in experiment 04 it still consumes 77.93%. The measured no-stall equation is:

```text
cycles = M*ceil(K/LANES) + K*ceil(N/LANES) + M*N + M + N + 2
```

The winner requires eight valid input bytes per beat for this full tile.
At an assumed 100 MHz, that is 800 MB/s of input interface capacity and would
give 13.14 us and 4.988 GMAC/s for this transaction. These are conditional
conversions, **not achieved board throughput or timing-closed latency**.
The original 8-bit DMA path cannot deliver this cycle reduction unchanged.

## Physical validation of experiment 04

Vivado 2026.1, xc7z020clg400-1, standalone controller, MAX_K=256, LANES=8,
10 ns clock; synthesis, placement and routing completed successfully.

| Resource | Synthesis | Routed | Device capacity |
| --- | ---: | ---: | ---: |
| LUT | 29,854 | 29,781 | 53,200 |
| FF | 17,361 | 17,376 | 106,400 |
| BRAM tiles | 72 | 72 | 140 |
| DSP | 192 | 192 | 220 |

Routed setup WNS is **-0.337 ns**, hold slack **+0.136 ns**. The worst path
runs from A bank 6, lane 1 BRAM through the lane-selection MUX and DSP
multiplier to the first-column PE product register. Data-path delay is
10.285 ns (6.925 ns logic, 3.360 ns routing). Increasing input width removed
most load cycles but added A-bank selection to this timing-sensitive path.
This identifies a remaining hardware limitation; no further pipelining or
clock changes were attempted after the cycle stopping condition was reached.

This is OOC controller evidence. It excludes PS, DMA, AXI-Lite and interconnect;
external ports have no I/O-delay/placement constraints. There is no full-overlay
timing signoff or physical board benchmark. The experimental controller lives
under `design/`; production RTL, wrapper, DMA configuration and runtime retain
their previous interface. Deployment would require packed input/TKEEP support,
row-tail packing in software, integration and timing closure.

## Correctness and evidence

Each of the four widths passed Verilator lint and nine independently checked
matrix transactions, including K=1/32/256, partial tiles, signed data, input
gaps, output backpressure and poisoned invalid tail lanes. Three additional
abort/recovery scenarios exercise reset during B loading, early B TLAST and
malformed B TKEEP. Every valid result element and output TLAST is checked;
the target also verifies exactly 8,192 valid input bytes and all 256 outputs.
The PE arithmetic is unchanged. K<=256 INT8 products cannot overflow INT32,
so the matrix reference sum is exact for this campaign's supported range.

Local raw evidence for each experiment:

- `config.json`: hypothesis, parameters, source SHA-256, exact commands,
  working directory, temporary-directory settings and outcome.
- `sources/`: frozen RTL and testbench used for that run.
- `results.json` and `artifacts/simulation.log`: all nine measurements and PASS.
- `artifacts/lint.log`, `artifacts/compile.log`, `scratch/`: tool outputs.
- Experiment 04 `artifacts/vivado/`: synthesis/routed utilization, timing,
  critical paths, checkpoints and `status.txt` recording the timing failure.
- `campaign.json`: four experiments and the explicit stop reason.

Experiment 01 had one failed tool-setup attempt (Verilator could not locate
its installation library); `attempts/attempt01/` preserves that evidence.
Fixing VERILATOR_ROOT and rerunning the same design is a retry, not another
hardware experiment. Replays verify frozen runs without changing the ledger.
Generated files remain local and ignored by Git; this report, per-experiment
records, canonical `design/`, driver and Vivado Tcl are reproducible sources.

Final verification replayed all four frozen configurations with unchanged
cycles and no additional experiments. The legacy RTL lint and all nine
testbenches also passed with generated output redirected into experiment 04:

```sh
make -B -C src/test lint sim BUILD=../../exp/0911_performance_optimization/exp04_ingress64/artifacts/regression
```

The combined log is `exp04_ingress64/artifacts/regression.log`. This legacy
suite covers the production RTL; the separate four packed-controller replays
cover the experimental RTL. A final local evidence audit is recorded in
`exp04_ingress64/artifacts/completion-audit.json`.

## Reproduction

From the repository root in a fresh checkout with Python 3.12, Verilator and
Icarus Verilog on PATH, run these four commands sequentially. Each command
creates all generated files inside the named experiment directory.

```sh
python exp/0911_performance_optimization/run_experiment.py exp01_ingress8 --lanes 1 --hypothesis "Control: reproduce 8482 cycles with all input and output data"
python exp/0911_performance_optimization/run_experiment.py exp02_ingress16 --lanes 2 --hypothesis "Halve loading to 4096 cycles; predict 4386 total"
python exp/0911_performance_optimization/run_experiment.py exp03_ingress32 --lanes 4 --hypothesis "Input alone needs 2048 cycles; predict 2338 total"
python exp/0911_performance_optimization/run_experiment.py exp04_ingress64 --lanes 8 --hypothesis "Load in 1024 cycles with striped A banks; predict 1314 total"
```

On the completed local campaign, use the same command with `--replay` to
verify an existing run. New exploration is rejected once the ledger records
the stopping condition. Replay writes a separate timestamped artifact folder.

For physical validation, enter `exp04_ingress64/`, set TEMP, TMP and TMPDIR
to its absolute `scratch/` path, then run:

```sh
vivado -mode batch -source profile.tcl -log artifacts/vivado.log -journal artifacts/vivado.jou
```

The Tcl script writes generated Vivado files under `artifacts/vivado/` and
records resource overflow or final setup/hold status explicitly. A successful
Vivado process exit does not imply timing closure: inspect `status.txt`.
