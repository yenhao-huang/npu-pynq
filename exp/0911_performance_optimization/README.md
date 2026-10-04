# 0911 hardware performance optimization

The Git snapshot keeps the compact campaign records: this overview,
[`RESULTS.md`](RESULTS.md), [`PHASE1.md`](PHASE1.md),
[`matrix_results.csv`](matrix_results.csv), the experiment plan, and the
[three-design board results](board/BOARD_RESULTS.md). The local experiment
workspace also contains frozen RTL, execution drivers, Vivado projects and raw
board evidence; those larger generated/archive files are not part of this
snapshot. The reproduction commands below require that complete local workspace.

The revised objective is **more than 20 experiments (at least 21), regardless
of cycles**. All 21 distinct hardware configurations have been executed and
passed functional checks. The lowest full cold 16x16, K=256 transaction is
**579 cycles** (experiment 18), versus 8,482 in the control: 93.17% fewer cycles,
14.65x cycle speedup. That candidate fails 100 MHz setup (WNS -0.147 ns).
**Experiment 09 achieves 1,315 cycles and passes 100 MHz internal timing**
(WNS +0.386 ns). Experiment 19 preserves 1,123 cycles while cutting BRAM from
72 to 8 tiles, with timing not yet checked. These 21 standalone-controller
results do not claim board timing; the later three-design board comparison is
reported separately in [`board/BOARD_RESULTS.md`](board/BOARD_RESULTS.md).

The campaign is complete at 21 experiments. All 189 matrix cases and the final
provenance/storage/stop-condition audit pass. Selected physical validation is
complete: two new candidates routed, four additional configurations synthesized.

[All 21 results and interpretations](RESULTS.md) | [Initial phase](PHASE1.md) |
[Previous hardware archive](hardware/README.md)

## Workload and methodology

Every target run multiplies 16x256 by 256x16: 65,536 signed INT8 MACs, 8,192
valid input bytes and all 256 INT32 result words. Both operands are loaded in
the measured transaction. The no-stall metric counts the complete busy interval,
including output handshakes. No preload, smaller K or changed arithmetic is used.
Input gaps and output backpressure are tested in separate correctness cases.

Experiments 01-05 vary ingress width. Experiments 06-08 disable compute/load
overlap. Experiments 09-11 add a matched operand/valid pipeline stage after
BRAM selection. Experiments 12-18 vary output bandwidth and combine it with
faster ingress. Experiments 19-21 isolate A memory mapping and DSP allocation
while keeping the same transport and cycle count as experiment 13.

Each experiment has a hypothesis, frozen source hashes, parameters, exact
commands, measured phases and functional checks. Retries preserve their previous
attempts and do not add experiment numbers. The count-only stop guard prevents
starting a 22nd configuration. A cycle count below 2,000 never stops this phase.

## Experimental interface

One input beat carries up to LANES consecutive INT8 bytes within an A or B row,
low byte first. Row-tail TKEEP is a contiguous low-byte mask; invalid lanes are
padding. A (M by K) and B (K by N) use row-major layout and separate TLAST-ended
frames. A banks are striped by K modulo LANES; B retains column banks.

The v2 output packs OUT_LANES adjacent INT32 words per beat within a result row,
low word first. Four TKEEP bits are set for each valid word. Invalid tail words
have zero keep. TLAST marks the final beat of the complete matrix. Logical
strides remain K/N/4N bytes; row-tail padding belongs to the transport protocol.

Production RTL, wrapper, DMA configuration and runtime have not been switched
to this experimental interface. Three experimental overlays were subsequently
integrated and tested on the board; see the separate board result record.
OOC resource/timing reports exclude PS, DMA, AXI-Lite and interconnect and have
no external I/O-delay constraints. RTL cycles alone do not establish a usable
clock frequency or achieved board latency.

## Local archive and reproduction

The paths below describe the complete local experiment workspace. This Git
snapshot versions the compact summary records and plan, not the frozen RTL,
drivers or generated evidence required to rerun the commands.

- `design/`: preserved RTL/testbench for experiments 01-04.
- `design_v2/`: parameterized RTL and independent packed-output checker for 05-21.
- `experiments-v2.json`: 17 follow-on hypotheses and distinct parameter sets.
- `run_experiment.py`: freeze and run a single experiment; enforce count-only stop.
- `run_planned.py`: execute missing declared 05-21 runs sequentially; retain completed evidence.
- `profile_candidate.py` / `.tcl`: local Vivado validation of a passing frozen run.
- `<exp_name>/README.md`: concise experiment record.
- `<exp_name>/sources/`, `config.json`, `results.json`, `artifacts/`, `scratch/`:
  generated local evidence, ignored by Git. All per-experiment temporary files
  remain under that experiment, including compiler/simulator/Vivado scratch.
- `hardware/`: complete earlier hardware archive, including original logs and checkpoints.

In the complete local experiment workspace with Python 3.12, Verilator and
Icarus Verilog on PATH:

```sh
python exp/0911_performance_optimization/run_experiment.py exp01_ingress8 --lanes 1 --hypothesis "Control: reproduce 8482 cycles with complete input and output"
python exp/0911_performance_optimization/run_experiment.py exp02_ingress16 --lanes 2 --hypothesis "Halve loading; predict 4386 cycles"
python exp/0911_performance_optimization/run_experiment.py exp03_ingress32 --lanes 4 --hypothesis "Input alone requires 2048 cycles; predict 2338 total"
python exp/0911_performance_optimization/run_experiment.py exp04_ingress64 --lanes 8 --hypothesis "Eight input bytes per beat; predict 1314 cycles"
python exp/0911_performance_optimization/run_planned.py
```

To recheck an existing frozen experiment, use `run_experiment.py <name>
--lanes <original-lanes> --hypothesis "Verify frozen run" --replay`. Replays use
saved commands and frozen sources, store separate timestamped artifacts and do
not extend the experiment count. Failed simulation attempts require explicit
`--retry` with unchanged design parameters; they preserve prior files.

To reproduce selected local physical checks after their functional runs:

```sh
python exp/0911_performance_optimization/profile_candidate.py exp09_boundary64 exp18_in128_out256
python exp/0911_performance_optimization/profile_candidate.py --synth-only exp13_output128 exp19_distributed_a exp20_dsp128 exp21_dsp220
```

The physical runner refuses to overwrite existing reports. Inspect terminal
status and process exit together; a successful tool exit is not timing closure.
The initial experiment-09 implementation failure is preserved; its checkpoint
retry uses `exp09_boundary64/resume_route.tcl` and separate `artifacts/route_retry/`.

The original production lint and nine testbenches passed during the first
phase. The extended campaign's testbench covers all nine matrix cases per
configuration, including K=1/32/256, partial tiles and signed extremes, plus
reset/malformed B recovery. Every result word, input byte count, output keep,
TLAST and backpressure stability is independently checked. With K<=256, INT8
products cannot overflow the unchanged INT32 accumulator contract.
