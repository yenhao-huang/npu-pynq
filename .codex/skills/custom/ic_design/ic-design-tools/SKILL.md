---
name: ic-design-tools
description: Lint, simulate, debug waveforms, view and synthesize the RTL in this repository. Use when changing anything under src/hw/, when a testbench fails, when you need to find the cycle where a signal diverges from its reference, or when you need LUT/FF/timing numbers. Provides lint, sim, signals, first_mismatch, value_at, value_range, show_wave and synth.
---

# IC design tools

Second-scale feedback on RTL changes, without loading waveforms or logs into
context. Large artifacts stay on disk; you get handles.

## Setup

Check the toolchain once per session:

```bash
ic doctor
```

`ok: true` means every default backend is installed. If `ic` is not found:

```bash
pip install -e tools/ic
```

## The loop

**1. After every RTL edit, lint.** It costs a second and returns every
diagnostic in full.

```bash
ic lint --files src/hw/rtl/systolic_array/npu_pe.sv --top npu_pe
```

**2. Simulate to get a verdict and a waveform handle.**

```bash
ic sim --files src/hw/rtl/systolic_array/npu_pe.sv src/hw/tb/systolic_array/tb_npu_pe.sv --tb tb_npu_pe --top npu_pe
```

Returns `ok`, a few log lines explaining the verdict, and
`wave: "<run_id>/wave.fst"`. **Do not try to read the waveform file.** It is
hundreds of megabytes. Use the handle with the debug tools below.

**3. When a test fails, find the divergence.** This is the highest-value tool
here: it replaces scrolling a waveform.

```bash
ic first_mismatch --wave 166d08/wave.fst --ref expected_result --dut accumulator
```

Returns the first cycle where the two disagree, both values in decimal, hex
and binary, and a few cycles of context.

**4. Inspect state around that cycle.**

```bash
ic value_at --wave 166d08/wave.fst --signals a_in b_in enable --cycle 42
ic value_range --wave 166d08/wave.fst --signal accumulator --from-cycle 38 --to-cycle 46
```

**5. If you do not know the signal names**, list them first. Signal arguments
accept a unique suffix, so `accumulator` works for
`TOP.tb_npu_pe.dut.accumulator`.

```bash
ic signals --wave 166d08/wave.fst --pattern 'acc*'
```

**6. Area and timing.** `estimate` (Yosys) takes seconds and is safe in a
loop. `full` (Vivado) takes minutes and is the only trustworthy source of
timing for the Zynq-7020.

```bash
ic synth --files src/hw/rtl/systolic_array/npu_pe.sv --top npu_pe --mode estimate
```

**7. To hand a problem to a person**, open the waveform for them with the
right signals preselected and the view centred on the bad cycle:

```bash
ic show_wave --wave 166d08/wave.fst --signals accumulator a_in b_in --center-cycle 42
```

You cannot read anything back from this; it is for the human, not for you.

## Rules

- Never `cat` a `.fst`, `.vcd` or `sim.log`. Use the handle tools, or
  `ic artifact <run_id>/sim.log --tail 40` to peek at a bounded slice.
- Pass `--compact` when you only need the JSON, not a readable shape.
- Lint warnings do not fail a lint; check `error_count`, not just `ok`.
- A Yosys estimate has no timing at all. Never report it as timing.
- Verilator is two-state and will not reproduce an X-propagation bug. For
  reset and initialisation problems use `--backend icarus` on both `sim` and
  `lint`.
- Long runs may return `{"job_id": ...}` instead of a result. Wait with
  `ic job <job_id> --poll-s 240`.

## Reference

- `ic tools` prints every tool with its full JSON schema.
- `ic runs --limit 10` lists recent runs; `ic run <run_id>` shows one in full.
- Full documentation: `docs/ic-design-tools/README.md`.
