# FP16 Adder Evaluation Goal

Build the `run_eval` tool and design, verify, and optimize an FP16 floating-point
**adder**, matching the `fpadd_f16` benchmark in Section 7.1 of *RTLScout: Joint
Agentic Code and Synthesis Optimization for Efficient Digital Circuits*.
The local design name is `fpadd_fp16`.

## Design contract

- Operation: addition (`a + b`), not multiplication.
- Format: IEEE-754 binary16, 16 bits: 1 sign bit, 5 exponent bits, and 10 fraction
  bits; exponent bias 15. Both operands and the result are 16 bits.
- Support normal and subnormal values, round-to-nearest ties-to-even, signed
  zeros, infinities, and NaNs. Document the exact NaN and exception-interface
  policy and any difference from the paper's benchmark before comparisons.
- Verify against an independent reference, including cancellation, alignment,
  rounding ties, underflow, overflow, and special cases.
- The previous FP12 multiplier experiment is historical work and does not
  satisfy this goal. Do not include its runs in the FP16 adder acceptance count.

## Workflow

0. Generate candidate RTL.
1. Compile and functionally verify with Verilator.
2. Synthesize and map with Yosys/ABC; verify the mapped netlist.
3. Obtain area, delay, and available power estimates with OpenROAD.
4. Record hypothesis, configuration, commands, verification, PPA, and the next
   decision for each optimization iteration. Preserve failed attempts.

## Acceptance criteria

1. Complete more than 10 functionally passing FP16 adder evaluations through
   the workflow, and optimize toward area and performance similar to the
   paper's FP16 **adder** results. Section 7.1 reports area from 58 to 49 um^2
   and delay from 1610 to 1043 ps. Do not assume the best area and delay belong
   to one design, and do not use the FP16 multiplier results as adder targets.
   Record library, corner, mapping constraints, timing assumptions, and any
   differences that limit direct comparison. Report unmet targets explicitly;
   run count alone does not establish completion.
2. Provide `report.md` describing the final results, intermediate experiments,
   verification coverage, PPA, and comparison with the correct paper benchmark.
3. Provide `reproduce.md` with commands to reproduce the three best distinct
   designs under a joint area/performance criterion: minimize mapped area
   times delay, while also reporting the Pareto frontier. Verify fresh replay
   results against the recorded measurements.

## Environment

- Experiment directory: `npu_repo_in_pynq/exp/fp16_adder/`.
- Package environment: `npu_repo_in_pynq/exp/fp16_adder/.venv/`.
- Keep files and tools modular.
- Use the existing Verilator and Yosys installations.
- Use Docker-pulled OpenROAD, as requested; record the image digest.
- Keep build products and local environments out of Git. Publish the work with
  the `feat(exp)` scope and document `exp` in the Git scope rules.
