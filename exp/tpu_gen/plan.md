# TPU-Gen proof-of-concept plan

## Objective

Reproduce the TPU-Gen pipeline as a modular local experiment under exp/tpu_gen:
user request -> formatted prompt -> header generation -> RTL retrieval ->
elaboration -> OpenROAD place and route -> GDSII and measured PPA. Each
iteration has its own run directory and can feed validation errors into the
next prompt.

This is an ASIC Nangate45 study. It does not modify the PYNQ FPGA overlay or
establish its LUT, DSP, BRAM, or board performance.

## Boundaries and inputs

- OpenROAD is the sole source of area, worst setup slack, and total power.
  Yosys runs inside OpenROAD Flow Scripts (ORFS); it is not used as a substitute
  PPA estimator. Missing metrics or failed routing terminate the attempt.
- The default design is a 4x4 systolic array with 8-bit operands. Larger
  arrays need separate validation because routing time and memory grow.
- Restore the upstream ACADLab/TPU_Gen checkout at commit
  0abb8511369f859d3286d31caab2c37ab6bddde9. The parent README gives
  the exact archive extraction commands. Upstream RTL and datasets stay
  outside this PR.
- The validated ORFS default is openroad/orfs:26Q3-605-g2d29bdaf8 with the
  Nangate45 platform. The image must already be installed; the flow does not
  download it.
- LEC_CHECK=0 and SEC_CHECK=0 work around a local CPU instruction failure in
  kepler-formal. They do not replace placement, routing, GDSII, or PPA.
- The flow defaults to a 5 ns clock constraint and 30% core utilization.

## Design

The DesignState contract carries the request, target PPA, generated header,
parsed macros, retrieved Verilog files, GDSII path, measured PPA, errors, and
iteration number. PPA accepts only source=openroad.

| Module | Responsibility |
| --- | --- |
| prompt_formatter | Parse the request and format the description, target metrics, and later feedback. |
| llm | Produce a Verilog header through replay, local CLI, or optional API backends. |
| retrieval | Parse active macros and find the transitive RTL module dependencies. |
| synthesis | Stage source files and run Icarus Verilog elaboration before ORFS. |
| openroad | Run ORFS in Docker and read final GDSII and measured report fields. |
| validate | Check macros and summarize elaboration or ORFS errors for feedback. |
| flow | Manage iterations, run directories, result records, and stop conditions. |

The replay backend selects the nearest record from the 4,000-row upstream
training JSON. It exercises the pipeline without an LLM service, but a changed
nearest-neighbor match is not evidence that feedback improved PPA. The CLI
backends run an already signed-in Claude or Codex command without tool access;
the API backends require their respective environment credentials.

Retrieval starts at systolic_array_top and follows module dependencies from
the active header. The full mode stages the whole RTL library as a comparison.
Numeric macros such as M, N, DW, and WW do not themselves select files. A
header must pass macro checks and elaboration before physical design begins.

The OpenROAD metric mapping is:

| Reported measure | ORFS final-report key |
| --- | --- |
| Cell area (square micrometers) | finish__design__instance__area |
| Worst setup slack (ns) | finish__timing__setup__ws |
| Total power (W) | finish__power__total |

Each attempt writes to runs/<run-id>/iter_NN/. These generated logs, staged
RTL, GDSII files, and state JSON remain local and ignored. The versioned
docs/results.md records selected archival measurements and their limitations.

## Validation and limitations

The original local study produced a GDSII and PPA for 2x2 and 4x4
DRUM_APTPU/APPROX5 configurations. A 4x4 run retrieved 21 of the 97 upstream
RTL files; its area was 36,300 square micrometers, WNS was +2.4452 ns, and
total power was 0.0138475 W. The selected run record is described separately
in docs/results.md. The present PR checks the Python units and repository
RTL gates; it does not rerun OpenROAD.

The proof of concept checks syntax and elaboration. It does not supply a
functional testbench or formal proof for the upstream TPU RTL. It does not
reproduce the paper's large ResNet or VGG studies, fine-tune an LLM, or claim
that a generated design meets a target merely because the header was accepted.

## Risks

- ORFS can exhaust the original 3 GB test host; keep the default at 4x4.
- ORFS metric names can change with image versions; the parser should fail
  clearly when expected final-report fields are missing.
- Upstream RTL or an LLM-generated macro combination can fail elaboration.
  Feed the actual error into the next iteration rather than inventing PPA.
