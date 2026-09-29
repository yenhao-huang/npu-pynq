# Operation and negative-control review

This review checks engineering purpose, implementation behavior and exercised
failure paths. Registry names, schemas and passing inventory checks alone do
not establish these properties. The operation-level review covers all 50
operations. Historical physical coverage and the declared all-case benefit
remain open.

## Original ten operations

The [larger workflow](evidence/wide-workflow.json) exercises every operation on
the same CSD16 cores used by the registered physical study. The exhaustive check
covers all 65,536 binary inputs. Its combinational PPA records support integration
testing and stay outside the registered comparison aggregate.

| ID | Distinct engineering purpose and bound | Exercised negative control |
| --- | --- | --- |
| 01 | Retrieve attributed rewrite advice and prerequisites; it does not rewrite or prove RTL. | Unknown topic returns no invented rule. |
| 02 | Compute exact add/subtract/multiply interval bounds for independent mathematical integers; caller must establish ranges and preserve RTL signedness. | 23 output bits cannot contain the CSD16 product; overflow is reported. Unit cases also cover negative ranges and zero. |
| 03 | Enumerate the caller-declared binary combinational interface, up to 16 input bits; require complete ordered transcripts and defined outputs. | Highest output-bit mutation fails at input zero; unknown outputs and incomplete execution are rejected by tests. |
| 04 | Run Vivado and collect routed combinational LUT/FF counts and input-to-output delay. This is a different scope from registered timing. | Sequential input is rejected; a timed-out process cannot produce an accepted record. |
| 05 | Compare declared measurement context; it does not authenticate external reports. | Mismatched part is incompatible; version mismatch is also tested. |
| 06 | Compute signed resource/delay deltas and percentage reductions after context/unit checks; a zero baseline has no percentage. | Wrong units are rejected; zero-baseline tests require a null ratio. |
| 07 | Compute a Pareto frontier with strict dominance, preserving equal points and trade-offs. | Duplicate candidate identities are rejected; a slower/smaller trade-off is not discarded by scalar ranking. |
| 08 | Form a weighted improvement score gated by correctness and hard limits. This is an adaptation, not a reproduced training loss. | Failed correctness yields ineligible/null reward; missing metrics and hard-limit violations are tested. |
| 09 | Choose feasible actions with cost-normalized UCB and deterministic unseen-action/tie handling. Estimated cost does not enforce execution time. | Insufficient budget selects nothing. Three finite-input arithmetic-overflow cases now reject selection. |
| 10 | Compose checking, measurement and comparison with source-hash binding and early failure. This is more than a renamed single operation. | Wrong RTL produces only the failed check child; source changes between checking and measurement are rejected in tests. |

Implementation inspected: the lint advice backend, exhaustive checker,
combinational measurement backend, PPA analytics backend and RTLRewriter pipeline.
Relevant regression: `tools/ic/tests/test_exploration.py`, 28 passing tests after
the selector correction. Paper connections and adaptation limits remain in the
per-operation READMEs, [survey](survey.md) and operation inventory.

## Selector issue found during review

Development run `68600c` returned an infinite UCB score from finite reward/cost
inputs and selected that action. This result is rejected evidence. The backend
now checks the calculated scores and handles integer-to-float overflow before
choosing an action. It does not clamp or silently rank an invalid score.

[Actual controls](evidence/selector-finite-controls.json) cover division overflow,
addition overflow and oversized visit-count conversion. The same runner also
rechecks a normal selection using the committed CSD study's measured reward and
elapsed cost. Run:

```sh
python exp/tool-exploration/exp-tool-09-rtlrewriter/controls.py
```

These results close this specific numerical issue. They do not complete the
remaining operation review, whole-case benefit or physical-evidence gates.

## Operation 11: routed clocked measurement

`clocked_ppa` performs a real single-clock, register-to-register out-of-context
Vivado implementation. It reports LUT/FF/DSP/BRAM, requested setup slack, an
estimated critical period, caller-declared latency and II-normalized throughput.
It does not verify protocol, close hold or I/O timing, measure power, or claim
board frequency. Acceptance composes it with source-bound correctness and
`timing_constraint_audit` rather than trusting it alone.

The modular control audits one completed 16-bit, eight-lane dot-product record:
the source hash, routed reports, single clock, all registers, setup path and six
`check_timing` classes agree. An actual Vivado source with no declared `clk`
port fails without timing out (run 14046f) and produces no usable measurement.
See `clocked-ppa-controls.json`. Adversarial tests reject mixed run handles,
changed source, build/period/slack disagreements, incomplete reports,
unconstrained endpoints and active/unknown DSP storage. Latency and II remain
caller-verified inputs and are independently checked by sequential scoreboards.

## Complete SAT interface review (operation 13)

Development run `0ccec1` incorrectly accepted a candidate whose extra output `z`
differed from the reference: the miter observed only `x` and `y`. That run is
explicitly rejected in interface-development-failure.json. The backend now
requires both pre-optimization source netlists to contain exactly input `x` and
output `y`, at the declared widths. A successful SAT result cannot override this
check. Extra unused inputs, extra outputs, inout ports and width mismatches fail.

The Docker Yosys 0.23 controls cover all six normalization modes: 48 interface
cases and 60 arithmetic/undefined-value cases match their expected verdicts.
The historical audit now re-elaborates 98 source files from 49 successful SAT
study proofs, after matching each file set to its recorded source hash. All have
complete declared interfaces. The same pre-optimization netlists pass the
current total-binary semantics check. Native 32/64-bit barrel shifters use a
Yosys `$pmux`; that cell is accepted only when every selector is driven by a
distinct equality comparison against the same binary bus and the comparisons
exhaust every value. Duplicate, missing or uncontrolled selections reject. This
is not a new cross-design SAT proof or a retrospective change to the recorded
tool result; it binds the old result to the current source-safety contract.

Reproduce with `exp/tool-exploration/exp-tool-13-rover/interfaces.py`,
`bitwise.py` and `audit_interfaces.py`, each with
`--container codex-sandbox-agent-workspace`. See interface-controls.json,
normalization-interface-controls.json and historical-interface-audit.json in
`evidence/`. Operation 13's identified interface issue is fixed; review of the
remaining operations and historical physical coverage is still incomplete.

## Operation 15: exact affine proof and source semantics

Development run 14c87d incorrectly accepted `x[x] ^ x[x]` on an 8-bit input:
out-of-range indexing was simplified out before coefficient propagation. This
result is rejected, not accepted equivalence evidence. The affine checker now
composes `yosys_equivalence` self-checks for each source with its coefficient
algorithm. Each self-check requires complete interfaces, total binary source
semantics and matching source hashes before netlist optimization. A self-check
never replaces the cross-design affine proof. Runtime limits apply to each child
proof or synthesis process; a failed or unknown guard returns no affine proof.

Eight actual source controls match their expected verdicts, including cancelled
out-of-range indexing in either source separately. Valid constant division by
one passes. The original CRC64 pair still proves, a different polynomial fails,
and its concrete counterexample replays in Icarus. Nonlinear AND still rejects.
All four distinct historical CRC/LFSR source pairs (eight recorded study proofs)
prove again with the new guards, after validating original hashes. Old studies
and physical measurements remain unchanged; these are additional proof records.

```sh
python exp/tool-exploration/exp-tool-15-gf2/source_controls.py --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-15-gf2/recheck_studies.py --container codex-sandbox-agent-workspace
python exp/tool-exploration/exp-tool-15-gf2/study.py
```

See `affine-source-controls.json`, `affine-source-rechecks.json`,
`affine-guarded-controls.json` and `affine-source-development-failure.json` under
`docs/goals/0928-tool-exploration/evidence/`. No new operation is counted.

## Operations 12 and 14: wide vectors and modular reductions

| ID | Distinct purpose and contract | Exercised failure path |
| --- | --- | --- |
| 12 | Simulate directed all-zero/one, walking-bit/carry patterns plus seeded random vectors on 1..4096-bit packed interfaces. Require a caller-confirmed complete combinational contract, all ordered transcript rows, defined outputs and unchanged sources. This is sampling, not proof or automatic verification of the caller's contract. | Subtraction replacing addition, unknown outputs and truncated input widths reject; process timeout/incomplete transcripts reject. New 16/32-bit reduction mutations give concrete input/output mismatches. |
| 14 | Generate unsigned sums modulo 2**width, with width 8..128 and lanes 3..128. Serial, balanced and carry-save implementations are architectures of one operation. Width truncation at each node preserves the modular sum; leftover nodes are forwarded at each level. The same registered wrapper supplies latency 1/II 1. | Replace the first balanced addition by subtraction, or shift the first compressor carry by two instead of one. Both fail at both original widths with concrete vector mismatches. Odd lane counts 3/7 and declared large 16-lane cases also pass existing integration tests. |

`exp/tool-exploration/exp-tool-14-rover/controls.py` runs eight actual checks at
16 and 32 bits with 16 lanes. All four correct candidates pass and all four
mutations fail with observed data mismatches (not compilation errors/timeouts).
See reduction-mutation-controls.json. The four existing physical comparisons
remain complete but do not qualify the family: balanced gains are small and
compressor resource growth exceeds the throughput gate. This review adds no
PPA win, no new operation and no sampled-equivalence claim of formal proof.

## Operations 16-18: prefix and constant arithmetic graphs

| ID | Distinct purpose and contract | Exercised failure path |
| --- | --- | --- |
| 16 | Generate unsigned `a+b+cin` with a full carry bit. Native addition, Kogge-Stone and Sklansky are alternative prefix topologies over widths 8..128. Input packing and the one-cycle wrapper are explicit; names do not imply timing gains. | Replacing the first prefix generate-combine OR with AND fails at both 16 and 32 bits. Carry-boundary integer oracles cover all three architectures at 32/64 bits. |
| 17 | Generate a full unsigned product by a positive compile-time constant. Native multiplication, binary shift/add and canonical signed digits are alternatives; signed-digit subtraction uses the complete product width. | Replacing the first CSD subtraction with addition fails for constants 255 and 65535. Coefficient decomposition checks 1..65535 and integer oracles cover zero, maximum inputs and isolated bits. |
| 18 | Generate several packed constant products of one input. The shared variant memoizes powers and factors of `2^k +/- 1`; it is a bounded factoring heuristic, not a globally optimal MCM solver. Coefficients must be distinct and positive. | Replacing a shared graph addition with subtraction fails at both 16 and 32 bits. Duplicate coefficients reject; independent oracles verify packing and all four declared constants. |

`exp/tool-exploration/exp-tool-16-prefixllm/controls.py` runs twelve actual
wide-vector checks over the original substantial configurations. All six correct
architectures pass and all six structural mutations produce concrete mismatches.
See `arithmetic-graph-mutation-controls.json`. CSD and MCM have qualifying
physical evidence at both sizes. Prefix adders regress physically and stay in
the denominator. Architecture choices are not counted as extra operations.

## Operations 19-22: signed datapaths and iterative arithmetic

| ID | Distinct purpose and contract | Exercised failure path |
| --- | --- | --- |
| 19 | Generate a signed FIR arithmetic window. The caller supplies sample history; this is not a streaming delay line. Direct and symmetric forms retain a full-precision result, with one extra preaddition bit and palindromic coefficients required for symmetry. | Broken preaddition is rejected by vectors. Oracles cover signed extrema, negative/zero coefficients, odd taps and maximum coefficients. Formal equivalence for both declared comparisons remains unknown, so no PPA result is accepted. |
| 20 | Generate a signed vector dot product with full-width products and sign extension before serial, balanced or carry-save accumulation. Lane packing and logic-only multiplier policy are explicit. | Removing signed interpretation fails. Oracles cover signed extrema and non-power-of-two lane counts. Both physical comparisons complete, but 9.30%/6.18% throughput gains miss the 15% gate. |
| 21 | Generate exact signed full products using native, radix-2 or radix-4 Booth rows, including odd widths. Sign extension precedes negation to preserve the most-negative operand. | Mutated sign extension and radix-4 negative-two decoding fail. Tests cover widths 8,16,17,32,64, signed extrema and runs of ones. Both declared formal comparisons remain unknown. |
| 22 | Generate single-outstanding ready/valid unsigned multipliers with parallel, one-bit and two-bit iterative reuse. Reset cancels work, output holds under backpressure, and latency/II include setup cycles. | Independent scoreboards reject result, reset, hold, latency and II mutations. Width edges 8/17/64 pass; fixture observation delay is checked separately. |

The focused operation 19-22 regression passes 90 tests. This establishes the
documented arithmetic and cycle contracts, including negative controls. It does
not resolve FIR/Booth SAT timeouts: simulation and integer oracles are not formal
proofs. Iterative multiplier physical results qualify only for the one-bit to
two-bit digit comparison; severe folding throughput losses remain visible.

## Operations 23-28: count, selection and shift networks

| ID | Distinct purpose and contract | Exercised failure path |
| --- | --- | --- |
| 23 | Count every asserted bit, returning `log2(width)+1` bits. Linear accumulation and balanced width-growing trees are alternatives for power-of-two widths 8..1024. | Replacing a tree addition with subtraction fails at 64 and 128 bits. |
| 24 | Return the highest asserted index with a valid MSB; zero input returns zero. Linear priority and hierarchical valid/index reduction preserve the same tie rule. | Replacing a valid-combine OR with AND fails at 64 and 128 bits. |
| 25 | Count leading zeros and return the complete width for zero. Linear priority, recursive tree and binary-search narrowing are architectures of one operation. | Inverting a tree's high-half-empty test fails at 64 and 128 bits. Binary-search oracles independently cover both sizes. |
| 26 | Perform a logical left shift modulo word width. The input packs data and the complete shift amount; decoded and logarithmic staged networks are alternatives. | Changing the first staged left shift to right shift fails at 16 and 32 bits. |
| 27 | OR selected lanes with total semantics: one-hot selects one word, multi-hot returns bitwise OR and zero-hot returns zero. Linear and balanced OR networks differ only structurally. | Replacing the first tree OR with XOR fails at 16 and 32-bit lanes; explicit multi-hot vectors distinguish the semantics. |
| 28 | Return the unsigned maximum with the lowest lane index on ties. Linear comparison chains and balanced tournaments preserve that stable tie rule. | Replacing `>=` with `>` fails at 16 and 32-bit lanes on explicit equal-maximum cases. |

`exp/tool-exploration/exp-tool-23-rover/controls.py` executes 24 actual controls
over the declared substantial sizes. Twelve correct candidates pass and twelve
architecture-specific mutations produce concrete mismatches. See
`network-mutation-controls.json`. Independent integer-oracle tests additionally
exercise all-zero/one, every isolated input bit and seeded values per case.

## Operations 29-38: transforms, storage and stateful datapaths

| ID | Distinct purpose and contract | Exercised failure path |
| --- | --- | --- |
| 29 | Generate MSB-first CRC next-state transforms as an unrolled recurrence, GF(2) matrix, or shared XOR graph. State/data packing and polynomial width are explicit. | An independent bit-serial oracle checks both declared 64-bit data cases; a changed polynomial fails. |
| 30 | Generate exact multi-step LFSR jumps by recurrence, GF(2) matrix exponentiation, or shared XOR graphs. | Independent recurrence models cover 16/32/64-bit states and reject altered taps or step counts. |
| 31 | Generate unsigned quotient/remainder with parallel division or one/two-bit restoring iterations. Division by zero returns all-one quotient and the dividend remainder. | The cycle oracle rejects quotient, divide-by-zero, reset, output-hold, latency and II mutations at substantial widths. |
| 32 | Reduce an unsigned word to the canonical residue modulo `2**k-1` with native `%` or bounded end-around folding. | Changing canonical `>= modulus` correction to `> modulus` returns the forbidden modulus encoding and fails at exact multiples. |
| 33 | Generate signed or unsigned saturating add/subtract with dual or shared arithmetic. Result and overflow are both contractual outputs. | Suppressing overflow or changing signed interpretation fails; tests cover both polarities and signed extrema through 64 bits. |
| 34 | Generate shift-register or circular-buffer FIFO storage with full-throughput simultaneous pop/push. | Independent queue traces reject ordering, reset flush, capacity, fixture delay and memory-policy mutations at 16x64 and 32x128. |
| 35 | Generate two-read/one-write register files using replicated registers or banked storage with an explicit same-bank conflict rule and write-first behavior. | Cycle models reject conflict, address mapping, reset and write-first mutations; all addresses are visited for the declared 16x64 and 32x128 geometries. |
| 36 | Generate cyclic phase controllers with binary or one-hot state. `advance` and reset define the complete transition contract. | Scoreboards visit every state through 256 states and reject direction, hold, reset and timing-fixture mutations. |
| 37 | Generate elastic or skid-buffer pipelines under ready/valid backpressure. Capacity differs by architecture and is reported rather than hidden. | Independent traces reject loss, duplication, spill, reset, capacity, latency and backpressure mutations at 8 and 16 stages. |
| 38 | Generate signed parallel or systolic matrix products with exact accumulation width and an explicit batch ready/valid contract. | Matrix oracles reject skew, forwarding, signedness, accumulation, reset, hold, latency and II mutations, including signed extrema. |

The focused regression passes 227 tests. CRC/LFSR affine proofs are exact only
after their guarded source checks. Stateful operations use bounded independent
cycle models; those checks establish the declared traces and performance
contracts, not unbounded sequential equivalence.

## Operations 39-50: attribution, measurement and study control

| ID | Distinct purpose and contract | Exercised failure path |
| --- | --- | --- |
| 39 | Produce a source-bound generic or XC7 Yosys netlist and operator/resource histogram. Generic operators are structural estimates, not routed area. | Missing Yosys, synthesis failure, source changes and malformed netlists cannot return an accepted profile. |
| 40 | Trace endpoint fan-in to state boundaries and compute unweighted combinational cell depth. It does not claim a routed critical path. | Unknown endpoints and combinational cycles reject instead of yielding a finite depth. |
| 41 | Count connected sink pins and identify their drivers in a digest-bound netlist. It reports load count, not capacitance or delay. | A modified netlist digest fails; constants are excluded and clock/control sinks are retained. |
| 42 | Classify logical memories, block/distributed RAM primitives, shift registers and flip-flop output bits separately. | Malformed dimensions and modified netlists fail; register storage is not silently relabeled as inferred RAM. |
| 43 | Bind a routed Vivado clocked record to source hashes, report handles, clock/register coverage, setup paths and six `check_timing` classes. | Missing/constant/multiple clocks, unconstrained endpoints, mixed handles, changed sources and active unknown DSP registers fail closed. |
| 44 | Check arithmetic or matrix ready/valid results plus declared first-valid latency and sustained II against an independent transaction model. | Result, reset, hold, latency and II mutations fail; incomplete or unknown-valued transcripts cannot produce cycle evidence. |
| 45 | Apply the 15% LUT or throughput gates after normalizing throughput by II and checking matched physical context and resource limits. | Identical sources, incompatible contexts, invalid metrics, changed latency kind and an apparent Fmax gain erased by II all fail the gate. |
| 46 | Summarize distinct paired repeats conservatively: every one of at least three repeats must pass and the geometric objective ratio retains the worst evidence upstream. | Duplicate report handles, source/context/cycle changes and a single losing repeat reject qualification. Zero-LUT candidates use a stated finite lower bound. |
| 47 | Compute all conditional, main and interaction contrasts for a complete one-to-four-factor binary physical design. | Missing/duplicate/confounded cells, mismatched hashes, unequal repeats and reused physical handles reject; declared failed cells remain visible and suppress effects. |
| 48 | Verify FIFO, phase, register-file and stream protocols with independent cycle models and source identity. | Ordering, capacity, conflict, state, reset, stall and fixture-delay mutations fail with the first observed mismatch. |
| 49 | Run generation, correctness, cycle checking and physical measurement with environment-keyed checkpoints and bounded retries. | Changed checkpoint files, source mismatch, concurrent writers, failed proofs/cycle checks and exhausted measurements stop downstream acceptance. |
| 50 | Enumerate every declared case and every evidence file, bind generated/check/physical sources, retain failed and retry observations, then apply diversity and all-case gates. | Missing declarations, synthetic records, duplicate evidence, changed sources, failed correctness, cycle mismatch and favorable-retry cherry-picking cannot supply an aggregate ratio. |

These analysis operations intentionally make narrower claims than their names
might suggest. Netlist depth and fanout are structural diagnostics; only routed
clocked records enter PPA gates. Repeats measure reproducibility under matched
conditions and are not independent statistical samples. The acceptance audit
still reports historical timing records without the newer coverage handles as
unaudited, so operation review does not convert the current study into a whole
project success.

The review found that `acceptance_audit` trusted historical proof success and
source hashes without directly requiring complete-interface and total-binary
source evidence. New proofs must carry both fields. Historical SAT proofs now
require a source-hash-bound re-elaboration row, and historical affine proofs
require a guarded affine recheck for the exact source pair. All 98 audited SAT
source interfaces pass the stricter contract, so the current complete physical
comparison count remains 55.

## Historical physical-record authenticity

The current acceptance snapshot encounters 168 pre-coverage records in its
valid observations. Across the full evidence tree, 182 unique historical run
handles are preserved, including older retries. The independent historical
record audit matches every one to exactly one successful local `clocked_ppa`
run. It verifies the exported record byte-for-byte, every measured source file
and combined source fingerprint, the recorded Vivado identity, parsed resource
and timing metrics, matched Tcl flow, declared artifact sizes, and SHA-256
digests of all raw reports and Tcl scripts.

This establishes local report provenance for the preserved runs. It does not
close timing coverage: those in-memory Vivado runs did not emit register/clock
coverage or `check_timing` reports and saved no routed checkpoint from which to
reconstruct them. They remain explicitly `historical_unaudited_records` in
operation 50. See `historical-record-authenticity.json`.

The declaration-ordering audit matches 57 distinct architecture-sweep evidence
envelopes byte-for-byte to their preserved parent runs. Across 362 unique
successful physical children, every child starts after the parent input has
fixed the case name, generator, objective and complete baseline/candidate
payloads; generation checkpoints repeat those exact payloads. Failed and
verify-only studies remain included. This establishes ordering within the
preserved workflow, while making no claim about deleted external history or
when an idea was first conceived. See `predeclaration-ordering.json`.
