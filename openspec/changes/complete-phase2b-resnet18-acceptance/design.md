## Context

The Phase 2A package is deterministic and fail-closed, and its runtime lowers
matrix work only through `NPURuntime.run`. ResNet-18 acceptance adds two kinds
of inputs that cannot live in Git: trained quantized weights and a labeled
validation corpus. It also needs physical cycle, synthesis, timing, resource,
and board provenance that host tests cannot fabricate.

The existing ABI already provides stable `CYCLES_LO` and `CYCLES_HI`
registers. The runtime currently discards them, so Phase 2B can expose cycle
telemetry without changing RTL or the model-package format.

## Goals / Non-Goals

**Goals:**

- Bind every acceptance input and result to cryptographic digests.
- Reject incomplete, substituted, stale, or topologically non-ResNet-18 assets
  before executing a physical job.
- Produce exact host reference and PYNQ-Z1 evidence from the same bundle.
- Measure end-to-end and physical work without changing quantized arithmetic.
- Make repeated execution and recovery objective acceptance gates.

**Non-goals:**

- Train, download, redistribute, or silently select a model or dataset.
- Claim ImageNet accuracy from a synthetic corpus.
- Change the Phase 0 numeric contract, ABI major, RTL, overlay topology, or
  Phase 2A package encoding.
- Commit BIT/HWH files, generated Vivado projects, credentials, datasets,
  weights, or board evidence.

## Decisions

### 1. Acceptance assets are external and content-addressed

An acceptance bundle directory contains a small canonical JSON descriptor and
references a Phase 2A `.npu.json/.npu.bin` pair plus one NPZ corpus. The corpus
contains signed INT8 `inputs`, signed INT8 `expected_outputs`, integer `labels`,
and stable `sample_ids`. The descriptor records every byte length and SHA-256,
reference framework/version, preprocessing identifier, class count, sample
count, and required accuracy threshold. Paths must be relative basenames and
must not escape the bundle directory.

The loader validates all files and arrays before loading the model or creating
the runtime. Unknown descriptor fields, duplicate JSON keys, object arrays,
pickle, non-finite thresholds, and inconsistent sample shapes fail explicitly.

### 2. Canonical topology is validated independently of tensor names

The validator recognizes a stem convolution/ReLU/max-pool, four stages with
two basic blocks each, one projection shortcut in the first block of stages
2-4, global average pool, flatten, and fully connected output. It checks 20
convolutions, eight residual additions, 17 ReLUs, one max pool, one global
average pool, one flatten, and one fully connected command, plus branch
dependencies and stage shape transitions. Stable names are reported but are
not used as proof of topology.

### 3. Cycle telemetry preserves the NumPy return contract

`NPURuntime.run` continues returning an owned INT32 NumPy array. After a
successful DONE result it reads the high/low cycle pair consistently and sets
an immutable `last_metrics` record. The lowerer snapshots this record after
each physical call and sums cycles when every call exposes compatible
metadata. Fakes or older public runtimes without metrics remain supported and
report cycles as unavailable. Model metrics sum lowerer cycles without
inventing missing values.

### 4. Layer capture is explicit and bounded

`NPUModelRuntime.run` accepts an optional tuple of declared tensor names. It
copies only those tensors after their producing command and returns an
immutable mapping of owned C-contiguous INT8 arrays. Unknown, input, constant,
or duplicate capture names fail during input preflight. Normal execution pays
no full-graph capture cost.

### 5. One runner produces canonical evidence

The acceptance runner loads one validated bundle and executes samples in
stable corpus order. It compares every captured tensor and final output
exactly, derives top-1 accuracy from logits and labels, performs a configured
repeat pass, and reports latency distribution, throughput, physical jobs,
MACs, operations, and cycles. Host mode uses a deterministic matrix fake and
is evidence of software correctness only. Board mode requires matching
BIT/HWH hashes and records environment, source commit, package/corpus digests,
ABI, capabilities, runtime limits, and timing/resource report digests.

Evidence is canonical JSON written through a temporary file and atomically
promoted only after all required gates pass. A failed run leaves the previous
known-good evidence and deployment target unchanged.

### 6. Trusted hardware gates remain explicit

Synthesis, implementation, routed timing, DRC, resource use, and PYNQ-Z1
execution must come from trusted runners. The local implementation supplies
validation and delivery mechanisms, but Phase 2B remains incomplete until the
approved external bundle, licensed Vivado runner, and protected board runner
produce matching evidence.

## Risks / Trade-offs

- Exact INT8 output comparison is stricter than floating-point tolerance and
  may reveal exporter/reference drift; this is intentional under Phase 0.
- A full validation corpus can be slow on a 2x2 accelerator. The bundle makes
  sample count and thresholds explicit so smoke evidence cannot masquerade as
  acceptance evidence.
- Reading split cycle registers can tear at rollover. The runtime reads high,
  low, then high again and retries within the existing software deadline.
- External asset availability can block completion. The blocker is reported
  with expected digests rather than replaced by synthetic success.

## Migration Plan

1. Land Phase 2A into `dev`.
2. Add cycle telemetry and capture behavior with backward-compatible tests.
3. Add bundle/topology validation and host acceptance using generated fixtures.
4. Add the standalone board delivery path and dry-run tests.
5. Run trusted CI/Vivado/PYNQ gates with the approved external bundle.
6. Promote evidence transactionally; rollback selects the prior immutable
   deployment and evidence pair.
