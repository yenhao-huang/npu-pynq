## 1. Tracking and decomposition

- [ ] 1.1 Create dependency-linked sub-issues for cycle/capture telemetry, acceptance assets/topology, host acceptance, and board delivery; verify Issue #7 reads them back in that order.
- [ ] 1.2 Strictly validate and commit this tracking change; verify its PR targets `dev`, links Issue #7 and its sub-issues, and contains no Phase 2A-only diff after dependencies merge.

## 2. Runtime observability

- [x] 2.1 Read the ABI cycle-counter pair consistently after successful jobs and expose immutable physical metrics without changing the public NumPy result contract; verify rollover, failure, timeout, and recovery behavior.
- [x] 2.2 Aggregate physical cycles through matrix lowering and model execution when available, and report unavailable if any physical result lacks telemetry; verify job/MAC/operation/cycle accounting against fake traces.
- [x] 2.3 Add explicit bounded tensor capture to model execution; verify selected layer tensors are owned, immutable by mapping, exact, and absent unless requested.

## 3. Acceptance assets and topology

- [x] 3.1 Implement a canonical, duplicate-key-safe acceptance descriptor with relative path confinement, exact lengths/digests, reference provenance, thresholds, and fail-closed NPZ loading without pickle.
- [x] 3.2 Implement name-independent canonical ResNet-18 topology validation, including eight basic blocks and three projection shortcuts; verify valid generated topology and structurally similar invalid variants.
- [x] 3.3 Add deterministic test-fixture generation for a reduced-shape complete ResNet-18 graph and corpus; keep third-party weights and datasets outside Git.

## 4. Host acceptance

- [x] 4.1 Implement stable sample-order inference, exact final/layer comparison, top-1 accuracy, repeatability, latency distribution, throughput, bandwidth, and model-work aggregation.
- [x] 4.2 Write canonical evidence atomically only after all configured host gates pass; verify a failure preserves prior evidence and never reports a partial success.
- [x] 4.3 Verify the reduced complete ResNet-18 fixture passes through export, bundle validation, host reference, fake physical runtime, and repeated acceptance end to end.

## 5. Standalone board delivery

- [x] 5.1 Add `examples/resnet18/` packaging with an explicit source allowlist, manifest verification, path-free archives, and no committed generated artifacts.
- [x] 5.2 Add dry-run and PYNQ-Z1 board runners that verify BIT/HWH/model/corpus/source digests, ABI/capabilities/limits, recovery, and atomic deployment promotion.
- [x] 5.3 Record synthesis, implementation, timing, DRC, utilization, accuracy, latency, throughput, bandwidth, cycles, repeated inference, and recovery in one provenance-bound evidence document.

## 6. Validation and handoff

- [x] 6.1 Run focused/full Python, example tests, RTL lint/simulation, strict OpenSpec, path/diff/secret/generated-artifact checks, and record exact results.
- [ ] 6.2 Run trusted Vivado and PYNQ-Z1 acceptance with the approved external bundle; keep this task open with exact runner/asset blockers until all required evidence passes.
- [ ] 6.3 Merge all sub-issue PRs into `dev`, update and close Issue #7 with reproducible evidence, and hand the immutable accepted artifact/evidence contract to Phase 2C Issue #8.
