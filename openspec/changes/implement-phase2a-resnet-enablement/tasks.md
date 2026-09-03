## 1. Tracking and decomposition

- [x] 1.1 Create four dependency-linked Phase 2A implementation sub-issues for graph/numeric contracts, planner/exporter, matrix lowering, and model runtime; verify Issue #6 reads them back as sub-issues with explicit dependency order (#33 -> #34 -> #35 -> #36, with #35 also blocked by #33 and #36 also blocked by #34).
- [ ] 1.2 Validate this tracking change with OpenSpec strict mode, commit it on the Issue #6 branch, and verify its pull request targets dev with Issue #6 and all sub-issues linked.

## 2. Quantized graph and operator contract

- [x] 2.1 Promote Phase 0 integer primitives to production-visible src/model code with compatibility imports, and verify existing numeric tests plus signed endpoint, rounding, saturation, and overflow tests pass (15 focused/existing tests PASS).
- [x] 2.2 Implement immutable tensor, quantization, command, and graph records with complete graph/operator validation; verify unsupported ranks, layouts, operators, parameters, duplicate ids, cycles, and shape mismatches fail before export (7 graph validation tests PASS).
- [x] 2.3 Implement integer-only golden convolution, residual add, ReLU, max pool, global average pool, flatten, and fully connected operators; verify focused vectors match independent scalar references (6 operator tests and 16 combined Phase 2A focused tests PASS).

## 3. Memory planner and deterministic exporter

- [ ] 3.1 Implement deterministic live-interval and 64-byte-aligned first-fit arena planning; verify residual lifetimes, safe reuse, stable offsets, peak size, overflow, and capacity failures.
- [ ] 3.2 Implement per-output-channel accumulator-safety certificates; verify exact boundary acceptance and unsafe convolution/fully-connected rejection.
- [ ] 3.3 Implement stable little-endian weight packing and canonical manifest serialization; verify repeated exports in different directories are byte-identical and contain no host paths or timestamps.
- [ ] 3.4 Implement two-file atomic publication and package structure validation; verify a failed export preserves an existing valid pair and leaves no final partial package.

## 4. Bounded matrix lowering

- [ ] 4.1 Implement on-demand NHWC/HWIO convolution patch lowering with M/N edge tiles and input-zero-point padding; verify fake-runtime calls stay within discovered physical limits and match the golden convolution.
- [ ] 4.2 Implement certified ordered K slicing with INT64 partial assembly, one-time bias, and Phase 0 requantization; verify multi-slice and edge K cases match the unsliced reference and missing/invalid proofs fail before runtime calls.
- [ ] 4.3 Implement fully connected lowering through the same bounded path; verify M=1, N edges, and K slices match the golden fully connected operator.
- [ ] 4.4 Enforce one finite monotonic deadline across all tiles and slices; verify expiration between submissions prevents the next physical call.

## 5. Package loader and model runtime

- [ ] 5.1 Implement full manifest/payload preflight including version, digest, lengths, ranges, references, memory, ABI, capabilities, proofs, and physical limits; verify corrupt or unsupported packages fail before allocation or physical calls.
- [ ] 5.2 Implement deterministic host activation, pooling, residual, and reshape command execution over planned tensor views; verify an integrated residual block equals command-by-command golden output.
- [ ] 5.3 Implement convolution/fully-connected dispatch exclusively through the public Phase 1 runtime with contextual error propagation; verify fake-runtime call traces and mid-model failure recovery behavior.
- [ ] 5.4 Implement owned output and immutable execution metrics; verify repeated changed inputs have no stale data and accounting matches observed commands and physical calls.

## 6. Phase 2A integration and handoff

- [ ] 6.1 Export and execute a deterministic synthetic ResNet-18-shaped operator sequence through a fake physical runtime; verify every supported layer matches the quantized reference and invalid variants fail explicitly.
- [ ] 6.2 Run focused and full Python regressions, make -C src/test lint, make -C src/test sim, strict OpenSpec validation, path/diff/secret/generated-artifact checks, and record exact outcomes in STATE.md.
- [ ] 6.3 Verify all implementation sub-issue PRs are merged into dev, update Issue #6 with reproducible evidence and the Phase 2B handoff contract, and leave physical-board accuracy/performance gates explicitly assigned to Issue #7.
