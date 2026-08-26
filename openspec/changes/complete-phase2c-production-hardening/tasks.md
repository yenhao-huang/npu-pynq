## 1. Dependency and tracking gates

- [ ] 1.1 Verify Issue #7 is merged and closed, then validate one immutable trusted Phase 2B evidence root containing an approved trained package/corpus and protected PYNQ-Z1 result before any Phase 2C product edit.
- [ ] 1.2 Create dependency-linked sub-issues for ABI lifecycle, diagnostics, reliability soak, and release operations; verify Issue #8 reads them back in implementation order.
- [ ] 1.3 Commit this tracking change and verify its PR targets `dev`, links Issue #8 and its sub-issues, and contains no synthetic evidence or unmerged Phase 2B implementation diff.

## 2. ABI lifecycle and recovery

- [ ] 2.1 Implement explicit compatible-minor negotiation for ABI, required capabilities, runtime limits, and evidence schemas; verify incompatible inputs fail before overlay programming, DMA, or job MMIO.
- [ ] 2.2 Implement one bounded idle/running/failed/recovering/unavailable lifecycle with deterministic reset; verify timeout, DMA, hardware, and reset-failure transitions and no silent work replay.
- [ ] 2.3 Implement stable typed failures with operation context and retryability; verify every failure category is bounded and contains no tensors, credentials, or private paths.

## 3. Health and diagnostics

- [ ] 3.1 Implement a read-only versioned health snapshot; verify it reports release/runtime/ABI/reset/failure state without clearing failures or starting work.
- [ ] 3.2 Implement a fixed-capacity chronological failure ledger with deterministic eviction and dropped-entry accounting; verify ordering across jobs, failures, recovery, promotion, rollback, and revocation.
- [ ] 3.3 Implement canonical allowlisted support bundles; verify unexpected files, environment keys, credential markers, tensor payloads, and private paths fail without publishing a partial archive.

## 4. Long-running reliability

- [ ] 4.1 Implement canonical changed-input soak schedules binding sample order, seed, warmup, count/duration minima, failure points, cadence, and thresholds; verify unchanged-input and partial schedules are rejected.
- [ ] 4.2 Aggregate exact outputs, work, cycles, latency, throughput, traffic, memory/buffer stability, temperature availability, recovery, and drift; verify absolute and slope gates with deterministic fake traces.
- [ ] 4.3 Implement deterministic failure campaigns and post-recovery changed-sample checks; verify a failed invocation publishes no result and stale or unrecovered execution fails acceptance.
- [ ] 4.4 Publish soak evidence transactionally only after both duration and invocation gates pass; verify interruption or any gate failure preserves prior known-good evidence byte-identically.

## 5. Release, promotion, and rollback

- [ ] 5.1 Implement a canonical typed evidence DAG binding source, tools, environment, overlay, model, corpus, runner, board, and direct claims; verify cycles, dangling inputs, trust-class substitution, and identity mismatch fail closed.
- [ ] 5.2 Implement reproducible production manifests and archives from committed sources and external content-addressed assets; verify identical inputs reproduce and tool/environment drift changes identity.
- [ ] 5.3 Implement shared candidate/retained-release verification, transactional promotion, retention protection, and authenticated revocation; verify every mid-operation failure leaves the active selector unchanged.
- [ ] 5.4 Implement independently verified immutable rollback with a new rollback event; verify modified, incompatible, missing-evidence, active-deletion, and revoked targets are rejected.

## 6. Trusted production qualification

- [ ] 6.1 Run focused/full Python, example, Verilator lint, Icarus simulation, strict OpenSpec, path/diff/secret/generated-artifact checks, and a deterministic reduced-fixture soak; record exact results.
- [ ] 6.2 Rebuild from clean merged source with trusted Vivado and verify synthesis, implementation, routed timing, DRC, utilization, BIT/HWH, reports, and evidence-graph provenance.
- [ ] 6.3 Execute the approved changed-input soak and failure campaign on the protected PYNQ-Z1; verify accuracy, cycles, latency, throughput, bandwidth, drift, recovery, diagnostics, promotion, and rollback from observed board evidence.

## 7. Integration and closure

- [ ] 7.1 Merge dependency-ordered sub-issue PRs into `dev`, rerun the complete production qualification on the merged commit, and verify no provisional branch identity remains in release evidence.
- [ ] 7.2 Promote one version and independently roll back to the prior retained version through the protected workflow; verify both immutable evidence graphs and selector events.
- [ ] 7.3 Update and close Issue #8 only after all acceptance criteria and dependency links read back from GitHub with reproducible evidence.
