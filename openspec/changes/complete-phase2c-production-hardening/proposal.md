## Why

Phase 2B defines exact model acceptance, but production operation also needs a
fail-closed lifecycle for compatibility, recovery, soak evidence, release
promotion, rollback, and diagnostics. Issue #8 requires those guarantees across
hardware, exporter, runtime, build, and release without allowing a synthetic or
unaccepted Phase 2B result to become a production baseline.

## What Changes

- Define ABI compatibility, reset, capability, limits, timeout, and typed-failure
  behavior across overlay and model runtime boundaries.
- Add deterministic health snapshots, failure ledgers, bounded recovery, and
  long-running changed-input soak acceptance.
- Bind production releases and evidence to immutable source, tool, overlay,
  model, corpus, runner, and target-board identities.
- Require transactional promotion, independently verified rollback, retention,
  and operational diagnostics with secret-safe output.
- Gate every implementation and production claim on merged, trusted Phase 2B
  acceptance from Issue #7.

## Capabilities

### New Capabilities

- `production-abi-lifecycle`: Version, capability, reset, timeout, failure, and
  compatibility behavior at production runtime boundaries.
- `production-reliability`: Changed-input soak, repeated inference, recovery,
  leak/drift detection, and deterministic regression evidence.
- `production-release-provenance`: Reproducible release identity, evidence
  topology, promotion, retention, and independently verified rollback.
- `production-operations-diagnostics`: Bounded health, counters, failure-ledger,
  environment, and support-bundle diagnostics without credentials or raw data.

### Modified Capabilities

None. The repository has no synchronized main specifications yet; Phase 2C
depends on the Phase 2B delta contracts and must be rebased after they merge.

## Impact

The future implementation will affect `src/hw/`, `src/export/`, `src/runtime/`,
trusted Vivado/PYNQ workflows, standalone delivery under `examples/`, and their
tests. It may extend ABI minor capabilities and evidence schemas, but must not
silently change Phase 0 numeric behavior or accept an ABI-major mismatch.
