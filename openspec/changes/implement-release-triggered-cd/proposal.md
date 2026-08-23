## Why

The repository can build an overlay and run host-side examples, but it has no
single release-gated path that produces provenance-checked deployment assets,
installs a self-contained example on the PYNQ-Z1, and records objective board
evidence. Release publication must become the only production CD trigger so
ordinary branch pushes cannot rebuild or overwrite the board.

## What Changes

- Replace tag-push bitstream automation with a workflow triggered only by a
  published GitHub Release.
- Check out the release tag, prove that its commit is contained in `main`, and
  run Vivado only on a trusted self-hosted runner.
- Verify and publish the matching BIT, HWH, manifest, and build evidence as
  deterministic Release assets.
- Assemble a standalone matrix-multiplication deployment package from tracked
  example/runtime sources and generated overlay artifacts.
- Add a non-interactive board entrypoint that verifies provenance, runs normal,
  non-aligned, and repeated Phase 1C cases, and writes machine-readable
  evidence.
- Deploy and validate the package only from a trusted board-capable runner;
  credentials, generated projects, and bitstreams remain outside Git.

## Capabilities

### New Capabilities

- `release-continuous-deployment`: Defines release-only triggering, immutable
  tag validation, trusted Vivado/board runner boundaries, Release asset
  publication, and board evidence handling.
- `matrix-example-board-package`: Defines the deterministic standalone package
  layout and non-interactive Phase 1C board execution contract.

### Modified Capabilities

- None.

## Impact

- GitHub Actions workflows under `.github/workflows/`.
- The Phase 1C matrix-multiplication example and its host tests.
- Shared overlay provenance/runtime code under `src/runtime/` as consumed by
  the board entrypoint.
- Self-hosted runner configuration: one trusted Vivado-capable runner and one
  trusted runner with SSH access to the PYNQ-Z1 (which may be the same host).
- GitHub Release assets and board-validation evidence; no numeric, register-map,
  AXI, or RTL behavior changes.
- Existing `npu_matrix` RTL lint cleanup required for the repository CI gate;
  explicit width handling, complete combinational defaults, and narrowly scoped
  annotations MUST preserve the current hardware behavior.
