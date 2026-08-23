## Purpose

Defines a deterministic, self-contained Phase 1C deployment package and board
entrypoint that can be mounted on a PYNQ-Z1 without requiring a repository
checkout on the board.

## ADDED Requirements

### Requirement: Package uses an explicit standalone layout
The package builder SHALL copy only an explicit allowlist of tracked Phase 1C
example sources, shared runtime sources, and generated overlay artifacts into a
standalone directory. The package MUST NOT require the original repository
checkout at runtime.

#### Scenario: Valid package is assembled
- **WHEN** the required tracked sources and generated overlay tuple are present
- **THEN** the output contains the notebook, non-interactive runner, example runtime, shared runtime, and `artifacts/npu_matrix.{bit,hwh,manifest.json}` in deterministic paths

#### Scenario: Required input is missing
- **WHEN** an allowlisted source or overlay artifact is absent
- **THEN** packaging fails without producing a package reported as complete

#### Scenario: Developer assembles a local package
- **WHEN** a developer runs `python examples/matrix-multiplication/package_example.py` after a valid default-path overlay build
- **THEN** the builder imports without `PYTHONPATH`, reads provenance from the overlay manifest, and writes `mount/matrix-multiplication/local-<commit>` without requiring command-line metadata

### Requirement: Package provenance is verified before hardware loading
The board entrypoint MUST validate the packaged BIT, HWH, and manifest before
creating the PYNQ overlay or submitting a matrix job.

#### Scenario: Package artifact was modified
- **WHEN** a packaged artifact size, hash, target, or HWH metadata differs from the manifest
- **THEN** the board entrypoint fails before loading the overlay

### Requirement: Board entrypoint exercises Phase 1C behavior
The standalone board entrypoint SHALL run signed INT8 normal, non-aligned M/N,
and repeated matrix-multiplication cases through the public Phase 1C and Phase
1B runtime boundaries. Each output MUST equal the NumPy reference exactly.

#### Scenario: Required cases pass
- **WHEN** the overlay and runtime execute normally
- **THEN** all three cases match the reference, the non-aligned case uses multiple physical tiles, and the process exits zero

#### Scenario: A required case mismatches
- **WHEN** any board result differs from the reference or runtime execution raises an error
- **THEN** the process exits nonzero and does not emit the final PASS marker

### Requirement: Board evidence is deterministic and non-secret
The board entrypoint SHALL write JSON evidence containing provenance, physical
limits, case dimensions, tile counts, elapsed time, operation counts,
throughput, and a stable PASS marker. It MUST NOT include credentials, private
keys, environment dumps, or unrelated host information.

#### Scenario: Evidence is written after success
- **WHEN** all required board cases pass
- **THEN** the evidence file contains the required fields and can be uploaded unchanged by CD

### Requirement: Generated deployment payload is not source
The assembled package, BIT/HWH files, and board evidence MUST remain ignored
generated content under `mount/` or workflow storage and MUST NOT be committed
to Git.

#### Scenario: Package is prepared for deployment
- **WHEN** packaging completes
- **THEN** all generated outputs are located in ignored staging paths and tracked source remains unchanged
