## ADDED Requirements

### Requirement: Composable paper-informed operations
The toolchain SHALL expose ten new typed operations through its registry, with
at least eight supporting PPA optimization and primary-paper attribution.

#### Scenario: Client discovers the tools
- **WHEN** a client requests the catalogue
- **THEN** every new operation has input/output schemas and an available backend
  without changes to the CLI, MCP server, daemon, or pi extension.

### Requirement: Comparable and honest measurements
The toolchain SHALL reject incompatible measurement contexts and invalid numbers.

#### Scenario: Incompatible comparison
- **WHEN** tool versions, targets, constraints, stages or units differ
- **THEN** comparison fails instead of reporting a misleading improvement.

### Requirement: Correctness precedes measurement
The pipeline SHALL check explicitly combinational designs before spending a PPA run.

#### Scenario: Incorrect rewrite
- **WHEN** one binary input produces different or unknown outputs
- **THEN** the pipeline reports the failing input and does not launch synthesis.

### Requirement: Reproducibility and budget
Experiments SHALL preserve commands, source hashes, evidence scope and decisions.
The workflow SHALL stop when remaining Codex usage falls below 30 percent.

#### Scenario: Resume an experiment
- **WHEN** a completed experiment is reused
- **THEN** the runner verifies its input fingerprint and records reuse explicitly.
