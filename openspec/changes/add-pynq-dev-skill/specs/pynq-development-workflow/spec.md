## Purpose

Provide a repeatable, evidence-based development contract for PYNQ-Z1 and NPU
changes that uses OpenSpec planning and honors repository, hardware, and board
constraints before work is declared complete.

## ADDED Requirements

### Requirement: OpenSpec gates implementation
The workflow SHALL require a selected OpenSpec change with apply-ready planning
artifacts before product implementation begins, and SHALL update the change's
tasks as implementation progresses.

#### Scenario: New development request
- **WHEN** a user requests a PYNQ-Z1 or NPU implementation change
- **THEN** the workflow creates or selects an OpenSpec change, verifies that it
  is apply-ready, reads all apply context files, and only then edits product
  code

#### Scenario: Planning is incomplete
- **WHEN** the selected OpenSpec change is missing required artifacts
- **THEN** the workflow completes or updates those artifacts instead of editing
  product code

### Requirement: Repository rules constrain every change
The workflow SHALL read `AGENTS.md` and all applicable files under
`docs/rules/` before editing, SHALL treat them as authoritative, and SHALL stop
when the requested change conflicts with a hard repository rule.

#### Scenario: Requested path violates the file tree
- **WHEN** an implementation would add a prohibited directory or generated
  artifact to version control
- **THEN** the workflow rejects that placement and reports the governing rule

### Requirement: Development is routed by affected area
The workflow SHALL identify affected areas before implementation and SHALL load
the corresponding contracts for RTL, verification, export, runtime, Vivado,
deployment, and board interaction.

#### Scenario: Numeric or RTL behavior changes
- **WHEN** a change affects arithmetic, quantization, saturation, accumulator
  behavior, registers, or RTL interfaces
- **THEN** the workflow identifies the numeric and interface contracts and
  requires matching golden-model or testbench coverage

#### Scenario: Board interaction is not required
- **WHEN** a change cannot affect overlay loading, MMIO, physical I/O, timing,
  or board-only behavior
- **THEN** the workflow may mark board validation not applicable with a written
  rationale

### Requirement: Completion requires proportional evidence
The workflow SHALL define validation before implementation and SHALL not report
completion without recording command results and artifacts appropriate to the
affected areas.

#### Scenario: RTL change completes
- **WHEN** RTL or a testbench is modified
- **THEN** lint and simulation results are recorded, with synthesis or board
  evidence additionally required when the change can affect those stages

#### Scenario: Validation is unavailable
- **WHEN** a required tool, license, board, or network path is unavailable
- **THEN** the workflow records the exact blocker and reports partial completion
  without claiming the blocked gate passed

### Requirement: Generated artifacts and secrets remain controlled
The workflow SHALL keep generated Vivado projects, bitstreams, build products,
credentials, and private keys out of authored source and skill directories.

#### Scenario: Deployable output is produced
- **WHEN** synthesis or board deployment creates generated files
- **THEN** the workflow stores or stages them only in repository-approved
  locations and does not add them to version control unless the rules explicitly
  permit it

### Requirement: State and handoff are auditable
The workflow SHALL track steps in `STATE.md`, SHALL update OpenSpec task
checkboxes as work completes, and SHALL summarize changed files, validation,
blocked gates, and remaining risks at handoff.

#### Scenario: Work is interrupted
- **WHEN** development stops before every required gate completes
- **THEN** state and OpenSpec tasks distinguish completed, pending, and blocked
  work so a later run can resume without guessing
