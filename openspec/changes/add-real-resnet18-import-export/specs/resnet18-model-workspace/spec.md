## Purpose

Defines the local example workspace and readiness boundary used to turn a real
converted ResNet-18 into a reproducible standalone package without tracking it.

## ADDED Requirements

### Requirement: Gitignored example model workspace

The ResNet-18 example SHALL use `examples/resnet18/model/` for downloaded
checkpoints, calibration inputs, converted NPU files, descriptors, and local
evidence. Generated contents SHALL be ignored by Git while the directory
contract and reproducible commands remain tracked.

#### Scenario: Clean repository after model preparation
- **WHEN** download, conversion, and host validation complete successfully
- **THEN** Git reports no generated model, corpus, package, or evidence file as trackable content

### Requirement: Package readiness boundary

The example package command SHALL consume the model workspace, verify its
source and conversion provenance, require the complete NPU manifest/payload and
validation descriptor assets, and fail before archive publication when any
required input is absent, stale, substituted, or incomplete.

#### Scenario: Only the public checkpoint was downloaded
- **WHEN** package creation is requested before conversion and validation assets exist
- **THEN** the command reports that conversion is required and creates no archive

### Requirement: Copyable ordered runbook

The example README SHALL provide copyable commands in this order: prepare the
host environment, download the pinned checkpoint, convert and validate the real
model, build or select matching Vivado artifacts, create the package, and run
host or board acceptance. Each step SHALL state its expected output and stop
condition.

#### Scenario: Human follows a clean-checkout runbook
- **WHEN** all documented prerequisites and external hardware are available
- **THEN** the commands operate only on documented paths and reach an unambiguous evidence result without hidden model preparation

### Requirement: Honest evidence levels

The example SHALL distinguish reduced-fixture software tests, real-model host
validation, and physical PYNQ-Z1 acceptance in pass markers and evidence types.
No fake runtime, placeholder overlay, or generated fixture SHALL emit a marker
that claims physical-board acceptance.

#### Scenario: Fake matrix runtime succeeds
- **WHEN** the converted graph executes successfully through a fake matrix runtime
- **THEN** its result is labeled real-model host evidence and not physical-board evidence
