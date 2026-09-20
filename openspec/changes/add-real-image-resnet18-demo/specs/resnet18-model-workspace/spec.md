## Purpose

Extends the tracked example runbook so real-image preparation has a defined
place between real-model validation and Vivado artifact selection.

## MODIFIED Requirements

### Requirement: Copyable ordered runbook

The example README SHALL provide copyable commands in this order: prepare the
host environment, download the pinned checkpoint, convert and validate the real
model, prepare the real demo image, build or select matching Vivado artifacts,
create the package, and run host or board acceptance. Each step SHALL state its
expected output and stop condition.

#### Scenario: Human follows a clean-checkout runbook
- **WHEN** all documented prerequisites and external hardware are available
- **THEN** the commands operate only on documented paths and reach an unambiguous evidence result without hidden model preparation

#### Scenario: Human wants a real prediction
- **WHEN** the runbook is followed through the demo preparation step
- **THEN** the deployed notebook can display the image and its predicted ImageNet class without further preparation on the board
