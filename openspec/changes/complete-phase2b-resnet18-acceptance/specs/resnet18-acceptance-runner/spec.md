## Purpose

Defines deterministic accuracy, repeatability, performance, and evidence
behavior shared by host integration and PYNQ-Z1 acceptance.

## ADDED Requirements

### Requirement: Stable exact acceptance

The runner SHALL process samples in stable corpus order, compare requested
layer tensors and final signed INT8 outputs exactly, compute top-1 predictions
from classifier outputs, and enforce the descriptor's accuracy and exact-match
thresholds.

#### Scenario: One layer mismatch
- **WHEN** a captured intermediate tensor differs even though the final class matches
- **THEN** the acceptance run fails and identifies the sample and tensor without publishing success evidence

### Requirement: Repeatable inference

The runner SHALL repeat the configured samples with changed inputs and require
identical per-sample outputs, predictions, work accounting, and successful
recovery after an injected or observed failed invocation.

#### Scenario: Stale activation
- **WHEN** a repeated sample output depends on a prior sample's arena contents
- **THEN** repeatability acceptance fails

### Requirement: Performance accounting

The runner SHALL report sample count, latency distribution, throughput,
logical input/output bytes, physical jobs, MACs, operations, and physical cycle
sum when exposed. Unavailable cycle metadata SHALL be explicit and SHALL fail a
board run whose descriptor requires cycles.

#### Scenario: Missing board cycles
- **WHEN** a board acceptance descriptor requires cycles and any physical job lacks them
- **THEN** performance acceptance fails rather than estimating cycles

### Requirement: Transactional evidence

Evidence SHALL be canonical, provenance-bound JSON and SHALL replace prior
known-good evidence only after every configured gate succeeds. Failures SHALL
leave the previous evidence byte-identical and return a non-success status.

#### Scenario: Accuracy below threshold
- **WHEN** exact execution completes but top-1 accuracy is below the declared threshold
- **THEN** no new final evidence is published
