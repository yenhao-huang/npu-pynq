## Purpose

Defines content-addressed external assets and canonical topology required for
honest ResNet-18 acceptance without committing weights or datasets.

## ADDED Requirements

### Requirement: Content-addressed acceptance bundle

The acceptance loader SHALL require a canonical descriptor binding the exact
Phase 2A manifest, payload, corpus, reference identity, preprocessing identity,
class count, sample count, thresholds, byte lengths, and SHA-256 digests. Every
asset path SHALL be a relative basename confined to the bundle directory.

#### Scenario: Substituted corpus
- **WHEN** one corpus byte differs from its descriptor digest
- **THEN** bundle validation fails before model loading, allocation, or runtime calls

### Requirement: Safe corpus structure

The corpus SHALL contain dense signed INT8 inputs and expected outputs, integer
labels, and unique stable sample identifiers with aligned leading dimensions.
Loading SHALL disable pickle and reject object arrays, unsupported dtypes,
invalid shapes, duplicate identifiers, and labels outside the declared class
range.

#### Scenario: Object array corpus
- **WHEN** an NPZ member requires pickle or has object dtype
- **THEN** validation rejects it without deserializing executable objects

### Requirement: Canonical ResNet-18 topology

The package SHALL contain the canonical stem, four two-block stages, eight
residual additions, three projection shortcuts, global average pool, flatten,
and classifier with valid branch and stage-shape transitions. Validation SHALL
derive topology from command dependencies and tensor shapes rather than names.

#### Scenario: Missing projection shortcut
- **WHEN** a downsampling block reuses an incompatible identity tensor or omits its projection
- **THEN** topology validation fails before acceptance execution

### Requirement: External asset policy

Trained weights, datasets, generated model packages, overlays, credentials,
and acceptance evidence SHALL remain untracked. Test fixtures SHALL be
deterministically generated from repository source and SHALL NOT be described
as model-accuracy evidence.

#### Scenario: Synthetic fixture
- **WHEN** the reduced deterministic fixture passes host acceptance
- **THEN** the result is labeled software integration evidence, not ImageNet or board acceptance
