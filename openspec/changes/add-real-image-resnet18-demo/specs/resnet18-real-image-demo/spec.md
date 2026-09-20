## Purpose

Defines how one real photograph becomes a reproducible ResNet-18 NPU input and
how the notebook shows the image and its predicted ImageNet class without
claiming dataset accuracy or redistributing dataset images.

## ADDED Requirements

### Requirement: Pinned redistributable demo assets

The demo SHALL identify every downloaded demo asset by kind, filename,
immutable commit-pinned HTTPS URL on an approved host, expected byte length,
complete SHA-256 digest, and SPDX license. Download SHALL fail closed on
missing or non-canonical metadata, an unapproved host, an unpinned revision, a
size or digest mismatch, a redirect outside the pinned path, or an existing
destination. Downloaded bytes SHALL remain in the ignored model workspace.

#### Scenario: Pinned asset digest differs
- **WHEN** a downloaded demo asset does not match its pinned byte length and SHA-256
- **THEN** no asset file is published and no demo input is prepared

#### Scenario: Metadata names an unapproved source
- **WHEN** demo metadata points outside the approved host or omits the pinned revision
- **THEN** loading the metadata fails before any network request

#### Scenario: Repository redistribution
- **WHEN** the repository is checked out clean
- **THEN** no sample image or class list is present as tracked content

### Requirement: Reproducible preprocessing contract

Preprocessing SHALL convert the image to RGB, resize its shorter side to 256,
center crop 224, normalize with the ImageNet channel mean and standard
deviation, and quantize symmetrically to signed INT8 with zero point 0, using
the input scale recovered from the exported manifest's own Q1.31 quantization
record. Rounding SHALL be half away from zero and values SHALL saturate to the
inclusive range -127 to 127. The parameters used SHALL be recorded with the
prepared input.

#### Scenario: Repeated preparation of one image
- **WHEN** the same image and the same exported manifest are preprocessed twice
- **THEN** the published signed-INT8 tensor is byte-identical

#### Scenario: Manifest declares a different input scale
- **WHEN** the exported model is re-converted with a different input quantization
- **THEN** preparation derives the new scale from the manifest rather than a recorded constant

### Requirement: Host-referenced demo input record

Preparation SHALL execute the prepared tensor through both the independent
integer reference and the production host model runtime, SHALL fail before
publishing when any capture differs, and SHALL publish a canonical write-once
record holding the image provenance and digest, the preprocessing parameters,
the input digest and scale, the logit scale, the class-list digest, the
declared expected class when one is given, the host capture digests, and the
host top-5. The record SHALL be labeled host evidence and SHALL NOT claim a
physical board result.

#### Scenario: Reference and production host paths disagree
- **WHEN** any capture differs between the two host paths
- **THEN** no demo tensor or demo record is published

#### Scenario: Preparation is repeated
- **WHEN** a demo tensor or record already exists
- **THEN** preparation refuses rather than overwriting recorded evidence

### Requirement: Image selection without runtime changes

A person SHALL be able to select their own image and declare its expected
ImageNet class by index or exact class name through preparation arguments, with
no edit to `src/runtime/`, `src/export/`, or the notebook. An image with no
declared expected class SHALL still be prepared, and the demo SHALL then report
the top-5 without a verdict.

#### Scenario: User supplies a photograph
- **WHEN** preparation is invoked with an image path and an expected class name
- **THEN** the demo record names that image and that class, and no production runtime source mentions the sample image

#### Scenario: Expected class is omitted
- **WHEN** no expected class is declared
- **THEN** the demo reports the top-5 and states that the person must judge the result

### Requirement: Visible input before inference

The notebook SHALL validate the demo record and every referenced asset by
digest, and SHALL display both the original photograph and the exact prepared
tensor dequantized back to pixels, before executing the real image on the NPU.

#### Scenario: Substituted demo asset
- **WHEN** the image, tensor, or class list digest differs from the record
- **THEN** the notebook fails at the loading step and no inference runs

#### Scenario: Human inspects the input
- **WHEN** the display step runs
- **THEN** the photograph and the dequantized NPU input are shown side by side with the declared expected class

### Requirement: Board-verified label prediction

The notebook SHALL execute the real image on the physical runtime, SHALL
compare every output capture digest with the host record before interpreting
any result, and SHALL then display the top-5 ImageNet class names and scores,
the predicted class, and whether it matches the declared expected class. The
real-image result SHALL be recorded in the assembled evidence alongside the
synthetic result.

#### Scenario: Board capture differs from host
- **WHEN** any board capture digest differs from the host record
- **THEN** the notebook fails before decoding a label

#### Scenario: Prediction is decoded
- **WHEN** every capture digest matches
- **THEN** the notebook prints the top-5 labels, the predicted class, and a correct or incorrect verdict

### Requirement: Preserved deterministic regression path

The synthetic validation tensor, its notebook execution and comparison steps,
its host acceptance descriptor, and the CLI board entry point SHALL remain
available and unchanged in behavior. The real-image demo SHALL be an additional
execution, not a replacement.

#### Scenario: Deterministic acceptance after the change
- **WHEN** the notebook runs on the board
- **THEN** the synthetic tensor is executed and compared against `acceptance.json` before the real image is loaded
