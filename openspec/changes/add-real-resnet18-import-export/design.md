## Context

See `proposal.md` for motivation. Phase 2A deliberately stops at a validated
framework-neutral `QuantizedGraph`. Its arithmetic multiplies raw signed INT8
values, so the adapter must produce zero-centered activations; its residual
operator also requires both inputs and the output to have identical
quantization. The 2x2 accelerator and package format remain fixed.

## Goals / Non-Goals

**Goals:**

- Convert one pinned official TorchVision ResNet-18 checkpoint rather than
  claiming support for arbitrary PyTorch models.
- Keep framework dependencies on the host conversion side and keep exported
  runtime packages NumPy-only.
- Exercise actual 224x224x3 tensors and pretrained weights through the existing
  exporter/runtime contract.
- Preserve Phase 0 arithmetic, Phase 1 ABI/RTL, and Phase 2A package encoding.

**Non-Goals:**

- General PyTorch, TorchScript, FX, or ONNX import.
- Training, fine-tuning, or a claim that smoke/calibration inputs measure
  ImageNet accuracy.
- Making full ResNet-18 fast on a 2x2 matrix engine; Issue #7 owns physical
  performance and accuracy acceptance.
- Committing or redistributing the downloaded checkpoint or generated model.

## Decisions

### 1. Pin the official float state dictionary

Use TorchVision `ResNet18_Weights.IMAGENET1K_V1`, downloaded from the exact
`download.pytorch.org` URL recorded in tracked metadata. The downloader uses
the standard library, restricts redirects to the approved HTTPS host, streams
to a sibling temporary file, validates full length and SHA-256, and atomically
renames only on success.

The float state dictionary is preferred over TorchVision's FBGEMM checkpoint.
FBGEMM uses backend-specific quantized modules and commonly unsigned activation
domains or residual rescaling that do not directly satisfy the current signed,
identically-quantized residual contract. A float source lets this adapter apply
the repository contract explicitly.

### 2. Make conversion narrowly architecture-aware

A host-only adapter loads the state dictionary with safe weight-only loading,
checks the complete expected ResNet-18 key/shape set, and interprets the fixed
stem, four two-block stages, three projection shortcuts, global average pool,
and classifier. It does not execute serialized source code or accept arbitrary
module graphs.

Every convolution folds its adjacent inference BatchNorm. OIHW weights become
HWIO, and the classifier OI matrix becomes IO. Per-output-channel symmetric
weight quantization uses deterministic nearest rounding with ties away from
zero. Biases use the exact input-scale times weight-scale accumulator units.

### 3. Calibrate zero-centered activations with residual scale groups

Tracked code deterministically generates a small set of full-shape normalized
calibration tensors. A fixed float interpreter records finite absolute maxima.
The converter derives symmetric signed INT8 scales and zero point zero.

Identity blocks force their input, second convolution, residual output, and
post-add ReLU into one scale group. Projection blocks force the main second
convolution, projection output, add, and ReLU into the next stage group. Stem
ReLU, max pool, and stage-one residual paths share the first stage scale.
Output scales are increased when necessary so every convolution scale ratio is
representable by the existing Q1.31 multiplier and non-negative right shift.

Alternative considered: extend residual add with implicit branch rescaling.
Rejected because that changes the Phase 2A numeric contract and adds behavior
not implemented by the existing package/runtime.

### 4. Separate conversion evidence from accuracy evidence

Conversion writes the existing `.npu.json/.npu.bin` pair plus canonical source
and conversion provenance. A deterministic real-shape validation corpus and an
independent vectorized integer reference check the stem, first stage, and final
output. These results prove importer/runtime agreement only; they are labeled
real-model host evidence.

An ImageNet-derived labeled corpus and physical PYNQ-Z1 run remain Issue #7
acceptance inputs. Unit fixtures retain software-fixture labels, and only an
actual `NPURuntime` with matching BIT/HWH may emit physical-board evidence.

### 5. Keep generated model data inside an ignored example workspace

The user-facing location is `examples/resnet18/model/`, matching the existing
example name rather than creating a second `examples/resnet/` tree. Tracked
scripts, README, notebook, metadata, and tests live beside it; all model
workspace contents except a placeholder are ignored. The package builder takes
the workspace directory, discovers only canonical named outputs, and validates
provenance before creating an archive.

## Risks / Trade-offs

- [A conservative symmetric calibration can reduce pretrained accuracy] ->
  Report float/quantized prediction agreement separately and leave formal
  ImageNet thresholds to Issue #7 with an approved corpus.
- [The full NumPy/runtime execution is expensive] -> Validate the stem and
  first stage independently, use vectorized matrix fakes for host integration,
  and never reinterpret host timing as board performance.
- [A future upstream weight file can change] -> Pin the immutable URL, full
  digest, byte length, source revision, and conversion dependency versions.
- [Pickled checkpoints can execute code under unsafe loading] -> Require a
  pinned PyTorch version with weight-only loading, reject non-tensor records,
  and never use generic pickle loading.
- [Residual scale grouping can saturate activations] -> Record saturation
  counts and fail configurable conversion quality gates rather than weakening
  the graph contract silently.

## Migration Plan

1. Land the source metadata, ignored workspace, downloader, and offline tests.
2. Land deterministic converter tests using a source-shaped state fixture.
3. Download the pinned public checkpoint, convert it twice, and compare bytes.
4. Validate real-shape stem, first-stage, full graph, and package readiness.
5. Publish one PR to `dev`; Issue #7 consumes the generated workflow after
   merge and supplies formal corpus/Vivado/PYNQ evidence.

Rollback removes the adapter/example files and leaves the existing Phase 2A
graph, package, runtime, ABI, and RTL unchanged.
