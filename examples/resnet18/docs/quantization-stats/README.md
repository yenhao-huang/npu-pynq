# ResNet-18 INT8 quantization statistics

Everything needed to answer "what does INT8 quantization cost on this model":
the scripts, the measurements they produced, the visualization, and the
conclusions. Self-contained; nothing here is required by the demo workflow in
[`../../README.md`](../../README.md).

**Measured:** 2026-09-20 · **Related issues:** #49 (demo), #63 (benchmark), #68 (calibration defect)

---

## 1. Conclusions

**Weight quantization is essentially lossless.** Per-output-channel symmetric
INT8 on the BatchNorm-folded weights costs at most **1.6%** relative RMS error
on any single layer. Original and reconstructed densities overlap visually on
every layer. The scheme the repository already implements is the right one.

**Activation quantization currently costs about 18 percentage points of
ImageNet top-1.** The cause is not the quantization format, the numeric
contract, or the hardware ABI. It is that activation scales are estimated with
`max-abs` from only **two** calibration inputs.

**The fix does not touch hardware.** Recalibrating with 32 real images, changing
nothing else, closes the gap to within noise. Tracked in #68.

---

## 2. Results

### 2.1 Weights — nearly lossless

Per-layer, on the folded weights the exporter actually quantizes:

| layer | folded max\|W\| | max/std | ch max/min | rel err (per channel) | (per tensor if used) |
|---|---:|---:|---:|---:|---:|
| stem.conv | 0.3939 | 12.3 | 3.9e9 | 0.0092 | 0.0261 |
| layer1.0.conv1 | 0.3745 | 13.8 | 7.3 | 0.0156 | 0.0295 |
| layer1.0.conv2 | 0.7708 | 12.3 | 9.7 | 0.0114 | 0.0278 |
| layer1.1.conv1 | 0.2800 | 10.9 | 4.2 | 0.0132 | 0.0248 |
| layer1.1.conv2 | 1.0474 | 12.1 | 18.6 | 0.0115 | 0.0275 |
| layer2.0.conv1 | 0.2127 | 12.5 | 5.1 | 0.0120 | 0.0283 |
| layer2.0.conv2 | 0.7242 | 16.0 | 9.5 | 0.0146 | 0.0364 |
| layer2.0.projection | 0.6923 | 11.6 | **247.9** | 0.0094 | 0.0265 |
| layer2.1.conv1 | 0.3111 | 12.8 | 6.2 | 0.0146 | 0.0291 |
| layer2.1.conv2 | 0.8771 | 16.7 | 22.6 | 0.0120 | 0.0379 |
| layer3.0.conv1 | 0.2357 | 12.6 | 4.7 | 0.0134 | 0.0285 |
| layer3.0.conv2 | 0.5639 | 20.5 | 8.2 | **0.0160** | 0.0465 |
| layer3.0.projection | 0.4082 | 16.1 | 22.8 | 0.0093 | 0.0365 |
| layer3.1.conv1 | 0.2724 | 15.8 | 6.2 | 0.0139 | 0.0357 |
| layer3.1.conv2 | 0.9701 | 23.0 | 35.9 | 0.0132 | 0.0522 |
| layer4.0.conv1 | 0.3016 | 20.1 | 8.8 | 0.0127 | 0.0456 |
| layer4.0.conv2 | 1.1438 | 22.6 | 22.0 | 0.0148 | 0.0513 |
| layer4.0.projection | 0.9982 | 16.4 | 43.9 | 0.0104 | 0.0372 |
| layer4.1.conv1 | 0.2933 | 20.2 | 11.3 | 0.0126 | 0.0455 |
| layer4.1.conv2 | **3.6483** | 16.7 | 4.8 | 0.0142 | 0.0380 |

Two readings:

- **Per output channel is necessary, not decorative.** The last two columns are
  the same layer under per-channel and (hypothetical) per-tensor scaling.
  Per-tensor is 2-3x worse throughout, because output-channel magnitudes inside
  one layer span up to 248x. `layer2.0.projection` is a 1x1 pointwise
  projection, where each output channel is a pure linear combination of input
  channels and the spread is widest.
- **The `stem.conv` ratio of 3.9e9 is not a bug.** Eight of its 64 output
  channels are dead in the pretrained checkpoint itself (weights at 1e-8 to
  1e-6, matching `bn1.weight` of the same order). `quantize_conv_weight` still
  produces a positive scale for them, so nothing divides by zero.

**Visualization: [`weight_distribution.html`](weight_distribution.html)** —
open in any browser, works offline, no external assets. Two panels on one
canvas: a global histogram with log counts, and a per-layer ridgeline where
each row's x-axis is scaled to its own `+/-max|W|`. Blue fill is the original
density, orange line is the density after quantize-then-dequantize. Their
overlap on every layer is the visual form of the table above.

### 2.2 Activations — the 18 pp gap

1000 real images (one per class), same preprocessing, current on-disk
calibration:

| | FP32 | INT8 | delta |
|---|---:|---:|---:|
| top-1 | 79.7% | 61.6% | **-18.1 pp** |
| top-5 | 95.0% | 93.1% | -1.9 pp |
| top-1 agreement with FP32 | — | 71.5% | — |

Changing **only** the calibration inputs to 32 real images, disjoint from the
evaluation set — same exporter, same per-channel scheme, same numeric contract,
same ABI — on the same first 200 images:

| calibration | FP32 top-1 | INT8 top-1 | delta | agreement |
|---|---:|---:|---:|---:|
| 2 synthetic images (current) | 85.5% | 60.5% | -25.0 pp | 60.5% |
| 32 real images | 85.5% | **86.0%** | **+0.5 pp** | **96.0%** |

### 2.3 Root cause — `max-abs` from two samples is a lottery

Not synthetic content. Sample size. Same evaluation set (100 images, FP32 =
84.0%), varying only the calibration set:

| calibration | INT8 top-1 |
|---|---:|
| synthetic x2 (current) | 56% |
| real x2, pair (a) | 56% |
| real x2, pair (b) | 84% |
| real x2, pair (c) | 70% |
| real x4 | 85% |
| real x8 | 80% |
| real x32, three disjoint sets | 83% / 83% / 83% |
| real x128 | 85% |

**Two real photographs reproduce the same failure.** Three disjoint real pairs
span 56% to 84%; the synthetic pair simply sits at the bottom of that spread and
cannot be redrawn because it is hard-coded.

Why it is catastrophic rather than merely imprecise: the error is asymmetric.
Overestimating a tensor's range wastes resolution and degrades smoothly.
Underestimating it clips everything above the estimate to the same saturated
code. Under the current calibration `logits` gets `scale = 0.11192`, so the
representable range is `+/-14.214` while the same images need `32.24`. Only
**0.21%** of individual logit values exceed the threshold — but those are
concentrated on the winner: **82.8% of images have their single largest logit
clipped**. Top-1 is `argmax`, so flattening the winning score is exactly the
failure that destroys ranking. With 32 real images that figure is **0.0%**.

Why activations need data at all, when weights do not: a weight tensor is a
constant in the checkpoint, so `max|W|` is computable offline. An activation is
a feature map produced at run time, so its range can only be observed. A deep
ResNet layer has 512 class-selective channels and a single image strongly
excites only a few, so the layer-wide `max-abs` keeps growing with sample count
until enough channels have been hit. Two images sample a handful.

---

## 3. Files in this folder

| Path | What | Regenerable |
|---|---|---|
| `README.md` | This file. | — |
| `weight_distribution.html` | Interactive before/after weight distribution. **Tracked deliberately** — it is the primary artifact of this study and must survive without a PyTorch environment. | yes, by `scripts/build_weight_viz.py` |
| `scripts/download_int8_eval_corpus.py` | Fetches the 1000-image evaluation corpus. | — |
| `scripts/eval_int8_accuracy.py` | FP32 versus INT8 accuracy A/B. | — |
| `scripts/weight_distribution.py` | Per-layer weight statistics. | — |
| `scripts/build_weight_viz.py` | Builds `weight_distribution.html`. | — |
| `data/accuracy_synthetic_1000.json` | Section 2.2, top table. | yes |
| `data/accuracy_real32_200.json` | Section 2.2, bottom table, recalibrated side. | yes |
| `data/calibration_sensitivity.json` | Section 2.3 sweep. | yes |
| `data/weight_dist.json` | Section 2.1 table, machine-readable. | yes |
| `data/weight_dist.channels.csv` | Per-output-channel `max_abs` and `scale`, 4808 rows. | yes |

The 1000-image corpus itself is **not** committed. It lives under
`examples/resnet18/model/imagenet-calibration/` (107 MB, git-ignored per
[`docs/rules/generated-artifacts.md`](../../../../docs/rules/generated-artifacts.md))
and is fetched by the download script.

---

## 4. Reproduction

From the repository root, using the example's conversion environment (see
[`../../README.md`](../../README.md) step 1 for creating it):

```bash
# Prerequisite: the pinned checkpoint
python examples/resnet18/scripts/download_model.py

# Fetch the evaluation corpus (1000 real JPEGs, 107 MB, git-ignored)
python examples/resnet18/docs/quantization-stats/scripts/download_int8_eval_corpus.py

# Weight statistics and visualization
python examples/resnet18/docs/quantization-stats/scripts/weight_distribution.py \
    --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \
    --out-prefix examples/resnet18/docs/quantization-stats/data/weight_dist

python examples/resnet18/docs/quantization-stats/scripts/build_weight_viz.py \
    --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \
    --output    examples/resnet18/docs/quantization-stats/weight_distribution.html

# Accuracy with the current on-disk calibration
python examples/resnet18/docs/quantization-stats/scripts/eval_int8_accuracy.py \
    --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \
    --corpus     examples/resnet18/model/imagenet-calibration \
    --index      examples/resnet18/model/imagenet-calibration.index.tsv \
    --limit 1000 --calibration synthetic \
    --output examples/resnet18/docs/quantization-stats/data/accuracy_synthetic_1000.json

# Accuracy with 32 real calibration images
python examples/resnet18/docs/quantization-stats/scripts/eval_int8_accuracy.py \
    --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \
    --corpus     examples/resnet18/model/imagenet-calibration \
    --index      examples/resnet18/model/imagenet-calibration.index.tsv \
    --limit 200 --calibration real --calibration-size 32 \
    --output examples/resnet18/docs/quantization-stats/data/accuracy_real32_200.json
```

The 1000-image run takes about 66 minutes on a 2-core CPU. The bottleneck is
the vectorized integer reference, not the FP32 path.

---

## 5. Setup and caveats

| | |
|---|---|
| Model | TorchVision ResNet-18, `IMAGENET1K_V1` |
| Checkpoint | `resnet18-f37072fd.pth`, SHA-256 checked against `examples/resnet18/model-source.json` |
| Corpus | `EliSchwartz/imagenet-sample-images`, 1000 JPEGs, one per class |
| Preprocessing | Resize shorter side to 256 (bilinear), center crop 224, `/255`, ImageNet mean/std |
| FP32 path | `torchvision.models.resnet18`, eval mode |
| INT8 path | `src/export/torchvision_resnet18.py` for the graph, `src/test/model/quantized_graph_reference.py` for execution |
| Quantization | Per-output-channel symmetric INT8 weights, per-tensor symmetric INT8 activations, Q1.31 requantization, INT32 accumulator |

**The corpus is not the ImageNet-1K validation set.** It is a curated
one-per-class sample. FP32 measures 79.7 top-1 here against the published
69.758 on validation, because the sample is easier. **Absolute figures are not
comparable to published numbers; only same-image FP32-versus-INT8 deltas are.**
It was used because `image-net.org` and the Hugging Face mirror are outside this
environment's network allowlist while `raw.githubusercontent.com` is inside it.

**Sampling noise.** At n=1000 a single accuracy figure carries roughly +/-2.5 pp
at 95% confidence; at n=100, roughly +/-7 pp. The -18.1 pp gap and the
recalibration recovery are far outside those bands. The differences among the
4/8/32/128-image calibrations in section 2.3 are **not** — they are all within
noise of each other.

**Simulation only.** Every number here was produced on a host CPU through the
integer reference. None of it is physical PYNQ-Z1 evidence.

**`src/model/operators.py::conv2d_int8` cannot be used for dataset evaluation.**
It is a per-MAC Python loop, correct but far too slow for 1000 images. The
vectorized `quantized_graph_reference.py` is used instead, and
`test_vectorized_reference_matches_approved_scalar_operators` pins the two
bit-exact.
