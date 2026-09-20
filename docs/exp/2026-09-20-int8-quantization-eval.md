# ResNet-18 INT8 quantization evaluation

**Date:** 2026-09-20
**Branch:** `npu/issue49-b`
**Related:** #49 (real-image demo, merged), #63 (benchmark harness), #68 (activation calibration defect)

## Summary

- **Weight** INT8 quantization on ResNet-18 is essentially lossless: per-output-channel relative RMS error is at most 1.6% on any single layer, and reconstructed densities overlap the originals visually.
- **Activation** INT8 quantization with the currently shipped calibration (2 synthetic images) costs about 18 pp of ImageNet top-1 accuracy on real photographs.
- The defect is **not** the quantization format or the numeric contract. It is the `max-abs` estimator with sample size n = 2. Two real photographs reproduce the same failure; the synthetic pair is one fixed bad draw from that estimator.
- Re-calibrating with 32 real images, changing nothing else, recovers the accuracy gap to within noise. No change to hardware ABI, register map, or serialized package format is required.

## Setup

| | |
|---|---|
| Model | TorchVision ResNet-18, `IMAGENET1K_V1` |
| Checkpoint | `resnet18-f37072fd.pth`, SHA-256 checked against `examples/resnet18/model-source.json` |
| Corpus | `EliSchwartz/imagenet-sample-images`, 1000 JPEGs (one per class) |
| Preprocessing | Resize shorter side to 256 (bilinear), center crop 224x224, `/255`, normalize with ImageNet mean/std |
| FP32 execution | `torchvision.models.resnet18` in eval mode |
| INT8 execution | `src.export.torchvision_resnet18.build_quantized_resnet18` for the graph; `src.test.model.quantized_graph_reference.execute_quantized_graph_reference` for the run |
| Quantization | Per-output-channel symmetric INT8 weights; per-tensor symmetric INT8 activations; Q1.31 requantization; INT32 accumulator |

**Important caveat.** The corpus is a curated one-per-class sample, not the ImageNet-1K validation set. FP32 measures 79.7 top-1 against the published 69.758 because the sample is easier than validation. **Absolute figures are not comparable to published numbers; only same-image FP32-vs-INT8 deltas are.**

## Weight quantization is essentially lossless

Per-layer summary from `weight_distribution.py` on the folded (BN-absorbed) weights that the exporter actually quantizes:

| layer | folded max\|W\| | max/std | ch max/min | rel error (per ch) | (per tensor if used) |
|---|---:|---:|---:|---:|---:|
| stem.conv | 0.3939 | 12.3 | 3.9e9 | 0.0092 | 0.0261 |
| layer1.0.conv1 | 0.3745 | 13.8 | 7.3 | 0.0156 | 0.0295 |
| layer1.0.conv2 | 0.7708 | 12.3 | 9.7 | 0.0114 | 0.0278 |
| layer1.1.conv1 | 0.2800 | 10.9 | 4.2 | 0.0132 | 0.0248 |
| layer1.1.conv2 | 1.0474 | 12.1 | 18.6 | 0.0115 | 0.0275 |
| layer2.0.conv1 | 0.2127 | 12.5 | 5.1 | 0.0120 | 0.0283 |
| layer2.0.conv2 | 0.7242 | 16.0 | 9.5 | 0.0146 | 0.0364 |
| layer2.0.projection | 0.6923 | 11.6 | 247.9 | 0.0094 | 0.0265 |
| layer2.1.conv1 | 0.3111 | 12.8 | 6.2 | 0.0146 | 0.0291 |
| layer2.1.conv2 | 0.8771 | 16.7 | 22.6 | 0.0120 | 0.0379 |
| layer3.0.conv1 | 0.2357 | 12.6 | 4.7 | 0.0134 | 0.0285 |
| layer3.0.conv2 | 0.5639 | 20.5 | 8.2 | 0.0160 | 0.0465 |
| layer3.0.projection | 0.4082 | 16.1 | 22.8 | 0.0093 | 0.0365 |
| layer3.1.conv1 | 0.2724 | 15.8 | 6.2 | 0.0139 | 0.0357 |
| layer3.1.conv2 | 0.9701 | 23.0 | 35.9 | 0.0132 | 0.0522 |
| layer4.0.conv1 | 0.3016 | 20.1 | 8.8 | 0.0127 | 0.0456 |
| layer4.0.conv2 | 1.1438 | 22.6 | 22.0 | 0.0148 | 0.0513 |
| layer4.0.projection | 0.9982 | 16.4 | 43.9 | 0.0104 | 0.0372 |
| layer4.1.conv1 | 0.2933 | 20.2 | 11.3 | 0.0126 | 0.0455 |
| layer4.1.conv2 | 3.6483 | 16.7 | 4.8 | 0.0142 | 0.0380 |

Two things to note:

- **Per-channel is necessary, not optional.** The last two columns compare relative error under per-output-channel and (hypothetical) per-tensor quantization. Per-tensor is 2-3x worse on most layers because a single layer's output channels vary by up to 248x (`layer2.0.projection`, a 1x1 pointwise projection).
- **The current per-channel scheme is well matched to the weight distribution.** Every layer's relative RMS error is under 1.6%, and the overall distribution shape is preserved (see visualization).

**Visualization.** Regenerate with:

```
python examples/resnet18/scripts/build_weight_viz.py \
    --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \
    --output    examples/resnet18/model/weight_distribution.html
```

The HTML page carries two panels on one canvas: a global histogram (log count) with original and dequantized overlaid, and a per-layer ridgeline with each layer's x-axis scaled to its own `+/-max|W|`. Blue fill = original density; orange line = dequantized density. Their near-perfect overlap is the visual evidence that weight quantization is nearly lossless.

## Activation quantization: the 18 pp gap

Same 1000 real images, same preprocessing, current on-disk calibration (two synthetic images):

| | FP32 | INT8 | delta |
|---|---:|---:|---:|
| top-1 | 79.7% | 61.6% | **-18.1 pp** |
| top-5 | 95.0% | 93.1% | -1.9 pp |
| top-1 agreement with FP32 | — | 71.5% | — |

Only changing the calibration inputs to 32 real images, disjoint from the evaluation set — everything else identical (exporter, per-channel scheme, numeric contract, hardware ABI) — on the same first 200 images:

| calibration | FP32 top-1 | INT8 top-1 | delta | agreement |
|---|---:|---:|---:|---:|
| 2 synthetic images (current) | 85.5% | 60.5% | -25.0 pp | 60.5% |
| 32 real images | 85.5% | 86.0% | **+0.5 pp** | 96.0% |

The gap is not the quantization format.

## Root cause: `max-abs` with n = 2 is a lottery

Not synthetic content per se — it is sample size. Same evaluation set (100 images, FP32 = 84.0%), varying only the calibration set:

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

**Two real images reproduce the same failure.** The three disjoint pairs of real photographs span 56% to 84%: that spread is the estimator's variance at n = 2, and the synthetic pair sits at its bottom.

### Why n = 2 fails asymmetrically

Overestimating a tensor's range wastes resolution but degrades gracefully. Underestimating it clips every value above the estimate to +/-127, and the information is gone. The failure is not distributed evenly across the network — it concentrates on the final logits.

Under the current calibration, `logits` gets `scale = 0.11192`, so the representable range is +/-14.214 while the same real images need 32.24. Global logit-value clipping is only 0.21%, but **82.8% of images have their single top logit clipped**. Because top-1 is `argmax`, clipping the winning score is exactly the failure that destroys ranking. Under 32 real images, that figure drops to 0.0%.

### Why activation is different from weights

Weight scale is computed from the weight tensor itself — the file — with no data required. Activation scale requires observing intermediate feature maps, and those depend on the input. A deep ResNet layer has 512 channels, each a class-selective feature detector; a single image strongly excites only a small subset of channels, so the layer-wide `max-abs` over n images grows with n until enough channels have been sampled. n = 2 samples only a handful.

## Recovery

Two zero-format changes recover the gap without touching hardware or the ABI:

1. Increase `n` in `generate_calibration_inputs()` to a few dozen. **32 real images is empirically sufficient**; more does not help visibly.
2. Optionally replace `max-abs` with a **99.9 percentile** or **KL-divergence** estimator (the standard TensorRT approach). Percentile especially reduces sensitivity to outlier draws.

Issue #68 tracks the fix.

## Reproduction

From the repository root, with `build/resnet18-venv/Scripts/python.exe` (Windows) or the equivalent Python. The corpus and result files live under `examples/resnet18/model/`, which is git-ignored by design.

```
# One-time: fetch the pretrained checkpoint (existing script)
python examples/resnet18/scripts/download_model.py

# Fetch the evaluation corpus (1000 real JPEGs, 107 MB)
python examples/resnet18/scripts/download_int8_eval_corpus.py

# Weight distribution: per-layer statistics and visualization
python examples/resnet18/scripts/weight_distribution.py \
    --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \
    --out-prefix examples/resnet18/model/int8-eval/weight_dist

python examples/resnet18/scripts/build_weight_viz.py \
    --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \
    --output    examples/resnet18/model/int8-eval/weight_distribution.html

# Accuracy A/B on 1000 images with the current on-disk calibration
python examples/resnet18/scripts/eval_int8_accuracy.py \
    --checkpoint  examples/resnet18/model/resnet18-f37072fd.pth \
    --corpus      examples/resnet18/model/imagenet-calibration \
    --index       examples/resnet18/model/imagenet-calibration.index.tsv \
    --limit       1000 --calibration synthetic \
    --output      examples/resnet18/model/int8-eval/result_synthetic.json

# Same evaluation with 32 real calibration images
python examples/resnet18/scripts/eval_int8_accuracy.py \
    --checkpoint  examples/resnet18/model/resnet18-f37072fd.pth \
    --corpus      examples/resnet18/model/imagenet-calibration \
    --index       examples/resnet18/model/imagenet-calibration.index.tsv \
    --limit       1000 --calibration real --calibration-size 32 \
    --output      examples/resnet18/model/int8-eval/result_real32.json
```

## Files

| Path | Role | Tracked? |
|---|---|---|
| `docs/exp/2026-09-20-int8-quantization-eval.md` | This document. | yes |
| `examples/resnet18/scripts/eval_int8_accuracy.py` | Accuracy A/B script. | yes |
| `examples/resnet18/scripts/weight_distribution.py` | Per-layer weight statistics dumper. | yes |
| `examples/resnet18/scripts/build_weight_viz.py` | HTML visualization builder. | yes |
| `examples/resnet18/scripts/download_int8_eval_corpus.py` | Corpus downloader. | yes |
| `examples/resnet18/model/imagenet-calibration/` | 1000 JPEG images. | no, per `docs/rules/generated-artifacts.md` |
| `examples/resnet18/model/int8-eval/` | Result JSONs, HTML visualization, weight-distribution JSON/CSV. | no, same rule |
