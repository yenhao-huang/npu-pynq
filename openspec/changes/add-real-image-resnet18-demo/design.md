## Context

The converted ResNet-18 already exposes `logits` as one of its three captured
outputs, and the exported manifest already records every activation scale as a
Q1.31 identity. A real-image demo therefore needs no change to the graph, the
package format, or the runtime: it needs an input the model was trained to see
and a way to read its output as a class name.

The board runs Pillow 9 and NumPy 1.21; the conversion host runs much newer
versions. Image decoding and resampling are not guaranteed to be bit-identical
across those versions, so preprocessing on the board would make host and board
disagree for reasons that have nothing to do with the NPU.

## Goals / Non-Goals

**Goals:**

- Show a human-viewable image and the exact tensor the NPU receives, before
  inference runs.
- Produce an ImageNet label from a physical board run, bound to a prior host
  reference by digest.
- Let a person classify their own image without editing runtime code.
- Keep the deterministic synthetic path intact as the numerical regression.

**Non-goals:**

- Claim ImageNet accuracy. One labelled photograph is a demonstration.
- Redistribute ImageNet dataset images.
- Change the numeric contract, ABI, RTL, or serialized package format.
- Move preprocessing into the runtime or onto the board.

## Decisions

**Preprocess on the host, ship the tensor.** `prepare_demo_image.py` writes
`resnet18.demo.npy` and the board reloads it after a digest check. The board's
image library then only decodes the photograph for display, where a one-pixel
difference is irrelevant. The alternative, preprocessing on the board, would
make the host/board capture comparison depend on Pillow's resampling rather
than on the accelerator.

**Recover the input scale from the manifest.** `input_scale()` reads the
`input` tensor's `multiplier_q31` and `shift` instead of taking a scale from a
side channel, so a re-converted model cannot be paired with a stale tensor
scale. The same function reads the `logits` scale for decoding.

**Record the host top-5, compare only the ordering.** Capture digests must
match exactly, which already pins the INT8 logits. Softmax probabilities are
floating point and may differ in the last ulp between x86 and ARM, so the
notebook asserts that the class ordering matches and prints both.

**Pin the sample assets like the checkpoint.** `demo-source.json` follows
`model-source.json`: approved host, pinned commit, byte length, SHA-256, and
SPDX license, fail-closed, never committed. The sample is a BSD-3-Clause
photograph published with the PyTorch examples, not an ImageNet image.

**Write-once outputs.** Preparation refuses to overwrite an existing
`resnet18.demo.*`, matching every other generated step in this example, so a
second image cannot silently invalidate recorded evidence.

## Risks / Trade-offs

- A single image proves nothing statistical. The README, the scripts, and the
  notebook all say so; the marker stays `real-model-host` on the host side and
  the physical marker is unchanged.
- Pillow becomes a host preparation dependency. It is not needed by the
  runtime, and `src/export/imagenet.py` imports it lazily so the module and its
  tests load without it.
- An upstream repository could delete the pinned blob. The commit-pinned URL
  and digest make that a visible failure rather than a silent substitution, and
  any other image can be supplied with `--image`.
