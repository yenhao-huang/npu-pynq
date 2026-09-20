## Why

The ResNet-18 demo runs only `resnet18.validation.npy`, a deterministic
synthetic RGB gradient. That tensor proves host and board agree bit for bit,
but it shows nothing a person can look at and produces no ImageNet label, so
the notebook cannot demonstrate that the imported pretrained network makes a
meaningful prediction. Issue #49 closes that gap without weakening the
deterministic regression path.

## What Changes

- Pin one redistributable sample photograph and the ImageNet-1K class list by
  URL, byte length, SHA-256, and license, and download them into the ignored
  model workspace. No ImageNet dataset image is fetched or committed.
- Add the TorchVision preprocessing contract as shared production code: RGB
  conversion, shorter-side resize to 256, center crop to 224, ImageNet
  normalization, and symmetric signed-INT8 quantization at the manifest's own
  input scale.
- Preprocess the chosen image once on the conversion host, check it through the
  independent integer reference and `NPUModelRuntime`, and publish the exact
  tensor plus a record of image provenance, preprocessing parameters, expected
  class, host capture digests, and host top-5.
- Display the original image and the dequantized preview of the exact tensor in
  the notebook before any inference, then execute it on the PYNQ-Z1, compare
  every capture digest with the host record, and decode the top-5 ImageNet
  labels with a correct/incorrect verdict.
- Keep the synthetic validation tensor, its notebook steps, and its acceptance
  evidence exactly as they are.

## Capabilities

### New Capabilities

- `resnet18-real-image-demo`: Pinned demo asset acquisition, the reproducible
  preprocessing and quantization contract, the recorded demo input boundary,
  and the notebook's visual and label-decoding requirements.

### Modified Capabilities

- `resnet18-model-workspace`: The copyable runbook gains a real-image
  preparation step between real-model validation and Vivado artifact
  selection.

## Impact

- Affected paths: `src/export/imagenet.py`, `src/test/tests/`,
  `examples/resnet18/`, `docs/rules/filetree.md`, and this OpenSpec change.
- Host preparation adds Pillow; the exported model, the PYNQ runtime, and the
  serialized package format are unchanged.
- Downloaded images, class lists, prepared tensors, and demo records stay
  untracked under `examples/resnet18/model/`.
- No RTL, register map, ABI, numeric contract, or board-network change.
