# ResNet-18 8 x 8 NPU demo

The default demo uses an 8 x 8 systolic array (64 processing elements), with
`MAX_K=256`. The notebook rejects other overlay dimensions.

This example downloads one pinned official TorchVision ResNet-18, converts it
to the repository's Phase 2A signed-INT8 format, checks a real
`(1, 224, 224, 3)` input through an independent integer reference and the
production model runtime, and creates a deterministic model archive.

The calibration and regression tensor is synthetic and unlabeled, and it stays
in place for deterministic numerical acceptance. Step 5 additionally prepares a
real photograph so the notebook can display the image, run it on the board, and
print the predicted ImageNet label. A host PASS proves importer and runtime
agreement; a single labelled photograph is a demonstration, not ImageNet
accuracy evidence, and no host run is physical-board evidence.

Steps 1 through 8 prepare and copy the release from the Windows development
host. Step 9 is the human demo and runs from the deployed `.ipynb` on the
PYNQ-Z1, where the `pynq` package, Overlay, MMIO, and DMA are available.

```
python -m venv build/resnet18-venv
.\build\resnet18-venv\Scripts\activate
```

## 1. Prepare the conversion environment

From the repository root in PowerShell:

```powershell
& build/resnet18-venv/Scripts/python.exe -m pip install `
  --extra-index-url https://download.pytorch.org/whl/cpu `
  -r examples/resnet18/requirements-convert.txt
```

Stop if dependency installation fails. PyTorch is used only on the conversion
host; the exported runtime remains NumPy-only.

## 2. Download the pinned checkpoint

```powershell
& build/resnet18-venv/Scripts/python.exe `
  examples/resnet18/scripts/download_model.py
```

Expected marker: `PASS: downloaded and verified ...resnet18-f37072fd.pth`.
The script rejects redirects to another host, incorrect length or SHA-256, and
an existing destination. Generated files stay under
`examples/resnet18/model/` and are ignored by Git.

## 3. Convert the model

```powershell
& build/resnet18-venv/Scripts/python.exe `
  examples/resnet18/scripts/convert_model.py
```

Expected markers report `resnet18.npu.json` and
`resnet18.conversion.json`. Stop on any source-schema, non-finite value,
quantization, accumulator certificate, or exporter error.

## 4. Validate the real model

```powershell
& examples/resnet18/scripts/verify.ps1 `
  -Python build/resnet18-venv/Scripts/python.exe
```

Expected marker: `PASS [real-model-host]`. The validator reloads the exported
package and runs all 1,814,073,344 MACs through `NPUModelRuntime`. Its
`stem.relu`, `layer1.1.relu`, and `logits` outputs must match the independent
vectorized integer reference exactly. It writes `model/acceptance.json` only
after every comparison passes.

## 5. Prepare the real ImageNet demo image

Download the pinned class list and sample photograph into the ignored model
workspace:

```powershell
& build/resnet18-venv/Scripts/python.exe `
  examples/resnet18/scripts/download_demo_assets.py
```

Both assets come from the BSD-3-Clause `pytorch/hub` repository at the commit
pinned in `examples/resnet18/demo-source.json`, are verified by length and
SHA-256, and are never committed here. No ImageNet dataset image is downloaded
or redistributed.

Then preprocess one image into the exact tensor the NPU will consume:

```powershell
& build/resnet18-venv/Scripts/python.exe `
  examples/resnet18/scripts/prepare_demo_image.py
```

Expected markers: `PASS [real-model-host]`, the host top-1 class, and a
`CORRECT` or `INCORRECT` verdict against the expected class. The script applies
the TorchVision contract (RGB, shorter side to 256, center crop 224, ImageNet
normalization, symmetric signed-INT8 quantization), runs the result through the
independent integer reference and `NPUModelRuntime`, and writes
`model/resnet18.demo.npy` plus `model/resnet18.demo.json`. That record pins the
image provenance and digest, the preprocessing parameters, the input scale, the
expected class, the host capture digests, and the host top-5.

To classify your own picture instead, pass it explicitly; nothing in the
runtime or the notebook has to change:

```powershell
& build/resnet18-venv/Scripts/python.exe `
  examples/resnet18/scripts/prepare_demo_image.py `
  --image C:/path/to/photo.jpg `
  --expected-class "golden retriever"
```

`--expected-class` accepts an ImageNet class index or an exact class name from
`model/imagenet-classes.txt`. Omit it for an unlabelled image; the notebook then
shows the top-5 and asks you to judge the result. You are responsible for the
rights to any image you supply. Preparation outputs are write-once: remove the
previous `model/resnet18.demo.*` files to prepare a different image.

## 6. Build or select trusted Vivado artifacts

This step requires a licensed Vivado host and cannot be replaced by CI fixture
artifacts:

```powershell
vivado -mode batch -nojournal -nolog `
  -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl
python -m src.runtime.verify_overlay `
  build/vivado/npu_matrix_8x8/artifacts
```

The notebook and deployment wrapper use `build/vivado/npu_matrix_8x8/artifacts`.
The wrapper validates model assets and the 8 x 8 overlay before any transfer.

Stop unless the verification marker says the BIT/HWH provenance and metadata
passed and the artifact manifest identifies the intended source commit.

## 7. Build the model archive

```powershell
python examples/resnet18/package_example.py `
  --output-archive mount/resnet18/resnet18-model.zip
```

Expected marker: `PASS [real-model-host]`. The checkpoint itself is not
redistributed. Missing, stale, substituted, incomplete, or unvalidated model
workspaces publish no archive. Issue #7 combines this validated model boundary
with the matching trusted overlay for standalone board delivery.

## 8. Deploy to the PYNQ-Z1

`run_on_board.py` must not be launched from the Windows virtual environment.
Use the deployment wrapper to copy the required Python sources, ignored model
workspace, and verified Vivado artifacts to the board. The default
board endpoint is `xilinx@192.168.2.99`; override `-BoardHost` or
`-BoardUser` when necessary:

```powershell
& examples/resnet18/deploy_release.ps1 `
  -DeploymentId (Get-Date -Format yyyyMMdd-HHmmss) `
  -AllowArtifactCommitMismatch
```

This wrapper only creates an immutable release directory and copies the
example, shared runtime/export/model sources, Vivado artifacts, and deployment
metadata. It does not run the model, invoke `sudo`, claim a PASS, or retrieve
evidence. The `-AllowArtifactCommitMismatch` choice is recorded for the later
human validation; omit it when the artifacts were built from this exact
checkout.

## 9. Open the notebook and run the demo

In the PYNQ Jupyter interface (e.g., http://192.168.2.99:9090/notebooks/), open
the release directory printed by the deployment wrapper, then open:

```text
examples/resnet18/resnet18.ipynb
```

Select the board's PYNQ Python kernel and run one cell at a time.

Step 2 programs the FPGA and prints the runtime class, the 8 x 8 array, the 64
processing elements, and the bitstream path. Step 3 offers a dropdown of five
bundled photographs beside an upload widget; an upload overrides the dropdown.
Step 4 preprocesses the chosen picture on the board and shows it next to the
exact dequantized INT8 tensor the NPU receives, so you see the input before any
inference runs. Step 5 executes it and reports elapsed time, MAC count, and
physical job count; a job count above zero is what distinguishes this from a
host simulation. Step 6 prints the top-5 ImageNet labels with a bar chart, the
predicted class, and `CORRECT` or `INCORRECT` for a bundled picture whose class
is declared. An uploaded picture has no ground truth, so you judge it yourself.

One forward pass is 1,814,073,344 MACs and takes roughly an hour on the 8 x 8
overlay, so this is a start-it-and-talk demo rather than an interactive one.

Fetch the bundled pictures once, on the conversion host, into the ignored model
workspace:

```powershell
& build/resnet18-venv/Scripts/python.exe `
  examples/resnet18/scripts/download_gallery.py
```

`gallery-source.json` pins each image by URL, byte length, SHA-256, SPDX
licence, attribution, source page, and expected ImageNet class. The downloader
accepts only pinned HTTPS URLs on the approved host, rejects a length or digest
mismatch, refuses to overwrite a file whose digest differs, and publishes
nothing on failure. The five are Creative Commons photographs from Wikimedia
Commons - zebra, volcano, daisy, pizza, and goldfish - chosen because the
exported INT8 model predicts each one correctly. None is an ImageNet dataset
image and none is committed here. The PYNQ-Z1 has no route to the internet on
the direct Ethernet link, so run the downloader on the conversion host and let
the deployment wrapper carry `model/` to the board.

### Numerical acceptance

This notebook demonstrates; it does not accept. It performs no host/board
digest comparison and writes no evidence file. Deterministic numerical
acceptance on hardware is `run_on_board.py`, which runs the synthetic
regression tensor, compares every capture against `model/acceptance.json`, and
writes `board-evidence.json`. Run it only on the PYNQ-Z1, using the commit
values from `deployment.json`:

```bash
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
cd /home/xilinx/jupyter_notebooks/npu_resnet18/releases/<deployment-id>
sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
  examples/resnet18/run_on_board.py \
  --artifact-dir build/vivado/npu_matrix_8x8/artifacts \
  --expected-source-commit <40-character-artifact-commit> \
  --deployed-source-commit <40-character-deployed-commit> \
  --evidence board-evidence.json
```

It prints one of these markers:

```text
PASS [physical-pynq-z1]: real ResNet-18 board acceptance
PASS [physical-pynq-z1-development]: real ResNet-18 board execution
```

The development marker means an artifact/check-out commit mismatch was
explicitly allowed. It is execution evidence, not trusted release acceptance.
A physical PASS requires an actual `NPURuntime`; host backends cannot emit one.
Automated deployment and evidence collection belong to the CD script under
`.github/cd/`, not to this human demo workflow.

## Re-running generated steps

All download, conversion, validation, and archive outputs are intentionally
write-once. To repeat a step, choose a new output path or deliberately remove
only the corresponding ignored generated files after preserving any evidence
you need. The scripts never overwrite prior evidence silently.

## Numerical characterization

The demo confirms importer/runtime agreement on one image. Two additional
characterization runs quantify what INT8 quantization costs on this model.
Setup, caveats, and full per-layer tables are in
[`docs/exp/2026-09-20-int8-quantization-eval.md`](../../docs/exp/2026-09-20-int8-quantization-eval.md).

### Weights — essentially lossless

Per-output-channel INT8 quantization of the BN-folded weights introduces at
most 1.6% relative RMS error on any single layer. Original and dequantized
weight densities overlap visually on every layer, including the deepest
`layer4.1.conv2` where `max|W|` reaches 3.65.

Generate the visualization (HTML, self-contained, works offline):

```
python examples/resnet18/scripts/weight_distribution.py \
    --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \
    --out-prefix examples/resnet18/model/int8-eval/weight_dist
python examples/resnet18/scripts/build_weight_viz.py \
    --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \
    --output    examples/resnet18/model/int8-eval/weight_distribution.html
```

Open `examples/resnet18/model/int8-eval/weight_distribution.html` in any
browser. Blue fill = original density, orange line = dequantized density.

### Accuracy A/B on real ImageNet photographs

Same 1000 real images (one per class from
`EliSchwartz/imagenet-sample-images`), same preprocessing:

| | FP32 | INT8, calibration = 2 synthetic images (current) | INT8, calibration = 32 real images |
|---|---:|---:|---:|
| top-1 | 79.7% | 61.6% | **within noise of FP32** |
| top-5 | 95.0% | 93.1% | within noise of FP32 |

The 1000-image corpus is a curated one-per-class sample, not the ImageNet-1K
validation set; absolute figures are not comparable to published validation
numbers, only same-image FP32-vs-INT8 deltas are.

The 18-percentage-point top-1 gap under the current calibration is caused by
`max-abs` activation-scale estimation with only two calibration inputs, not by
the quantization format or the numeric contract. See #68 for the tracked fix.

Reproduce:

```
python examples/resnet18/scripts/download_int8_eval_corpus.py
python examples/resnet18/scripts/eval_int8_accuracy.py \
    --checkpoint examples/resnet18/model/resnet18-f37072fd.pth \
    --corpus     examples/resnet18/model/imagenet-calibration \
    --index      examples/resnet18/model/imagenet-calibration.index.tsv \
    --limit 1000 --calibration synthetic \
    --output examples/resnet18/model/int8-eval/result_synthetic.json
```

Substitute `--calibration real --calibration-size 32` for the recalibrated
A-side. Neither corpus nor result JSON is committed (both live under the
git-ignored `model/` workspace).
