# ResNet-18 on PYNQ-Z1 (WSL)

[WSL](README.md) | [macOS](README.macos.md)

The host exports TorchVision ResNet-18 (IMAGENET1K_V1) through PyTorch:
our torchao config quantizes every convolution and the classifier to W8A8
(im2col plus one INT8 contraction each), `torch.export` captures the network,
torch-mlir imports it, `pynqo-middle-end` folds BatchNorm into the rescale after
each contraction and lowers it to linalg, and `pynqo-backend` (LLVM 22) builds the
package. On the board the 21
contractions run on the NPU and everything else on the Cortex-A9.

**Validated on PYNQ-Z1, 2026-11-01**, with the commands below and the notebook
(fresh root kernel, all cells): the demo image and the five gallery images are
all classified correctly (Samoyed 89.9%), 21 NPU calls and 35,174 jobs per
image, about 6 s per inference. Interactive uploads were not exercised.
After BatchNorm folding moved to `pynqo-middle-end`, the same images were run on
the board on 2026-11-04 with `src.runtime.cli classifier`, not the notebook: all
correct, Samoyed 90.2%, the same calls and jobs.

## Requirements

Run from the repository root in WSL Ubuntu. Install Python, CMake, Ninja and
LLVM/MLIR 22 as in the [SmolLM guide](../smollm-safetensors/README.md#requirements),
which also installs `src/export/requirements.txt` (torchvision and Pillow
included). The board needs its PYNQ environment, g++ and CMake, Pillow,
matplotlib and ipywidgets, and a 16 x 16, MAX_K=256, ISA ABI 1.1 BIT/HWH pair
(`vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl`).

## Export

```bash
python -m src.deploy.resnet18_assets model
python -m src.deploy.resnet18_assets demo
python -m src.deploy.resnet18_assets gallery
python -m src.export.cli build src/export/configs/resnet18.yml --out dist/resnet18/package
```

The downloaders verify size and SHA-256 and keep everything under the ignored
`dist/resnet18/model`. Check the package on the host simulator:

```bash
python -m src.runtime.cli classifier --backend sim --package dist/resnet18/package \
  --image dist/resnet18/model/demo-image.jpg \
  --labels dist/resnet18/model/imagenet-classes.txt --expected-class Samoyed
```

## Deploy and run

Set `OVERLAY` to the directory holding `npu_matrix.bit` and `npu_matrix.hwh`:

```bash
export OVERLAY=build/vivado/npu_matrix_16x16/artifacts
python -m src.deploy.resnet18 dist/resnet18/package --overlay "$OVERLAY" --out dist/resnet18/board
ssh xilinx@192.168.2.99 'mkdir -p ~/jupyter_notebooks/resnet18'
scp -r dist/resnet18/board/. xilinx@192.168.2.99:jupyter_notebooks/resnet18/
ssh -t xilinx@192.168.2.99 'cd ~/jupyter_notebooks/resnet18 && sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli classifier --package package \
  --overlay npu_matrix.bit --image model/demo-image.jpg \
  --labels model/imagenet-classes.txt --expected-class Samoyed'
```

The first run builds libnpu_rt on the board (about 30 s). Install missing
notebook packages with
`ssh -t xilinx@192.168.2.99 'sudo /usr/local/share/pynq-venv/bin/python3 -m pip install pillow matplotlib ipywidgets'`.

## Notebook

Open <http://192.168.2.99:9090/notebooks/resnet18/resnet18.ipynb> in the root
PYNQ kernel and run the six stages in order: find the bundle, load, choose or
upload a picture, preprocess, infer, and read the top-5 prediction.
`NPU_RESNET18_PACKAGE` and `NPU_RESNET18_OVERLAY` override the package and
overlay paths; `NPU_RESNET18_BACKEND=sim` selects the simulator.
