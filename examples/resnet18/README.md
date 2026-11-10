# ResNet-18 on PYNQ-Z1

Export ResNet-18 on your computer, then classify pictures with it on the
board. The NPU runs the 21 convolutions and the classifier; the ARM CPU does
the rest.

We last ran this on the board on 2026-11-04: the demo picture and all five
gallery pictures came out right (Samoyed at 90.2%), about 6 s per picture.
Uploading your own picture in the notebook hasn't been tried.

## Before you start

You need Linux. On Windows, run `wsl --install -d Ubuntu` in PowerShell, open
Ubuntu, and clone the repository there (under `~`, not `/mnt/c`, or the first
build crawls).

Install the tools as in the
[SmolLM guide](../smollm-safetensors/README.md#before-you-start). The board
also needs Pillow, matplotlib and ipywidgets for the notebook:

```bash
ssh -t xilinx@192.168.2.99 'sudo /usr/local/share/pynq-venv/bin/python3 -m pip install pillow matplotlib ipywidgets'
```

Download the model and the pictures into `dist/resnet18/model` (each file is
checked against a pinned hash):

```bash
# the model: ResNet-18 weights from https://download.pytorch.org/models/resnet18-f37072fd.pth
python -m src.deploy.resnet18_assets model
# the dataset: ImageNet class names and a test picture from https://github.com/pytorch/hub
python -m src.deploy.resnet18_assets demo
# the showcase: five photos from Wikimedia Commons for the notebook (optional)
python -m src.deploy.resnet18_assets gallery
```

## Export

```bash
python -m src.export.cli build src/export/configs/resnet18.yml --out dist/resnet18/package
```

Want to check it before touching the board? Run it on the host simulator:

```bash
python -m src.runtime.cli classifier --backend sim --package dist/resnet18/package \
  --image dist/resnet18/model/demo-image.jpg \
  --labels dist/resnet18/model/imagenet-classes.txt --expected-class Samoyed
```

## Build the NPU bitstream (recommended on Windows)

The export targets a 16 x 16 NPU, so the bitstream must be one too. Build it
on Windows, where Vivado is installed; PowerShell from the repository root
works. It refuses to run while `src/` has uncommitted changes.

```powershell
vivado -mode batch -source src/hw/vivado_tcl/npu_accelerator/build_overlay.tcl -tclargs --out dist/resnet18/overlay
```

Already have a 16 x 16 `npu_accelerator.bit` and `npu_accelerator.hwh`? Skip Vivado and
put them in `dist/resnet18/overlay/`. Cloned the repository inside WSL? Copy
the two files from Windows into WSL's `dist/resnet18/overlay/`.

## Run on the board

Copy the package, the bitstream, the pictures and the runtime to the board and
classify the demo picture:

```bash
bash src/deploy/deploy.sh dist/resnet18 examples/resnet18/resnet18.ipynb src/deploy/resnet18-gallery-source.json
ssh -t xilinx@192.168.2.99 'cd ~/jupyter_notebooks/resnet18 && sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli classifier --package package \
  --overlay overlay/npu_accelerator.bit --image model/demo-image.jpg \
  --labels model/imagenet-classes.txt --expected-class Samoyed'
```

It asks for the board's sudo password. The first run spends about 30 s
compiling the board-side libraries.

Prefer a notebook? Open <http://192.168.2.99:9090/notebooks/resnet18/resnet18.ipynb>,
pick the root PYNQ kernel and run the cells in order: you can pick a gallery
picture or upload your own and see the top-5 guesses.
