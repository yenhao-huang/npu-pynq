# ResNet-18 on PYNQ-Z1 (macOS)

[WSL](README.md) | [macOS](README.macos.md)

Run local commands from the repository root in macOS Terminal with Python 3,
OpenSSH and scp available. Obtain `dist/resnet18/npu-resnet18.zip` from the
[WSL compilation steps](README.md#compile-and-package), or save a matching AOT
release archive at that path. Build the overlay on a Windows/Linux Vivado host;
this guide deploys the complete compiled archive from macOS. Native macOS
execution of this revised guide has not been validated.

## Deploy and run

Use the archive produced by the WSL guide or a release archive containing the AOT model,
matching ISA ABI 1.1 overlay and delivery manifest. An older JIT package is not
interchangeable. Run these commands locally from the repository root:

```bash
export BOARD=xilinx@192.168.2.99
export REMOTE=/home/xilinx/jupyter_notebooks/npu-resnet18
scp dist/resnet18/npu-resnet18.zip "$BOARD:~/jupyter_notebooks/"
ssh "$BOARD" "mkdir -p '$REMOTE' && /usr/local/share/pynq-venv/bin/python3 -m zipfile -e /home/xilinx/jupyter_notebooks/npu-resnet18.zip '$REMOTE'"
ssh -t "$BOARD" "cd '$REMOTE' && sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -m src.runtime.image_cli --package-root . --expected-class Samoyed --evidence image-acceptance.json"
```

Enter the board sudo password when prompted. Success reports Samoyed as top-1
and a positive physical NPU job count. Install missing notebook dependencies
in the board's PYNQ environment if necessary:

```bash
ssh -t "$BOARD" 'sudo /usr/local/share/pynq-venv/bin/python3 -m pip install pillow matplotlib ipywidgets'
```

Open [resnet18.ipynb](resnet18.ipynb) in the root PYNQ kernel at
<http://192.168.2.99:9090/notebooks/npu-resnet18/resnet18.ipynb>.
Run the six stages in order: connect, load, choose/upload, preprocess, infer,
and inspect top-5 predictions. The gallery retains image licensing metadata.
`NPU_RESNET18_PACKAGE` and `NPU_RESNET18_OVERLAY` override package and overlay
paths; `NPU_RESNET18_BACKEND=sim` explicitly selects host simulation.

The revised six-stage notebook and image CLI passed on PYNQ-Z1 with an existing
16 x 16 overlay and compiled model: top-1 Samoyed (89.80%), 35,174 physical NPU
jobs. Gallery downloads and interactive uploads were not exercised in that run.
A fresh Vivado build and the full release workflow were not rerun. Notebooks are
committed without outputs.
