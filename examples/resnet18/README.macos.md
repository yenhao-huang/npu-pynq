# ResNet18 on macOS

Platform guides: [Linux](README.md) | [Windows (PowerShell)](README.win.md) | [macOS](README.macos.md).

Run local commands from the repository root in the macOS Terminal (Bash or zsh). Commands after an interactive
`ssh` login run in the board's Linux shell. Install Python 3 and make sure
`ssh` and `scp` are available. Keep deployment products under `dist/`.

## Requirements and build boundary

Use a compiled AOT archive produced by the [Linux build guide](README.md),
or download the ResNet18 archive from a release containing the AOT example.
An older JIT archive is not interchangeable. Save it as
`dist/resnet18/npu-resnet18.zip`; retain the matching model, overlay and manifest.
This guide covers native macOS deployment of that archive.
[AMD lists Windows and Linux as supported Vivado hosts](https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Supported-Operating-Systems); macOS cannot run Vivado natively. Use a Linux build host for the complete overlay and AOT package.

## 1. Prepare the board once

Run the following locally; the quoted command installs missing notebook
dependencies in the board's PYNQ environment. SSH may ask for the sudo password.

```bash
ssh -t xilinx@192.168.2.99 "sudo /usr/local/share/pynq-venv/bin/python3 -m pip install pillow matplotlib ipywidgets"
```

## 2. Check the archive and deploy

```bash
test -f dist/resnet18/npu-resnet18.zip
scp dist/resnet18/npu-resnet18.zip xilinx@192.168.2.99:~/jupyter_notebooks/
ssh xilinx@192.168.2.99 'cd ~/jupyter_notebooks && unzip -o npu-resnet18.zip -d npu-resnet18'
```

## 3. Notebook and board test

Open <http://192.168.2.99:9090/notebooks/npu-resnet18/resnet18.ipynb> in the
board's PYNQ kernel. Connect to the package, program the FPGA, choose a bundled
or uploaded picture, and run every cell. Inspect top-5 labels, NPU counters and
the six block timings. The notebook defaults to physical `pynq` execution.
Transfer alone does not establish a physical PASS.

## Release CD

The [Linux guide](README.md#release-cd) describes the build, archive, separate
deployment and acceptance steps used by release CD. CI's non-interactive sudo
policy is separate from this interactive notebook workflow.
