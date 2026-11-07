# ResNet-18 on PYNQ-Z1 (macOS)

[WSL](README.md) | [macOS](README.macos.md)

The export back end (`npu-compile`) is built for Linux with LLVM 22, so macOS
cannot export the package yet. Export and stage it with the
[WSL guide](README.md) and copy `dist/resnet18/board` to the Mac. Deploying
from macOS has not been run against the board; the steps are those of the WSL
guide.

## Deploy and run

From the repository root, with OpenSSH:

```bash
ssh xilinx@192.168.2.99 'mkdir -p ~/jupyter_notebooks/resnet18'
scp -r dist/resnet18/board/. xilinx@192.168.2.99:jupyter_notebooks/resnet18/
ssh -t xilinx@192.168.2.99 'cd ~/jupyter_notebooks/resnet18 && sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python3 -m src.runtime.cli classifier --package package \
  --overlay npu_matrix.bit --image model/demo-image.jpg \
  --labels model/imagenet-classes.txt --expected-class Samoyed'
```

Then open <http://192.168.2.99:9090/notebooks/resnet18/resnet18.ipynb> in the
root PYNQ kernel and run the six stages in order.
