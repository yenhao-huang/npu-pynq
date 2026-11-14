# Python environments on this machine

Two virtual environments hold the export front end and the host runtime.
Both were built from [src/export/requirements.txt](../../../src/export/requirements.txt)
and have the same versions, apart from the platform's own torch-mlir wheel.
The C++ tools (`pynqo-middle-end`, `pynqo-backend`) are not in either
environment; they build under `build/` with WSL's LLVM 22.

| Name | Where | Python | Runs on |
| --- | --- | --- | --- |
| `pynqo` | `/home/howard/.venvs/pynqo` (WSL) | 3.12.3 | Linux, everything in WSL |
| worktree `.venv` | `<worktree>/.venv` (Windows) | 3.12.10 | Windows; the C++ tools still run through `wsl.exe` |

From Windows Explorer, `pynqo` is at `\\wsl$\Ubuntu\home\howard\.venvs\pynqo`.

## Use

`pynqo`, in WSL:

```bash
source ~/.venvs/pynqo/bin/activate
cd /mnt/c/Users/User/Desktop/agent_workspace/npu/worktrees/npu-issue119-a
python -m src.export.cli build examples/resnet18/export.yml --out dist/resnet18/package
python -m src.runtime.cli classifier --backend sim --package dist/resnet18/package \
  --image dist/resnet18/model/demo-image.jpg --labels dist/resnet18/model/imagenet-classes.txt
```

Without activating it, call `~/.venvs/pynqo/bin/python` directly.

The worktree `.venv`, in PowerShell from the worktree root:

```powershell
.venv\Scripts\Activate.ps1
python -m src.export.cli build examples/resnet18/export.yml --out dist/resnet18/package
```

`src/export/pipeline.py` translates paths and runs the C++ tools in WSL, so
an export can start from either environment. On 2026-11-07 the ResNet-18
export was run from the worktree `.venv`, and host-simulator runs of
`python -m src.runtime.cli` were made in WSL (the runtime builds `libnpu_rt`
and the `__delegate__` dispatcher with the machine's compiler). An export
started from `pynqo` has not been run.

## Installed versions

Recorded on 2026-11-07.

| Package | `pynqo` (WSL) | worktree `.venv` (Windows) |
| --- | --- | --- |
| torch | 2.11.0+cpu | 2.11.0+cpu |
| torch-mlir | 20260930.892 | 20260930.602 |
| torchao | 0.18.0+cpu | 0.18.0 |
| torchvision | 0.26.0+cpu | 0.26.0+cpu |
| transformers | 5.21.1 | 5.21.1 |
| safetensors | 0.8.0 | 0.8.0 |
| numpy | 1.26.4 | 1.26.4 |
| PyYAML | 6.0.3 | 6.0.3 |
| pillow | 13.0.0 | 13.0.0 |

torch-mlir's dated wheels carry a different build number per platform; the
requirements file pins each one.

## Recreate

`pynqo`, in WSL:

```bash
python3 -m venv ~/.venvs/pynqo
~/.venvs/pynqo/bin/pip install -r src/export/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

The worktree `.venv`, in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r src/export/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
