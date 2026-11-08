# Reproduce independent JIT and ResNet18 AOT

Run from `worktrees/npu-issue119-a`. Preserve its existing staged/unstaged refactor.
The accepted physical results and exact identities are in [Result.md](Result.md)
and [evidence/pynq-notebook.json](evidence/pynq-notebook.json).

## Dependency isolation and host tests

On Windows, place temporary files inside the worktree (the sandbox's default
system temporary directory can reject nested test files):

```powershell
$goal = 'exp/.docs/goals/jit-independent-and-aot-support-resent18'
New-Item -ItemType Directory -Force build/tmp | Out-Null
$env:TMP = (Resolve-Path build/tmp).Path
$env:TEMP = $env:TMP
python "$goal/verify_without_jit.py"
python -m unittest discover -s src/jit/tests -q
python -m unittest discover -s src/jit/examples/resnet18/tests -q
```

Expected: 97 core tests (11 LLVM-dependent tests skipped on Windows), 6 public
ResNet image/delivery tests, 35 matrix/delivery tests, 90 optional JIT tests and
87 optional JIT example tests. The isolation script hides the whole JIT source
in a verified unused `build/` path and restores it in `finally`.

## Prepare the checkpoint, image and graph

The example README describes a normal single-host build with all pinned
requirements installed. This campaign reused an existing Windows PyTorch
environment and used the default Python for Pillow:

```powershell
$goal = 'exp/.docs/goals/jit-independent-and-aot-support-resent18'
$TorchPython = '../../npu_repo_in_pynq/build/resnet18-venv/Scripts/python.exe'
python examples/resnet18/scripts/download_model.py
python examples/resnet18/scripts/download_demo_assets.py
& $TorchPython "$goal/compile_staged.py" stage
python -c "from PIL import Image; import numpy as np; from src.export.graphs.resnet18.model import preprocess; np.save('examples/resnet18/model/reference-input.npy',preprocess(Image.open('examples/resnet18/model/demo-image.jpg')))"
& $TorchPython "$goal/reference_image.py"
```

The checkpoint SHA256 is
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
All downloads are verified against the example's source metadata. The reference
reads `reference-input.npy` if Pillow and PyTorch are split across environments.
Expected: Samoyed, approximately 89.8023%, and 1000 saved reference logits.
Model sources, images, NumPy arrays and compiled packages remain ignored under
`examples/resnet18/model/`.

## Compile on Linux

The configured SSH alias `agent_workspace` mounts this same worktree at
`/workspace/npu/worktrees/npu-issue119-a`. Run there on Debian 12 amd64:

```bash
cd /workspace/npu/worktrees/npu-issue119-a
GOAL=exp/.docs/goals/jit-independent-and-aot-support-resent18
python3 "$GOAL/prepare_toolchain.py" build/llvm22 --lock "$GOAL/evidence/llvm-packages.json"
export NPU_LLVM_BIN="$PWD/build/llvm22/root/usr/lib/llvm-22/bin"
export LD_LIBRARY_PATH="$PWD/build/llvm22/root/usr/lib/x86_64-linux-gnu:$PWD/build/llvm22/root/usr/lib/llvm-22/lib"
export NPU_OPT_BUILD="$PWD/build/resnet-npu-opt"
export NPU_BUILD_JOBS=4
python3 "$GOAL/compile_staged.py" compile
python3 "$GOAL/compare_native.py"
```

The helper extracts exact LLVM/MLIR 22.1.8 packages into ignored `build/`; it
changes no system packages or apt sources. The lock records package filenames,
versions and hashes. The graph interchange is MLIR, JSON and NumPy, not pickle.
CMake and a system C++ standard library are required. The freestanding ARM
runtime and linker do not require an ARM sysroot for this image model.

Expected: 21 NPU tasks, 35,174 jobs and two shared libraries (`host/model.so`,
`pynq/model.so`). The native comparison requires all 1000 logits within
rtol=0.002/atol=0.002 and the correct Samoyed top-1; this campaign measured an
exact match (maximum absolute difference 0). Cold compile timing includes the
first npu-opt build in the `partition` bucket; do not call that warm pass latency.

For a normal single-machine build, instead use
`python examples/resnet18/scripts/convert_model.py` with the example's pinned
PyTorch/Pillow requirements and the LLVM environment above.

The additional `test_ops` run used Linux Python 3.11: four tests passed, but the
existing embedding and linear-attention implementations require Python 3.12
f-string grammar and two tests could not import. Use Python 3.12 to run the
complete LLM compiler suite; those unrelated implementations were not modified.

## Deploy the independent board payload

PYNQ-Z1 is `192.168.2.99`, alias `pynq`, user `xilinx`. This run reused the
matched 16 x 16 ISA pair from the board's existing
`smollm-compiled-stage-20261009` deployment. It did not build a new overlay.
Confirm no other inference is active before programming the FPGA. Its source
commit is not inferred; the accepted BIT/HWH hashes are recorded in Result.md.

From Windows:

```powershell
New-Item -ItemType Directory -Force build/pynq-overlay | Out-Null
scp pynq:/home/xilinx/jupyter_notebooks/smollm-compiled-stage-20261009/smollm-compiled-stage-20261009.bit build/pynq-overlay/resnet18_aot.bit
scp pynq:/home/xilinx/jupyter_notebooks/smollm-compiled-stage-20261009/smollm-compiled-stage-20261009.hwh build/pynq-overlay/resnet18_aot.hwh
python "$goal/stage_board.py"
$send = '.codex/skills/deploy/send-to-pynq-board/references/scripts/send_to_pynq_board.ps1'
& $send -BitPath build/pynq-overlay/resnet18_aot.bit -HwhPath build/pynq-overlay/resnet18_aot.hwh -OverlayName resnet18_aot -BoardHost 192.168.2.99 -RemoteRoot /home/xilinx/jupyter_notebooks/npu-issue119-a-aot-20261018 -ExtraPath build/resnet18-board.tar.gz -DryRun
# After inspecting the exact source/destination pairs, run the same command:
& $send -BitPath build/pynq-overlay/resnet18_aot.bit -HwhPath build/pynq-overlay/resnet18_aot.hwh -OverlayName resnet18_aot -BoardHost 192.168.2.99 -RemoteRoot /home/xilinx/jupyter_notebooks/npu-issue119-a-aot-20261018 -ExtraPath build/resnet18-board.tar.gz
ssh pynq "cd /home/xilinx/jupyter_notebooks/npu-issue119-a-aot-20261018/resnet18_aot && tar --warning=no-timestamp -xzf resnet18-board.tar.gz && /usr/local/share/pynq-venv/bin/python3 run_on_board.py --package-root . --verify-only"
```

The 39-file software payload excludes the exporter, JIT, Torch checkpoint and
PyTorch. SHA256 verification precedes notebook execution. Existing demos remain
untouched. `stage_board.py` records the starting Git HEAD plus the actual dirty
working-tree file hashes rather than claiming those changes were committed.

## Execute the actual six-block notebook

The board already has Python 3.10.4, PYNQ, NumPy, Pillow 9.0.1, matplotlib,
ipywidgets, nbformat and nbclient. The runtime supports the older Pillow API.
No board packages or network settings were changed.

```powershell
python "$goal/board_command.py" --sudo 'env XILINX_XRT=/usr NPU_RESNET18_BACKEND=pynq NPU_RESNET18_OVERLAY=/home/xilinx/jupyter_notebooks/npu-issue119-a-aot-20261018/resnet18_aot/resnet18_aot.bit /usr/local/share/pynq-venv/bin/python3 /home/xilinx/jupyter_notebooks/npu-issue119-a-aot-20261018/resnet18_aot/execute_notebook.py /home/xilinx/jupyter_notebooks/npu-issue119-a-aot-20261018/resnet18_aot'
```

`board_command.py` loads `PYNQ_PASSWORD` from the workspace `.env` into memory
and sends it only on stdin to `sudo -S`; it never prints or saves the password.
SSH uses the existing key. Do not put credentials into command arguments.

`execute_notebook.py` verifies every payload file, executes all six original
stages using the PYNQ kernel, saves `resnet18.executed.ipynb`, extracts the final
block-time JSON and requires physical backend, positive NPU jobs and correct
Samoyed prediction. The notebook explicitly enables inline plots; do not force
an Agg-only backend if figure outputs are required.

Expected final run: inference approximately 4.849 s; six-block sum 28.042 s;
full notebook 47.449 s; 35,174 physical jobs and 1,814,073,344 MACs. CPU loads,
kernel startup and plotting caches affect timings. Use the reported monotonic
wall durations, not the board's unsynchronized calendar clock.

Retrieve artifacts:

```powershell
scp pynq:/home/xilinx/jupyter_notebooks/npu-issue119-a-aot-20261018/resnet18_aot/notebook-evidence.json "$goal/evidence/pynq-notebook.json"
scp pynq:/home/xilinx/jupyter_notebooks/npu-issue119-a-aot-20261018/resnet18_aot/resnet18.executed.ipynb build/resnet18.executed.ipynb
```

The accepted executed notebook is retained at
`examples/resnet18/resnet18.ipynb`; its preview and prediction figures are also
saved in `evidence/`. It remains an interactive demo: choose a gallery picture
or upload one, then run preprocessing, inference and prediction in sequence.
The automated run selects the default bundled photograph. The upload path was
retained and syntax/structure checked; no manual upload interaction is claimed.

The first headless run is retained separately as
`evidence/pynq-notebook-headless.json`. It passed hardware inference but did not
emit figures, so the final accepted run explicitly enabled inline display.
Generated BIT/HWH, models, libraries, tar archives and raw logs are not committed.

## Open the deployed notebook interactively

The notebook now automatically resolves either `artifacts/npu_matrix.bit` or
`resnet18_aot.bit` at the release root, requiring its matching HWH. No overlay
environment variable is needed for this deployment. After updating an already
open notebook, reload it from disk and rerun blocks 1 and 2.
