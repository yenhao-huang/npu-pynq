# Official SmolLM2 INT8 deployment

## Shared source entry points

The inference implementation, model preparation and native runtime builds now
live under `src/`. This directory retains experiment wrappers and historical
acceptance evidence. For a deployment that survives removing this directory,
follow [the shared-source instructions](../../../docs/acceptance/smollm-official.md)
and keep the prepared model, runtime libraries and overlay outside `exp/`.
The independent demo is `examples/smollm_official/smollm_official.ipynb`.

This source-only patch builds on PR #126 (`npu/issue119-a`, base
`778c90c5d8340c362972a627c18eb27580d2251c`). Its ISA and inference modules
must be present before applying this patch to `dev`.

The official Hugging Face ONNX model is retained without requantization.
The graph adapter sends integer projection GEMMs to the physical NPU and
uses ONNX Runtime 1.20.1 for floating-point operators. It supports official
chat formatting, KV caching, streamed generation and CPU/ARM oracle checks.
The model's UINT8 activation and INT8 weight zero points are corrected exactly.
Weights share a single read-only file mapping to support the board's descriptor
limit. The default context limit is 512 tokens.

## Host preparation

Run from the repository root with Python 3.10 or newer. Generated artifacts
remain under this directory's ignored `packages/`, `downloads/`, `data/` and
`runs/` directories. The checked-in notebook intentionally has no outputs.

```powershell
python -m pip install -r exp/1007_sw_stack/deploy/requirements-host.txt
python exp/1007_sw_stack/deploy/scripts/fetch_official_onnx.py
python exp/1007_sw_stack/deploy/scripts/official_inspect.py
python exp/1007_sw_stack/deploy/scripts/official_export.py
python exp/1007_sw_stack/deploy/scripts/official_verify_export.py
python exp/1007_sw_stack/deploy/scripts/prepare_official.py operators
python exp/1007_sw_stack/deploy/scripts/official_reference.py
python exp/1007_sw_stack/deploy/scripts/prepare_official.py arm-inputs
```

`src/compiler/official-source.json` pins the upstream commit, byte counts and SHA-256 for
every downloaded file. Downloads are verified before replacing cached files.
The host reference compares all logits and KV outputs for 24 decode steps.

## ARM runtime and independent oracle

The following PowerShell commands build the Cortex-A9 runtime and run the
unchanged full ONNX model under ARM QEMU. The initial build can take hours.
The retained image is needed because the oracle links the runtime in `/opt`.

```powershell
$taskBuild = (Resolve-Path exp/1007_sw_stack/deploy/downloads/ort-arm-build).Path
docker run --name npu-official-ort-build --mount "type=bind,source=$taskBuild,target=/work" ubuntu:22.04 bash /work/build_ort_arm.sh
docker commit npu-official-ort-build npu-official-ort-arm:1.20.1
docker run --rm --mount "type=bind,source=$taskBuild,target=/work" npu-official-ort-arm:1.20.1 bash /work/run_arm_ort_oracle.sh
python exp/1007_sw_stack/deploy/scripts/prepare_official.py arm-reference
```

Use a new container name if that name already exists. The assembly helper checks
the oracle's shapes, finite logits, greedy IDs, prompt and model hash before
writing its reference package.

## Board deployment and notebook

Copy the repository's `src/` and this deploy directory with its generated
`packages/` and `data/` to the board, keeping their repository-relative layout.
Stage the matching PR #126 ISA overlay at
`hardware/overlay/npu_matrix.bit` and `hardware/overlay/npu_matrix.hwh` under
the deploy directory. The board requires its PYNQ environment, XRT and GCC.
Read any board connection password from `.env` as `PYNQ_PASSWORD`.

On the board, run `bash scripts/compile_official_ort_bridge.sh` from the deploy
directory. Open `smollm_board_official_quant.ipynb` with the root PYNQ kernel and
run all eight code cells in order. Cell 7 allows editing the prompt and generation
limit. Generation stops at EOS; `max_new_tokens` is an upper bound.

`execute_official_notebook.py` can execute and checkpoint the notebook with
nbclient. `audit_official_notebook.py --help` describes the independent final
audit. These gates compare physical generation, CPU integer execution, reset
repeat and the unchanged full ARM ONNX oracle, and verify real NPU job counts.

## Evidence and patch verification

`OfficialResult.md` records the earlier board run: eight executed cells,
25 passing audit gates and 24 matching ARM oracle tokens. That historical run
used the inference sources copied into this patch. Raw logs, model binaries,
runtime libraries, overlays and output-bearing notebooks stay outside Git.

Patch integration was checked separately: five Linux regression tests pass,
including two graph instances under a 64-descriptor limit, exact integer zero
point corrections, UTF-8 official chat formatting and entry points in a temporary
checkout containing only `src/`. Windows passes four
tests and skips the POSIX descriptor limit test. The shared tokenizer now reads
UTF-8 explicitly. The new pinned downloader's cache validation and all three
assembly phases were checked using the existing model and ARM oracle outputs;
698 initializer tensors retain identical bytes. This patch does not claim a
fresh physical board run of those helpers.

```powershell
python -m unittest src.test.tests.test_compiler -v
```

## Current compiled official-model acceptance

The named notebook now uses MLIR/LLVM ARM entry points through `CompiledModel`,
with explicit native ORT float kernels. Run `scripts/build_compiled_notebook.py`
to generate this version, `scripts/execute_official_notebook.py --run-id <id>`
to execute all cells, and `scripts/audit_compiled_notebook.py --run-id <id>` for
independent compiled-path gates. The historical graph-adapter builder/auditor
remain for their original runs and have different integer call accounting.

Run `compiled-board-20261009-04` passed eight cells and 35 gates, with exact
24-step unchanged ARM ONNX, CPU simulation and reset/repeat comparisons.
[CompiledResult.md](CompiledResult.md) contains timings, hashes, source-only
physical proof and exact prefill length constraints. Runtime and model assets
are independent of this experiment directory.
