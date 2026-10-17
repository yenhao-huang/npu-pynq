# Reproduce the software-stack goal (#119)

All commands run from the repository root. The compiled-stack commands use
PR #126 (`npu/issue119-a`); the official INT8 follow-up is currently local on
`npu/issue119-b`, based on PR #126 commit
`778c90c5d8340c362972a627c18eb27580d2251c`.

## 1. Toolchain

On a Mac, `.codex/skills/deploy/install-requirement-in-mac/references/scripts/mac_npu.sh`
does all of this: `doctor` lists what is missing, `install` adds it, and
`hw`, `sw`, `export` and `deploy` run the steps below.

| Need | Version used | Notes |
| --- | --- | --- |
| LLVM + MLIR + clang | 22.1 (Homebrew `llvm`; apt `llvm-22 llvm-22-dev mlir-22-tools libmlir-22-dev clang-22`) | `NPU_LLVM_BIN` overrides discovery |
| Linker for the board library | `ld.lld` (apt `lld-22`) or the `ziglang` wheel | used only for the ARM target |
| Python | 3.12 with `exp/1007_sw_stack/requirements-export.txt` | PyTorch/Transformers only for weights, AWQ and references |
| Verilator, Icarus | any recent | RTL checks |

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r exp/1007_sw_stack/requirements-export.txt ziglang regex
```

`npu-opt` builds itself on first use (`build/npu-opt`, CMake against the MLIR
found above).

## 2. Hardware (ISA, IF/ID front end)

```bash
make -C src/test lint sim                      # 14 testbenches incl. tb_npu_isa_frontend
python -m src.test.vectors.generate_isa_vectors --check
vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl   # self-hosted only
```

## 3. Compiler and operators

```bash
python -m unittest src.test.tests.test_isa src.test.tests.test_compiler src.test.tests.test_ops
```

## 4. Export and run on the host (simulated NPU)

```bash
python exp/1007_sw_stack/host/acceptance.py --json exp/1007_sw_stack/host/host_acceptance.json
```

This exports SmolLM2-135M-Instruct, Qwen3-0.6B and ResNet-18 for both the
host and the PYNQ-Z1, generates text / classifies the gallery on the host,
and records the results. Single models:

```bash
python -m src.inference.cli export smollm --out build/pkg/smollm --awq            # W8A8 + AWQ
python -m src.inference.cli export qwen3  --out build/pkg/qwen3 --bits 4 --group 64 --awq --kv int8
python -m src.inference.cli export resnet18 --out build/pkg/resnet18
python -m src.inference.cli run build/pkg/smollm --prompt "What is the capital of France?" --chat --stream
```

Other models: `qwen3_5` (hybrid linear attention, `Qwen/Qwen3.5-0.8B`) and
`qwen3_moe` (`python exp/1007_sw_stack/host/make_tiny_moe.py build/tiny-qwen3-moe` first).

## 5. Accuracy against Transformers / TorchVision

```bash
python exp/1007_sw_stack/host/compare_hf.py build/pkg/smollm          # logits, greedy text
python exp/1007_sw_stack/host/quant_eval.py build/pkg/smollm          # vs FP32 and fake-quantized
python exp/1007_sw_stack/host/awq_eval.py smollm                      # W8/W4 +- AWQ
python exp/1007_sw_stack/host/arch_check.py qwen3 --layers 2          # FP32 architecture check
python exp/1007_sw_stack/host/arch_check.py qwen3_moe --layers 4
python exp/1007_sw_stack/host/arch_check_qwen3_5.py --layers 4
python exp/1007_sw_stack/host/resnet_eval.py build/pkg/resnet18
```

## 6. On the PYNQ-Z1

The `exp-board` workflow does this end to end: a push to an issue-119 branch
that changes `exp/1007_sw_stack/request.json` builds the overlay (self-hosted
Vivado runner), exports the three packages (`exp/1007_sw_stack/export_all.sh`,
hosted runner), copies overlay + packages to the board and runs
`exp/1007_sw_stack/board/run_models.sh` (ISA GEMM check, ResNet-18 gallery,
SmolLM2 and Qwen3 chat), returning `results/` as an artifact.

By hand, with the overlay's `npu_matrix.bit/.hwh` and the packages copied to
the board:

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 exp/1007_sw_stack/board/run_models.py \
    --overlay overlay/artifacts/npu_matrix.bit --packages export --out results/models.json
```

## 7. From macOS, without the Windows runner

With the board cabled to the Mac (adapter set to 192.168.2.1/24) and an SSH
key installed (`ssh-copy-id xilinx@192.168.2.99`):

```bash
S=.codex/skills/deploy/deploy-pynq-macos/references/scripts/pynq_deploy.sh
$S preflight
$S all --run <exp-board run id with the ISA overlay>     # or --release vX.Y.Z / --dir DIR
ls build/deploy/results/*/results/                      # board.log, isa_gemm.json, models.json
```

`--dry-run` before the command prints the plan only.

## 8. Official SmolLM2 INT8 shared-source follow-up

This prepares the unchanged official ONNX model and executes its integer
contractions through the ISA runtime with ORT floating-point operators. It is
separate from the MLIR/LLVM compiled-package route above. Runtime and model
preparation do not require the experiment directory.

```bash
python -m pip install -r exp/1007_sw_stack/deploy/requirements-host.txt
export NPU_ONNX_MODEL=/path/to/smollm2-official
python -m src.inference.cli prepare-onnx prepare
python -m src.inference.cli reference --backend cpu \
  --prompt "Explain why the sky is blue in simple words." --max-new-tokens 24
python -m unittest src.test.tests.test_compiler -v
```

Build and stage the ARM runtime, prepared model and matching ISA overlay as
described in [the shared-source instructions](../../acceptance/smollm-onnx.md).
On the board, set `NPU_ONNX_MODEL` and `NPU_OVERLAY`, compile the bridge,
then run the source-only entry point:

```bash
bash src/runtime/build_onnxruntime_bridge.sh
python -m src.inference.cli reference --backend pynq \
  --prompt "write a paragraph about npu" --max-new-tokens 32
```

The notebook demo is
[`examples/smollm_onnx/smollm_onnx.ipynb`](../../../examples/smollm_onnx/smollm_onnx.ipynb).
Keep model/runtime/overlay assets outside `exp/` so that deleting experiment
directories preserves this entry point. Read the PYNQ connection password from
`.env` through `PYNQ_PASSWORD`. Historical board evidence and current verification
limits are recorded in [official-int8.md](details/official-int8.md).

## 9. Compiled official SmolLM ONNX on PYNQ

Follow the MLIR/LLVM section in
[shared-source instructions](../../acceptance/smollm-onnx.md) to compile the
pinned official model. Keep the package and matching ISA overlay outside `exp/`.
On the root PYNQ kernel, the shared source entry point is:

```bash
export XILINX_XRT=/usr
python -m src.inference.cli run --chat --stream --backend pynq --package "$COMPILED_PACKAGE" \
  --overlay "$NPU_OVERLAY" --prompt "write a paragraph about npu" \
  --max-new-tokens 32
```

For the exact eight-cell acceptance notebook, use
`exp/1007_sw_stack/deploy/scripts/build_compiled_notebook.py`,
`execute_official_notebook.py --run-id <run-id>`, then
`audit_compiled_notebook.py --run-id <run-id>` on the board. The notebook checks
`data/official-arm-oracle` and its runtime/reference hashes, not the separate
host x86 oracle. The compiled CPU reference runs in another process so it
cannot reconfigure the parent physical runtime.

Run `python -m unittest src.test.tests.test_compiler -v` in an LLVM 22 environment.
The source-only demo now uses `NPU_COMPILED_PACKAGE`. Default prefill
specializations are exactly 36, 40 and 42 complete chat tokens; unsupported
lengths require compilation rather than quantization-changing chunks.

Successful run: `compiled-board-20261009-04`, 35/35 gates;
[full compiled evidence](../../../exp/1007_sw_stack/deploy/CompiledResult.md).
