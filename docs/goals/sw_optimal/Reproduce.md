# Reproduce the software-stack goal (#119)

All commands run from the repository root of `npu/issue119-a`.

## 1. Toolchain

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
