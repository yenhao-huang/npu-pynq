# Official SmolLM2 INT8 progress for issue #119

Updated 2026-10-09. Parent: PR #126, `npu/issue119-a`, commit
`778c90c5d8340c362972a627c18eb27580d2251c`. Follow-up: local
`npu/issue119-b`; changes are reviewable in its worktree and patch, not yet pushed.

## Scope within #119

The parent PR integrates ISA/front-end #123, MLIR/LLVM compiler #124 and
inference framework #125. This follow-up establishes physical execution of
the official quantized SmolLM2 artifact with unchanged quantization, tokenizer,
KV state and generation semantics. Integer products use the ISA NPU runtime;
floating-point operators use native ONNX Runtime 1.20.1.

The historical Python executor did not compile the ONNX graph through
MLIR/LLVM. The subsequent compiled official-model run below completes that
manual SmolLM milestone; the original automated multi-model campaign remains
open.

## Completed evidence

| Gate | Result | Scope |
| --- | --- | --- |
| Official model identity | PASS | HF revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`; 137,147,867 bytes; SHA-256 pinned in `src/compiler/onnx-source.json` |
| Lossless export | PASS | All 698 initializer tensors retain original bytes |
| Physical generation | PASS | Run `official-board-20261009-02`: 24/24 tokens match unchanged full ARM ONNX; maximum logit error `3.1948089599609375e-05`, fixed `rtol=atol=1e-3` |
| CPU integer comparison / reset-repeat | PASS | Every token and logit exactly equal to physical execution |
| Full notebook | PASS | Eight code cells, zero errors, four figures, 32-token editable prompt and all 25 independent audit gates |
| Physical activity | PASS | 5,040 integer calls; 842,400 jobs; 460,911,894 NPU cycles |
| Physical timing | Recorded | TTFT 41.2222 s; TPOT 17.1520 s/token; 0.0583 tokens/s |
| Source-only runtime | PASS on host | Checkout contains `src/` and no `exp/`; 24/24 tokens match full host ONNX, KV allclose, max logit error `2.384185791015625e-05` |
| Regression coverage | PASS | Five Linux tests; Windows four passed and POSIX descriptor limit skipped |
| Relocated native C bridge | Build PASS | ARM cross-compilation succeeds from source-only checkout using the staged ORT library |

Historical physical evidence is summarized in
[OfficialResult.md](../../../../exp/1007_sw_stack/deploy/OfficialResult.md).
The board notebook for that run is
[smollm_board_onnx_quant.ipynb](http://192.168.2.99:9090/notebooks/smollm_validation_20261009/exp/1007_sw_stack/deploy/smollm_board_onnx_quant.ipynb).
Its verified executed-file SHA-256 is
`6b1ab8ef3cb6f065f68018e69645fe1ff2ed425d3b46f967fc8a80e1c0728753`.
Raw model binaries, outputs and logs remain outside the patch.

## Source ownership after PR alignment

- `src/compiler/export.py`: pinned download, lossless export and operator
  preparation. This follows PR #126's removal of the legacy `src/export` tree.
- `src/inference/{engine,graph,tokenizer,cli}.py`: graph execution, official tokenizer, prefill/decode,
  KV state, generation, ORT bindings and CLI.
- `src/runtime/`: physical integer GEMMs, native C bridge and ARM build scripts.
- `examples/smollm_onnx/`: user-facing notebook; imports shared source only.
- `exp/1007_sw_stack/deploy/`: historical acceptance and compatibility entry
  points; runtime code does not import from this directory.

Reproduction: [shared-source instructions](../../../acceptance/smollm-onnx.md)
and [Reproduce.md section 8](../Reproduce.md#8-official-smollm2-int8-shared-source-follow-up).
Model/runtime/overlay assets must remain outside the experiment directory.

## Compiled official-model follow-up: PASS

Run `compiled-board-20261009-04` executed the ONNX frontend, existing NPU MLIR
partition/lower, LLVM ARM library and `CompiledModel` on PYNQ-Z1. All eight cells,
four figures, interactive generation and 35 independent audit gates passed.
Every 24-step token and logit exactly matches unchanged full ARM ONNX,
separate compiled ARM CPU ISA simulation and reset/repeat (max error 0).

- TTFT 12.860208 s; TPOT 1.543225 s/token; 0.647994 tokens/s.
- 2,880 fused integer calls; 842,400 physical jobs; 453,076,654 cycles.
- Named output-bearing notebook SHA-256:
  `80c862bc5d00768a857a5dfbede4a31520e91892abd625127c54ca94d7b4066f`.
- Source-only physical CLI PASS in a board directory with only `src/`, no
  `exp/`, independent package/overlay, 129,600 jobs and runtime error 0.
- Seven focused Linux regressions PASS, including actual compiled UINT8
  arithmetic and residual buffer lifetime.

`src/models/smollm/model.py` and `src/inference/cli.py` own the
frontend/build command; `src/inference/engine.py` and
`compiled_cli.py` own compiled generation. Native float kernels are explicit
ORT external calls; this is not a claim that every float op is LLVM arithmetic.
Exact prefill lengths 36, 40 and 42 are compiled, with context capacity 512.

Full evidence and failure/fix history:
[CompiledResult.md](../../../../exp/1007_sw_stack/deploy/CompiledResult.md).
The URL above now opens this compiled executed notebook; the earlier notebook
hash refers to the preserved historical run file.

## Remaining gates

1. Complete the cancelled original automated campaign for W8+AWQ SmolLM2,
   Qwen3 and ResNet-18, plus routed ISA-overlay timing at 100 MHz.
2. Compile additional exact prefill lengths when other prompts are needed.
3. Retain model quality evaluation separately from numeric equivalence: the
   official NPU paragraph prompt's topic substitution also occurs in the
   unchanged ARM reference.
