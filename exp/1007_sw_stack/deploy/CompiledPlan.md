# Official ONNX -> MLIR/LLVM integration

Goal: execute the exact named physical notebook through compiled entry points,
with original official quantization semantics and independent ARM oracle checks.

## Design

1. Add a pinned SmolLM ONNX frontend under `src/models/smollm/official.py`.
   Read official initializers, verify topology/attributes, preserve original
   UINT8 activation quantization and zero points, and fuse compatible QKV and
   gate/up contractions. Compute weight sums and NPU tile layout at export time.
2. Emit MLIR for the forward graph, integer centering/corrections, contractions
   and scale arithmetic. Use existing NPU partition/lower and LLVM codegen.
3. Retain native ORT float kernels as explicit external compiled calls for
   RMSNorm, dynamic quantization, RoPE, GQA, sigmoid and the FP32 head. The earlier
   independent evidence showed float rounding can change quantization boundaries.
   These calls are numerically required initially; they are not Python graph
   interpretation. Record the hybrid kernel provenance honestly.
4. Compile exact prefill lengths because official dynamic quantization spans
   the full input tensor; replacing it with arbitrary chunks changes the model.
   Provide decode plus explicit supported prefill specializations and reject
   unsupported lengths rather than silently changing quantization.
5. Extend existing codegen to link the native operator bridge; load compiled
   packages with `CompiledModel`. Keep model assets independent of `exp/`.
6. Adapt the named notebook to compiled generation and preserve all eight
   executable gates, independent full ARM oracle, physical jobs and saved outputs.

## Validation loop

| Hypothesis | Command / evidence | Result | Next decision |
| --- | --- | --- | --- |
| Existing LLVM and board environment are reusable | Mac LLVM version; board SSH/process inspection | LLVM 22.1.4; board reachable; existing user kernels retained | Implement frontend/native ABI without replacing board environment |
| Official projections permit fusion without requantization | Inspect original ONNX node inputs and initializer zero points | QKV and gate/up share quantized activation; all projection weight zero points are zero | Fuse output columns while retaining each branch scale |
| Reuse Mac LLVM for compilation | Source/prototype SCP request | Automatic approval rejected source transfer to Mac because that payload/destination lacked explicit authorization; no source transfer performed | Use local Docker LLVM/MLIR 22.1.8 |
| Compile official unsigned projection on the ISA simulator | `test_official_compiler` in the local LLVM image | First configure failed because CMake chose Make but the image provides Ninja | Set Ninja and an isolated build directory; retain failed configure artifacts |

Required final gates: compiler provenance and compiled runtime active; original
source identity; exact CPU-vs-NPU integer semantics; full 24-token ARM oracle
comparison with fixed tolerances; reset/repeat; all eight notebook code cells
executed with no errors; interactive generation and physical NPU activity.

## Compiled validation progress

- Real compiled UINT8 projection regression passed with zero points 0, 127, 128, 130 and 255, including fused branch scales.
- One-layer and full 30-layer MLIR/LLVM ARM compilation passed. Full package provides decode and exact prefill lengths 36, 40 and 42.
- First board invocation lacked XILINX_XRT under sudo; corrected runner environment and reset all stale execution counts.
- Compiled board run 02 passed physical integer probes and loaded backend 1 with MMIO, but failed the independent numerical oracle: first-token logit max error 21.456924. This run is not accepted. Its own CPU reference subprocess was interrupted, and notebook failure outputs were saved.
- Next experiment captures original first-layer outputs and compiled native-kernel inputs to locate the first numerical divergence.

- First-layer capture localized the first mismatch: all inputs through attention and O projection were exact; skip-normalization input 0 had been overwritten with O projection (input 1).
- Adding read_only to early to_buffer casts did not fix it; the new real compiled residual regression failed (19/21 elements). V2 package is retained and was not accepted.
- Keeping external-kernel inputs as dynamic tensor arguments with bufferization.access=read until function-boundary lowering fixed the regression. Both unsigned projection and native residual tests now pass with real MLIR/ISA/LLVM execution. Compile V3 and rerun physical validation.

- V3 physical run 03 retained the correct residual and first-layer capture is entirely exact. Full generation still failed the unchanged ARM oracle (first-row max logit error 6.3857799; token divergence begins at step 5). The run remains unaccepted.
- All seven focused Linux regressions passed. The all-layer diagnostic CPU integer graph was too slow; its own process was interrupted and replaced by the historically validated original graph with physical integer GEMMs, solely to locate the remaining mismatch. Final acceptance retains the independent unchanged full ARM ONNX oracle and separate ARM compiled ISA simulator.

- All-layer diagnostic showed all 30 layer boundaries, final head input and full
  FP32 tied-head weights were exactly equal to the original graph.
- Correction to the V3 interpretation above: the comparison accidentally used
  `data/official-oracle` (host x86), not `data/official-arm-oracle`. The actual
  unchanged full ARM ONNX oracle matches all 24 V3 physical token IDs and every
  logit exactly (max error 0). The remaining discrepancy was a validation-path
  error, not a second model arithmetic bug. Historical failed notebook outputs
  are retained. New notebook/audit explicitly require ARM architecture, original
  source/reference SHA-256 and identical ARM ORT binary.
- Run 04 executes all eight cells with the correct ARM oracle, separate compiled
  ARM ISA-simulator reference, reset/repeat and interactive generation.

- Final run 04: eight cells in order, zero errors, four figures, 32 interactive tokens, 35/35 independent gates. All 24 full ARM oracle logits/tokens, separate compiled ARM CPU logits/tokens and reset/repeat are exactly equal (max error 0). TTFT 12.860208 s; TPOT 1.543225 s/token; 842400 jobs and 453076654 cycles.
- Separate physical CLI generation passed in a directory containing only src and independent package/overlay assets, with no exp. Active Jupyter URL mapping and named notebook hash were saved. See CompiledResult.md.

## Source integration requested on 2026-10-10

Reuse the existing source files instead of maintaining a parallel official
package. Merge both CLIs into `src/inference/cli.py`, select the compiled
engine from package metadata, and keep tokenizer, reference graph, model
frontend, lowering helpers and runtime bindings in their existing modules.
Preserve native C ABI, full-prefill quantization and pinned source identity.
No new source files are required for this refactor.

Verify source-only commands, unsigned quantization, native residual lifetime,
original W8/W4 compiler behavior and byte-identical full-model MLIR. Then
rerun the exact board notebook and the shared CLI in a source-only checkout.
Preserve previous raw evidence and back up the board's source and notebook.
