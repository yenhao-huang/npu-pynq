# Compiled official SmolLM2 board acceptance

## Result

Session: 2026-10-09. Branch: `npu/issue119-b`, based on PR #126 commit
`778c90c5d8340c362972a627c18eb27580d2251c`.
Run: `compiled-board-20261009-04` on PYNQ-Z1 `192.168.2.99`.

**PASS:** all eight notebook code cells executed in order, zero errors,
four saved figures, 32-token interactive generation and 35/35 independent audit
gates. The user's exact URL opens the verified output-bearing notebook:

[Board notebook](http://192.168.2.99:9090/notebooks/smollm_validation_20261009/exp/1007_sw_stack/deploy/smollm_board_official_quant.ipynb).

The active Jupyter process 420 maps `/home/xilinx/jupyter_notebooks` to port
9090 with base URL `/`. Its named notebook matches the executed file and audit
SHA-256 `80c862bc5d00768a857a5dfbede4a31520e91892abd625127c54ca94d7b4066f`.

## Measured physical generation

| Metric | Result |
| --- | --- |
| Official model | SmolLM2-135M-Instruct, unchanged pinned official INT8 ONNX |
| Layers / original integer projections | 30 / 210 |
| Compiled integer contractions per forward | 120, with exact QKV and gate/up fusion |
| Prompt / generated tokens | 40 / 24 |
| TTFT | 12.860208 s |
| TPOT | 1.543225 s/token |
| Decode throughput | 0.647994 tokens/s |
| Physical integer calls | 2,880 |
| Physical NPU jobs | 842,400 |
| Physical NPU cycles | 453,076,654 |
| Full unchanged ARM ONNX comparison | 24/24 token IDs and all logits exactly equal, max error 0 |
| Separate ARM compiled ISA-simulator comparison | Every token and logit exactly equal, max error 0 |
| Reset/repeat | Every token and logit exactly equal, max error 0 |

These are the recorded notebook generation timings. Initial smoke prefill was
26.62 s; paging and model warmup affect first-run latency. The historical Python
graph run recorded TTFT 41.22 s and TPOT 17.15 s/token; these are separate runs.

## Execution and provenance

`src/models/smollm/official.py` reads the pinned official ONNX and emits tensor
MLIR through existing `npu-partition`, `npu-lower`, LLVM ARM and `CompiledModel`.
There is no Python interpretation of the model graph in this engine. Dynamic
UINT8 activation quantization, original INT8 projection weights, zero-point
correction and scale multiplication order are retained. Float RMSNorm, DQL,
RoPE, GQA, sigmoid and the FP32 head remain explicit native ORT 1.20.1 calls.
This is a hybrid compiled graph, not a claim that every float operator is LLVM
arithmetic. Projection ISA programs and weights are compiled and packed once.

| Artifact | SHA-256 |
| --- | --- |
| Official ONNX | `a7c33f9ef85d06734cc9d1f943f7e2ba57c77e769df727f4d3e217d9f672b0cc` |
| Compiled ARM library | `03231a60d8aaa660ac0c5f4cf512ebf555540c84ad86a7e3a94d2fb920dc3930` |
| ISA programs | `4adf86b7101d64aa9c119e5bd4ffe56a492f2cf032dd381efa76249b13792d15` |
| Compiled weights | `2ae462acf89e2bfe0bd8f1f3649425b8a1563c9380bbc963fb5f8cc3717b4e0f` |
| Unchanged full ARM reference | `07fb395598ffb7bfeeb1352b613c6853f0ef103d2ee53174475ec1e3cd720869` |
| Identical ARM ORT library | `2763b2406e978b12026b0b6ebd92b5ce14084a66f06838dd4b304760121d2fa2` |

The host x86 oracle is separate. ARM and x86 dynamic quantization can diverge;
this audit explicitly validates ARM architecture, reference hash, original
model hash, prompt IDs and the identical ARM runtime binary.

## Source-only physical proof

A separate board process generated two tokens from a directory containing only
`src/`, with no `exp/`. It used the independent compiled package
`/home/xilinx/jupyter_notebooks/smollm_validation_20261009/model-assets/official-compiled`
and the matching bit/hwh pair under the independent staging directory.
It ran 240 contractions, 129,600 physical jobs and
106,535,526 cycles, with runtime error 0. Retaining these assets lets
shared source generation run after deleting the experiment directory.

## Debugging and tests

- Native inputs initially crossed to memrefs too early. Bufferization lost the
  later residual read and overwrote the embedding with the O projection. A real
  compiled regression reproduced the failure; keeping external arguments as
  tensor SSA with explicit read access fixed it.
- Adding read_only to early casts alone failed; those build artifacts remain.
- A later apparent output-head mismatch was a verification mistake: the host
  oracle directory was selected. All 30 layer boundaries, tied-head weights
  and final head input were exact. The corrected full ARM oracle comparison is
  exact for all 24 generated steps.
- Seven focused Linux tests passed: readonly mapping, descriptor limit,
  quantization, tokenizer, source-only entry points, real compiled UINT8
  projection and real compiled residual lifetime. Windows four source tests
  passed and the POSIX descriptor test was skipped.
- The four output figures were extracted and visually checked.

Failed runs, diagnostics, original source/notebook backups and old packages are
retained separately. Raw binaries, model assets, outputs and logs are ignored
and excluded from the patch.

## Reproduction and limits

See [shared-source instructions](../../../docs/acceptance/smollm-official.md).
Build the notebook with `scripts/build_compiled_notebook.py`, execute it with
`scripts/execute_official_notebook.py`, then run
`scripts/audit_compiled_notebook.py --run-id <run-id>` on the board.
`PYNQ_PASSWORD` is loaded from `.env` by the host sudo helper.

Exact prefill specializations are 36, 40 and 42 complete chat tokens; compile
additional exact lengths for other prompts. Arbitrary chunking would change
original DQL semantics. Context capacity is 512. CPU ISA simulation must use a
separate process because each compiled library owns one runtime state.

This closes the official SmolLM ONNX-to-MLIR physical milestone and source-only
physical runtime check. The cancelled automated multi-model workflow, original
W8+AWQ SmolLM package, Qwen3, ResNet-18 and routed 100 MHz timing still need their
own completed campaign evidence.

## Source consolidation (2026-10-10)

This refactor adds no source files. Seventeen previously added split files were
merged into the existing inference, compiler, model, runtime and compiler-test
files. The independent `src/inference/official/` package is removed.
`src.inference.cli` owns export, run, preparation and reference execution;
`LLMEngine` selects the pinned ONNX compiled engine from package metadata.
Native C kernels and pinned model/package artifacts retain their identity.

Validation:

- All 11 regressions pass through `src.test.tests.test_compiler` on Linux,
  including original W8/W4 compilation and the native residual lifetime test.
- The full frontend's rendered MLIR is byte-identical to accepted V3 source.
- Physical run `compiled-board-20261010-integration-01` produced all 24 tokens
  and all 24 x 49,152 logits exactly equal to the unchanged ARM ONNX oracle.
  Maximum absolute error is zero; physical activity is 842,400 jobs.
- This run observed TTFT 19.0615 seconds and TPOT 12.0331 seconds/token.
  Before model load, system available memory was 212,748 KiB versus
  345,508 KiB in the accepted previous run; swap was full. These conditions
  differ from the earlier 1.5432-second/token measurement. Library, ISA program
  and weight hashes match that run. This is not a new speed improvement claim.
- The extra complete notebook rerun was stopped in its separate CPU ISA
  simulator after more than ten minutes under memory pressure. Its audit
  records six attempted cells and the intentional SIGTERM error, rather than
  a fresh full-notebook PASS. Partial outputs remain under that run directory.
- The user's pre-refactor nine-cell notebook and outputs were restored from
  the verified backup, with only the compiled engine import migrated and an
  explicit provenance note. The backup SHA-256 is
  `faf76b7b1266164e45296026bbbb5ad4b9abce757e9026666075f2b3cdfddfab`.

Current source reproduction uses the shared CLI and compiler test module.
Earlier test-module names in the historical report above describe that run's
layout and are superseded by the consolidated entry points.

The shared `src.inference.cli run` command also passed a separate physical
smoke test from a fresh directory containing only `src`, with no `exp` or
`src/inference/official` directory. It selected the compiled model automatically
from metadata and produced two tokens: 240 compiled integer calls, 129,600
physical jobs, 106,549,146 cycles and runtime error zero. Source-only raw output
is saved as `runs/integration-20261010/source-only-integration-20261010.json`.
