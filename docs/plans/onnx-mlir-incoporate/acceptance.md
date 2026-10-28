# ONNX-MLIR Acceptance: 2026-10-25

Parent issue: #119. Implementation units: #139 (operators), #140 (export),
#141 (physical acceptance and removal). No vendor, ISA, driver or RTL edits.

## Inputs and toolchain

- Pinned source ONNX SHA-256:
  `a7c33f9ef85d06734cc9d1f943f7e2ba57c77e769df727f4d3e217d9f672b0cc`.
- ONNX-MLIR 0.5.1 revision `20f2b509a1edddcc38000b04bca5ab38c1f183f9`,
  LLVM 24 importer; immutable image digest in `toolchain.json`.
- Generic ONNX IR parsed with MLIR 22; CPU/NPU code generation used LLVM 22.1.8.
- Cortex-A9 ARMv7, NEON, VFPv3, 32-bit indices, hard-float ABI.
- Existing 16×16, MAX_K=256, ISA ABI 1.1 overlay. BIT SHA-256
  `bd6b8f44f2b25445aef61e7692beda62d85a35a84ccbf6448ce40ca4cc12f3db`;
  HWH SHA-256 `8ba3849b16c63faba539511c997a400f8b1f64c4f7f8707fae0f4a5af9d46748`.

The official importer's missing ARM backend is bypassed through an explicit,
verified IR boundary. ARM code generation uses the existing LLVM 22 backend.
The full imported graph, including 151 custom operations, was parsed before
specialization and partition.

## Numerical validation

`src/test/tests/test_onnx_lowering.py`: 11 tests passed with real LLVM code
generation and ORT 1.20.1 references. Tests cover normalization/residual order,
quantized endpoints, zero/constant/tiny/large ranges, ties-to-even, Sigmoid,
rotary positions, grouped heads, causal masks, cache writes and real-width
prefill/decode accumulation tails. Normalization, Sigmoid and real-width
attention fixtures compare exactly; other floating fixtures use explicit
tolerances. A constant near an FP32 midpoint catches literal truncation.

The whole-model tolerances were fixed before candidate comparison:

| Tensor | Relative tolerance | Absolute tolerance | Observed maximum error |
| --- | ---: | ---: | ---: |
| Logits | 0.001 | 0.01 | 0.000041962 |
| FP32 key/value cache | 0.001 | 0.002 | 0 (exact) |
| Greedy token IDs | exact | exact | 0 |

All 12 cases passed on the host and physical PYNQ-Z1: prefill lengths 36, 40,
42, each followed by three decode steps. References execute the unchanged full
source ONNX through ORT 1.20.1; candidate code does not generate reference
values. Fixture hashes and source identity are checked before execution.
The board's PYNQ Python reports no installed `onnxruntime` module.

Initial comparisons failed and were retained. The fixes preserve division
order, full literal precision, explicit FMA, the reference softmax polynomial
and eight-lane sum, and four-term decode accumulation. Tolerances were not
relaxed to make the failures pass.

## Physical results

| Check | Result |
| --- | --- |
| Imported signed contraction | Exact INT32; 1 job, 183 cycles, zero error |
| Full 12-case acceptance | 583,200 hardware jobs, 427,528,510 hardware cycles |
| Prefill 36 / 40 / 42 | 30.879 / 17.094 / 17.719 seconds |
| Decode calls in acceptance | 4.263–5.358 seconds each |
| Peak process RSS | 381,280 KiB (about 372 MiB) |
| Notebook generation | 32 tokens, exact independent reference IDs and text |
| Notebook timings | 25.3228 s prefill; 137.1133 s decode |
| Notebook hardware counters | 3,840 calls, 1,101,600 jobs, 577,335,014 cycles; zero error |

Wall times include CPU work, weight paging and NPU dispatch. They reflect the
recorded page-cache state. They are FPGA/ARM system measurements.

Deployment directory:
`/home/xilinx/jupyter_notebooks/npu-onnx-mlir-20261025/`.
The output-free public notebook ran in a real root PYNQ kernel via nbclient;
the acceptance-only verification cell compared its IDs/text before cleanup.
The first verification helper read a nonexistent statistics field; correcting
that helper and rerunning all cells produced `PHYSICAL_NOTEBOOK_32_PASS`.

## NPU parity and package audit

All 480 partition tasks, instruction bytes, weight offsets/layouts and the
entire weight blob match the pre-migration frontend. Weight blob SHA-256:
`2ae462acf89e2bfe0bd8f1f3649425b8a1563c9380bbc963fb5f8cc3717b4e0f`.
The ARM library has no legacy ORT bridge symbols, and the package contains no
ORT library or operator-session models. ARM ELF attributes and hard-float
loading were checked, followed by physical execution.

Removed the four production C/build files, their asset attachment and SmolLM
native-object requirement. Generic native-object/runtime-hook APIs remain for
other callers. The independent host ORT reference and board NumPy/NPU reference
remain available. Preparation no longer generates operator sessions implicitly.

Consumer audit also removed 13 unused ORT-only experiment build/debug wrappers
under `exp/1007_sw_stack/deploy/scripts/`, and removed the native-ORT mode from
its integer probe. Experiment results, downloaded artifacts and pre-existing
user changes were preserved. Current source, workflow and executable experiment
files have no references to the deleted files or `NativeOrtOperators`.

## Review of the release reference

The old frozen `NLP ...` output did not match the unchanged source ONNX for
the current exact 42-token input. The replacement baseline was generated from
the full source ONNX before candidate comparison, then matched by the physical
notebook. Neither its expected IDs nor text came from the compiled candidate.
The source/weight pins are unchanged.

Prompt: `Write a paragraph about a neural processing unit (NPU).`

Exact chat input IDs:

```json
[1,9690,198,2683,359,253,5356,5646,11173,3365,3511,308,34519,28,7018,411,407,19712,8182,2,198,1,4093,198,19161,253,8510,563,253,8844,4604,2723,365,26282,69,595,2,198,1,520,9531,198]
```

The independent `reference32.json` SHA-256 is
`4431ef1ee18fe2ddd1b67725965f99dba50a65f7ccd91f31cf1d36aa614e85da`.
The reviewed 32 output IDs/text are frozen in
[the CD acceptance source](../../../.github/cd/smollm.py).
Its strict output comparison and physical-counter requirements remain enabled.

## Reproduction and retained evidence

Run the commands in [README.md](README.md). `src/test/onnx_mlir_acceptance.py`
generates independent fixtures and checks host/board execution. On the verified
board, use `package/`, `reference/`, and `npu-onnx-mlir-20261025.bit` in the
deployment directory.

Machine-local evidence remains ignored under `build/onnx-mlir/`:

- `host-evidence.json`, `board-evidence.json`, `notebook-evidence.json`.
- `cli-smoke.json`: final post-removal runtime CLI returned source-reference
  token 49 on physical NPU, 120 calls, zero error. Final deployment hashes
  passed for all 53 manifest entries.
- `reference-fixtures/`, `reference32.json`, `smollm_onnx.executed.ipynb`.
- `npu-parity/parity.json`, original programs/IR, and diagnostic traces.
- `final-software-tests.log`: 60 tests run, 59 passed and the Docker-only
  import test skipped in that process. The separate Windows run with
  `NPU_TEST_ONNX_MLIR=1` passed all 14 import tests, including the real importer.

No RTL, constraints or overlay logic changed. RTL simulation/synthesis were
not rerun for this software migration. No push, release or merge was performed.
