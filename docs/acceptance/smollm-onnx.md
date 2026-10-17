# ONNX SmolLM2 INT8 from shared source

The runtime and preparation commands require only `src/` and a prepared model
directory. Removing `exp/1007_sw_stack/deploy/` does not remove any runtime,
exporter, native bridge source or runtime build script. Downloaded weights,
exported tensors, operator sessions, runtime libraries and the hardware overlay
are separate deployment assets; retain them outside the experiment directory.
This patch builds on the ISA and inference modules in PR #126.

## Source ownership

| Path | Responsibility |
| --- | --- |
| `src/inference/{engine,graph,tokenizer,cli}.py` | Graph execution, ONNX tokenizer, KV cache, generation and shared CLI |
| `src/runtime/compiled.py` | Compiled package loading and native ORT bindings |
| `src/runtime/isa_runtime.py` | Bounded CMA allocation and physical integer GEMMs |
| `src/runtime/c/onnx_ort_bridge.c` | Native ARM ONNX Runtime C bridge |
| `src/runtime/build_onnxruntime_arm.sh` | Pinned Cortex-A9 ONNX Runtime cross build |
| `src/runtime/build_onnxruntime_bridge.sh` | Native board bridge build |
| `src/compiler/export.py` | Pinned downloads, lossless tensor export, verification and operator session export |
| `src/models/smollm/model.py` | Official ONNX frontend into the existing NPU MLIR compiler |
| `src/inference/cli.py` | ARM compiled package with explicit native float kernels |
| `src/runtime/c/onnx_compiled_ops.c` | Native float ABI called by compiled ARM entry points |
| `src/inference/engine.py` | Direct `CompiledModel` prefill/decode and generation |
| `src/compiler/onnx-source.json` | Upstream revision, byte lengths and SHA-256 values |
| `src/test/tests/test_compiler.py` | Quantization, mapping, tokenizer and source independence regressions |

The old experiment entry points call these modules. They contain no separate
inference implementation. Experiment-only acceptance traces, oracle runners
and historical board evidence remain under `exp/`.

## Prepare the model on a host

Use an existing virtual environment, then install the host requirements.
Choose a model directory outside `exp/`; `NPU_ONNX_MODEL` may point to an
external shared model store. The examples below use ignored deploy staging.

```powershell
python -m pip install numpy==1.26.4 onnx==1.17.0 onnxruntime==1.20.1
$env:NPU_ONNX_MODEL = "$PWD/mount/smollm2-onnx"
python -m src.inference.cli prepare-onnx prepare
python -m src.inference.cli reference --backend cpu --prompt "Explain why the sky is blue." --max-new-tokens 24
```

`prepare` downloads only the pinned upstream files, checks their identities,
exports all initializer bytes without requantization, verifies every exported
initializer and exports native ORT floating-point operator sessions.

The resulting layout is:

```text
MODEL_DIR/
  onnx/       unchanged model and tokenizer
  onnx-graph/      graph.json and read-only tensors.bin
  onnx-ops/    six small ONNX operator sessions
  onnx-runtime/        board-only runtime library, header and compiled bridge
```

## Build and run on PYNQ

Run `src/runtime/build_onnxruntime_arm.sh` in an Ubuntu 22.04 Docker container
with an empty output directory mounted at `/work`. It pins ONNX Runtime 1.20.1
and its Eigen source revision and builds for Cortex-A9 NEON hard-float. The
initial build can take hours. Copy its `libonnxruntime.so.1.20.1` and
`onnxruntime_c_api.h` outputs into `MODEL_DIR/onnx-runtime/`.

For example, from the repository root on the host:

```powershell
New-Item -ItemType Directory -Force mount/onnx-runtime-build | Out-Null
$taskOutput = (Resolve-Path mount/onnx-runtime-build).Path
$taskScript = (Resolve-Path src/runtime/build_onnxruntime_arm.sh).Path
docker run --name smollm-ort-arm-build --mount "type=bind,source=$taskOutput,target=/work" --mount "type=bind,source=$taskScript,target=/build.sh,readonly" ubuntu:22.04 bash /build.sh
New-Item -ItemType Directory -Force "$env:NPU_ONNX_MODEL/onnx-runtime" | Out-Null
Copy-Item "$taskOutput/libonnxruntime.so.1.20.1", "$taskOutput/onnxruntime_c_api.h" "$env:NPU_ONNX_MODEL/onnx-runtime/"
```

Choose another container name if that name already exists.

Deploy `src/`, the prepared model directory and the matching PR #126 ISA
overlay to PYNQ. Use the board's existing PYNQ Python environment with NumPy,
XRT and GCC. A host ONNX Runtime Python wheel is not needed on the board.
Load any connection password from `.env` through `PYNQ_PASSWORD`.

```bash
export NPU_ONNX_MODEL=/path/to/smollm2-onnx
export NPU_OVERLAY=/path/to/npu_matrix.bit
bash src/runtime/build_onnxruntime_bridge.sh
python -m src.inference.cli reference --backend pynq \
  --prompt "write a paragraph about npu" --max-new-tokens 32
```

The matching `.hwh` must sit next to the overlay. Use the authorized root PYNQ
kernel/environment if required by the board's device access configuration.
The graph retains its ONNX dynamic UINT8 activation quantization and INT8
weights. Integer contractions run on the NPU; ORT handles floating-point
operators. Generation is greedy, stops at EOS and has a 512-token context cap.

## Notebook / Python use

`examples/smollm_onnx/smollm_onnx.ipynb` is an output-free demo that
imports only shared source and uses the environment variables above. Its
generation cell is editable. The equivalent board API is:

```python
import os
from pathlib import Path
from pynq import Overlay
from src.inference.engine import ONNXReferenceEngine
from src.runtime.compiled import NativeOrtOperators
from src.runtime.isa_runtime import NpuGemm

model = Path(os.environ["NPU_ONNX_MODEL"])
ops = NativeOrtOperators(model / "onnx-ops",
                         model / "onnx-runtime/libonnx_ort_bridge.so")
npu = NpuGemm(Overlay(os.environ["NPU_OVERLAY"]))
engine = ONNXReferenceEngine(model, integer_gemm=npu, float_ops=ops)
try:
    prompt = "write a paragraph about npu"
    ids, text, stats = engine.generate(prompt, max_new_tokens=32, chat=True,
        on_token=lambda _, piece: print(piece, end="", flush=True))
finally:
    npu.close()
    ops.close()
```

## Verification scope

The original physical validation passed eight notebook cells and 25 audit
gates, matching 24 generated tokens against the unchanged full ARM ONNX oracle.
This relocation preserves the graph arithmetic and NPU execution implementation.
Source-only integration is checked separately in a temporary repository
containing `src/` and no `exp/`: regression tests, preparation commands and real
CPU generation use the shared modules and external model assets.
These checks are not a new physical board acceptance run of the relocated CLI.

Relocation checks passed on 2026-10-09:

- Linux: all five regression tests passed, including the 64-descriptor limit
  and subprocess entry points with no experiment directory.
- Windows: four tests passed; the POSIX descriptor test was skipped.
- A source-only checkout prepared the pinned model and generated 24 tokens.
  Against the unchanged full ONNX on host ORT 1.20.1 / NumPy 2.5.2, all token IDs
  matched, KV outputs passed `rtol=atol=1e-3`, and maximum last-token logit error
  was `2.384185791015625e-05`.

## MLIR/LLVM compiled upstream ONNX model

The compiled frontend is `src/models/smollm/model.py`. It imports the pinned
ONNX weights and emits the forward graph into the existing MLIR NPU partition,
ISA lowering and LLVM ARM pipeline. `LLMEngine` calls compiled ARM
entry points directly. Native ORT 1.20.1 float kernels are explicit external
calls; the graph is not interpreted in Python. QKV and gate/up projections share
original quantized activation tensors and are fused by output columns, reducing
210 integer contractions to 120 without changing scales or zero points.

After preparing the pinned model and native ARM ORT library, build the native
operator object and the compiled package in an LLVM 22 environment:

```bash
arm-linux-gnueabihf-gcc -std=c11 -O2 -fPIC \
  -I "$MODEL_DIR/onnx-runtime" -c src/runtime/c/onnx_compiled_ops.c \
  -o /tmp/onnx_compiled_ops.o
python -m src.inference.cli export smollm --source "$MODEL_DIR/onnx/model_int8.onnx" \
  --out "$COMPILED_PACKAGE" --native-object /tmp/onnx_compiled_ops.o \
  --prefill-lengths 36,40,42
```

Keep the compiled package and matching ISA overlay outside `exp/`. On the board:

```bash
export XILINX_XRT=/usr
python -m src.inference.cli run --chat --backend pynq \
  --package "$COMPILED_PACKAGE" --overlay "$NPU_OVERLAY" \
  --prompt "write a paragraph about npu" --max-new-tokens 32
```

The compiled package contains ARM code, ISA programs, original INT8 projection
weights, an FP32 expansion of the source quantized tied embedding/head,
small native float operator models and the pinned ARM ORT library. No experiment
script is needed to load or generate. `--backend cpu` executes the same ARM
library on ISA simulation and is intended for a separate board process.

**Prompt lengths:** ONNX DynamicQuantizeLinear computes its scale over the
complete activation tensor. Compile exact prefill lengths for the complete chat
token sequence; the runtime rejects uncompiled lengths. Defaults cover the
notebook acceptance prompt (40), interactive example (42), and the lowercase
NPU prompt (36). Arbitrary chunking would change ONNX quantization semantics.

**Compiler regression:** external float-kernel inputs remain tensor SSA reads
until function-boundary bufferization. Early conversion to memrefs hid residual
reads and allowed activation buffers to be overwritten. The real compiled
residual-lifetime regression reproduces that failure and validates the fix.
Physical numerical acceptance is recorded separately after a full board run.

Compiled physical acceptance passed in run `compiled-board-20261009-04`: eight
cells, zero errors, 35/35 gates, all 24 tokens/logits exactly equal to unchanged
full ARM ONNX, separate ARM compiled CPU simulation and reset/repeat. TTFT
12.860208 s, TPOT 1.543225 s/token, 842,400 jobs. A separate physical CLI run
with only `src/`, no `exp/`, and independent assets also passed.
See [CompiledResult.md](../../exp/1007_sw_stack/deploy/CompiledResult.md).

## Integration into the existing source files (2026-10-10)

The upstream ONNX path uses the existing inference CLI and compiler exporter.
There is no separate `src/inference/official/` package. `LLMEngine` selects the
compiled ONNX engine from package metadata; the checkpoint path remains the
existing engine. The unchanged operator interpreter is the explicit reference
path. Native C kernels retain their ABI and quantization semantics.

```bash
python -m src.inference.cli prepare-onnx prepare --model-dir "$MODEL_DIR"
python -m src.inference.cli export smollm \
  --source "$MODEL_DIR/onnx/model_int8.onnx" --native-object "$NATIVE_OBJECT" --out "$PACKAGE"
python -m src.inference.cli run "$PACKAGE" --backend pynq --overlay "$OVERLAY" \
  --chat --stream --prompt "write a paragraph about npu" --max-new 24
```

## Release CD: deterministic interactive generation

The source interface now follows commit `778c90c5d8340c362972a627c18eb27580d2251c`:
`src.inference.cli` calls `src.compiler.export.export()` for every format.
The exporter accepts an existing `ModelGraph` or a source path plus a model
frontend. `src.models.smollm.model.build()` detects ONNX and safetensors sources,
and `src.inference.weights.load_weights()` supplies their initializer arrays.
Both frontends produce `ModelGraph` and use the same partition, weight packing,
ISA lowering, LLVM code generation and manifest packaging. The quantized ONNX
frontend supports the pinned SmolLM topology; this is not an arbitrary ONNX
graph importer. It retains original quantization and native float operator
semantics rather than requantizing into checkpoint arithmetic.

Both packages use `LLMEngine`, `Tokenizer`, `Sampler`, `GenerationStats` and
`CompiledModel`. Entry-point argument names and full-prompt prefill lengths
are metadata, not a second engine. Native dependencies and operator lifecycle
hooks are handled by `CompiledModel`; the tokenizer reads the ChatML default
system prefix from `tokenizer_config.json`. The original scalar-token ABI and
legacy `state` placeholder remain readable. Historical names remain readable for saved notebooks; canonical callers use
`LLMEngine` and `Tokenizer`.

Canonical calls use no official format branch:

```python
from src.compiler.export import export
export('/path/to/model.safetensors', '/path/to/checkpoint-package', model='smollm')
export('/path/to/model_int8.onnx', '/path/to/onnx-package', model='smollm',
       native_object='/path/to/arm-native-kernels.o')
```

Keep adjacent config/tokenizer files and the native ARM runtime assets.
ONNX deployment no longer requires an `onnx-graph/` interpreter export.
The `reference` command remains an independent diagnostic interpreter.
`--format onnx` is optional for `.onnx` sources.

The common exporter rebuilt the full ONNX package in
`mount/smollm-unified-20261010/`. MLIR, ARM library, weights and ISA programs
are byte-identical to the accepted package; only manifest metadata changed.
Physical run `cd-smollm-unified-20261010` used the shared engine and rebuilt
package metadata, matching all 32 frozen IDs/text with 1,101,600 jobs and zero
errors. The safetensors regression exports a real small checkpoint through
the same source API and generates with both current and original manifest
argument conventions. Raw results remain ignored under the experiment runs.

`cd.yml` builds SmolLM from the validated release commit in `build-smollm`.
It downloads and verifies the pinned upstream files, cross-builds ARM ORT
1.20.1 from pinned sources, compiles the native object and shared MLIR package,
and seals every package file with SHA-256 and the release commit/tag.
The ARM runtime build can be cached by its build-script identity; the model
is freshly compiled for each run. The first runtime build may take hours.

Configure `PYNQ_SMOLLM_REMOTE_ROOT` in the protected `pynq-z1-production`
environment as an absolute deployment directory. Connection settings use the
existing `PYNQ_BOARD_HOST` and `PYNQ_BOARD_USER`; SSH and passwordless sudo use
the existing CD runner provisioning. A missing root fails board preflight
before model or Vivado builds. Local development reads `PYNQ_PASSWORD` from
the workspace `.env`; no password is packaged or saved in evidence.

After ResNet acceptance, the SmolLM stage deploys into a new immutable
`REMOTE_ROOT/releases/TAG-RUN_ID-RUN_ATTEMPT` directory. It verifies the archive
checksum, the overlay BIT/HWH build manifest and the model's sealed file tree.
The detached physical job uses the same release's ISA overlay and current
`src/`, with no experiment dependency. Its durable exit status is polled over
SSH; failed evidence is retained when available. It never changes an active
deployment or the user's notebook.

The acceptance call is exactly:

```python
prompt = 'Write a paragraph about a neural processing unit (NPU).'
ids, text, stats = engine.generate(prompt, max_new_tokens=32, chat=True,
    on_token=lambda _, piece: print(piece, end='', flush=True))
```

`.github/cd/smollm.py` freezes all 32 token IDs and exact decoded text from
`compiled-board-20261009-04`. The reference was independently confirmed using
the unchanged full upstream ONNX model with ARM ORT 1.20.1 under Cortex-A9
QEMU in `cd-smollm-20261010`. The output starts with `NLP is a subfield`:
this gate establishes deterministic reproduction, not factual answer quality.
Any token or text difference fails. Empty or one-token output also fails.
Physical backend 1, a real MMIO pointer, 3,840 contraction calls, positive jobs
and cycles, and zero runtime errors are required.

Draft and public release gates require `npu-smollm-TAG.tar.gz`,
`SMOLLM_SHA256SUMS` and `smollm-interactive.json`. The evidence records prompt,
IDs, text, timing, physical counters, package hashes, overlay hashes and release
identity. Standalone consumers use the matching source commit and model assets.

Local validation on 2026-10-10 passed all 13 compiler/CD regressions. The new
acceptance code passed on the physical board with the existing validated ARM
package: all 32 IDs/text matched, 1,101,600 jobs, 579,130,076 cycles and no
errors. Prefill was 20.965751 s; mean decode step was approximately 1.37 s.
Raw evidence is in ignored `exp/1007_sw_stack/deploy/runs/cd-smollm-20261010/`.
This local run used uncommitted integration source and previously built model
assets; it is not evidence of a fresh release build or a GitHub CD run.

The obsolete `src/model/` and `src/export/` directories contained only ignored
Python bytecode caches and were removed. Keep active `src/models/`,
`src/compiler/export.py` and archived `src/deprecate/` sources; existing examples
still depend on the archive.

## ONNX naming

Active source names describe the input format: `ONNXGraph`,
`ONNXReferenceEngine`, `onnx_compiled_ops.c`, `onnx_ort_bridge.c`, and
`onnx-source.json`. Compiled generation still uses the common `LLMEngine`
and `Tokenizer`; export still uses `export()`. New model assets use `onnx/`,
`onnx-runtime/`, and the `NPU_ONNX_MODEL` environment variable.
Legacy import names, asset paths, environment variables, and native ABI
symbols are accepted only for reading previously generated packages.

The renamed native ABI was rebuilt locally on 2026-10-10 and physically
validated in `cd-smollm-onnx-names-20261010`: all 32 IDs and decoded text
matched the frozen reference, with 3,840 calls, 1,101,600 jobs, 579,416,566
cycles, backend 1 and zero runtime errors. Weights and NPU program binaries
were unchanged from the shared-source package. All 15 host regressions passed;
legacy import aliases and build-script syntax also passed. The ignored evidence
is `exp/1007_sw_stack/deploy/runs/cd-smollm-onnx-names-20261010/interactive.json`.
This used uncommitted local source; GitHub CD has not run for this change.
