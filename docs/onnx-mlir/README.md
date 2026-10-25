# ONNX-MLIR Import and Compatibility Checks

The integration currently inventories an ONNX model and imports it to ONNX
dialect MLIR. It does not partition the imported module, produce a runnable
model package, or replace the production ORT path yet.

## Requirements

- Python with ONNX installed, as required by the existing ONNX export path.
- Docker with Linux containers for the pinned upstream compiler.
- Run commands from the repository root. Output directories must be new or
  empty, so a failed import cannot reuse an older successful artifact.

The version lock is
`src/export/compilers/onnx_mlir/toolchain.json`. Its image digest is immutable;
normal imports do not pull images or access the network. Pull it once:

```bash
docker pull ghcr.io/onnxmlir/onnx-mlir@sha256:5df1c1d3f1d8d5207c9b408e9d6e0ec2ae9926c6948b4b5c745358a2b47ad2e8
```

## Inspect and Import

These commands work from Bash and native PowerShell. Replace the model path
if the pinned source is stored elsewhere.

```text
python -m src.export.compilers.onnx_mlir.pipeline --source dist/smollm2-onnx/onnx/model_int8.onnx --out build/onnx-mlir/inventory --inventory-only
python -m src.export.compilers.onnx_mlir.pipeline --source dist/smollm2-onnx/onnx/model_int8.onnx --out build/onnx-mlir/import
```

Outputs:

| File | Meaning |
| --- | --- |
| `inventory.json` | Source SHA-256, opsets, tensor metadata, nodes, attributes, and conversion diagnostics |
| `import.json` | Import status, command, toolchain provenance, and later-stage status |
| `compiler.log` | Combined compiler output, including errors or timeout diagnostics |
| `model.onnx.mlir` | Imported ONNX dialect module on success |
| `model.tmp` | Upstream diagnostic IR with large constants abbreviated, if emitted |

Inventory statuses are conservative: `unverified` means a schema is known to
the installed ONNX library, not that ONNX-MLIR or ARM supports that operator.
`requires_conversion` identifies custom domains or missing schemas. Inventories
include nested graphs and local function bodies. Tensor attributes are
summarized instead of embedding their weight bytes in JSON.

`compile_ready` remains false even after successful import. Partition, codegen,
and board validation each remain `not-run`. The importer intentionally retains
custom operations for later conversion; their presence is not CPU lowering
coverage. External tensor bundles are inventoried but currently rejected for
import until their source identity and mount boundaries are verified.

An explicit native compiler can be selected with `--compiler /path/to/onnx-mlir`.
Its provenance is recorded as `unverified-native`; the pinned image's source
revision is not asserted for that binary. Native import does not establish
ARM32 or NPU compatibility.

## Verified Compatibility Results

On 2026-10-25:

- ONNX-MLIR 0.5.1 at `20f2b509a1edddcc38000b04bca5ab38c1f183f9` imports
  a dynamic-shape Sigmoid fixture and the pinned SmolLM2 model successfully.
- The pinned SmolLM2 source hash is
  `a7c33f9ef85d06734cc9d1f943f7e2ba57c77e769df727f4d3e217d9f672b0cc`.
  It contains 1,212 nodes and 698 initializers, with default opset 14 and
  `com.microsoft` opset 1.
- 151 operations require explicit conversion in this source: one
  `SimplifiedLayerNormalization`, 60 `RotaryEmbedding`, 30
  `GroupQueryAttention`, and 60 `SkipSimplifiedLayerNormalization` nodes.
  The imported IR preserves these as `onnx.Custom`.
- The upstream image uses LLVM 24 at
  `43574226b712f94de17f07392a64282ac8ddab23`. The existing NPU toolchain uses
  LLVM 22. Their interchange boundary remains unverified.
- A direct ARM object compilation of the Sigmoid fixture with `--EmitObj`,
  `--mtriple=armv7-unknown-linux-gnueabihf`, and `--mcpu=cortex-a9` fails:

```text
error: Target architecture is unknown: No available targets are compatible with triple "armv7-unknown-linux-gnueabihf"
```

The pinned image is therefore an import tool, not a validated ARM compiler.
Before production integration, supply an ARM-enabled ONNX-MLIR/LLVM build and
prove an adapter to the unchanged NPU partition input. No vendor or RTL changes
have been made for this integration.

## Tests

```text
python -m unittest src.test.tests.test_onnx_import -v
```

Enable the real compiler test after pulling the image:

```bash
NPU_TEST_ONNX_MLIR=1 python -m unittest src.test.tests.test_onnx_import -v
```

In PowerShell:

```powershell
$env:NPU_TEST_ONNX_MLIR = '1'
python -m unittest src.test.tests.test_onnx_import -v
```

Source inventory, compiler errors, missing artifacts, timeouts, external-data
rejection, and output preservation are covered by focused tests. The real
compiler test checks a dynamic tensor shape in imported Sigmoid IR. It does
not claim numerical execution or board validation.
