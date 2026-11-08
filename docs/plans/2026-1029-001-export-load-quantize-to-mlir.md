# Export front end: load, quantize, to_mlir

| Plan ID | Topic | Revision | Status |
|---|---|---|---|
| 2026-1029-001 | export-load-quantize-to-mlir | rev.3 (2026-10-29) | Implemented — see [11. Implementation notes](#11-implementation-notes) |

Issue: #119 (software stack). Branch: `npu/issue119-a`.

Goal: replace each model frontend's single `build()` with three explicit
stages, so that reading a source, choosing numbers, and emitting MLIR are
separate, testable, and individually runnable:

```
load(source)            -> SourceModel      read and verify; no numeric change
quantize(src, recipe)   -> QuantizedModel   every array in its final numeric form
to_mlir(q, shape)       -> ModelGraph       MLIR + weights; no file I/O, no quantization
```

`ModelGraph` is unchanged, so everything after it (partition, weight packing,
NPU encode, host codegen, manifest) keeps working as it does today.

---

## 1. Scope

In scope:

- `src/export/` front end: `graphs/`, `core/`, `quant/`, the quantization
  re-export in `compilers/mlir/ops/linear.py`, and the front-end half of
  `cli.export()`.
- `src/test/`: a golden snapshot test plus per-stage tests; existing tests are
  updated for moved imports with unchanged assertions.

Out of scope (follow-up plans):

- **Stage 4 compile and stage 5 package.** `cli.py:112-196` (partition, pack,
  encode, host codegen, manifest) stays as it is, behind a renamed
  `compile_graph(graph, out, ...)` that the new pipeline calls. `compilers/`
  stays where it is because stage 4 also uses `compilers/mlir/toolchain.py`.
- **`examples/`.** Kept working through a compatibility shim (section 6);
  migrated in a separate issue that also removes the shim.
- **`src/jit/`.** Self-contained by rule (`test_jit_boundary.py`); not touched.
- **`src/test/model/`.** The numeric contract does not change. Quantization
  code moves; its math does not.
- **The package format.** `npu-package/1`, `manifest.json` and the runtime
  stay unchanged, apart from the new timing keys in section 5.

## 2. Problems this solves

1. One `build()` does three jobs. In `graphs/smollm/model.py` and
   `graphs/smollm/onnx.py` file reading, quantization and graph construction
   are interleaved, so none of them can be tested or rerun alone.
2. Quantization lives under MLIR. `compilers/mlir/ops/linear.py` re-exports
   `quantize_weight`, and frontends and tests call `L.quantize_weight`.
   `core/weight_handler.py` is quantization code under `core/`.
3. Source-format detection is duplicated in `cli.py:76-81` and
   `smollm.build()`.
4. `LLMOptions` mixes quantization choices (bits, group, AWQ) with shape
   choices (chunk, max_seq, attention), and `build()` mutates `opts.group`.
5. Each frontend repeats its config parsing, per-layer weight gathering,
   fusion, and quantization calls.

## 3. Target layout

```
src/export/
  ir/
    graph.py          ModelGraph, StateSpec, build_prefill_decode_graphs, new_graph   (from core/graph.py)
    options.py        QuantRecipe, ShapeOptions, LLMOptions (compat facade)          (from core/graph.py)
    source.py         SourceModel, TensorSource
    quantized.py      QuantizedModel, QTensor (QuantizedWeight | OnnxQuantWeight | np.ndarray)
  load/
    __init__.py       load(model, source, format='auto', layers=None) -> SourceModel
    detect.py         the only source-format and architecture detection
    hf.py             resolve_hf                                                     (from core/model_loader.py)
    safetensors.py    SafeTensors                                                    (from core/model_loader.py)
    onnx.py           OnnxWeights, pin check, onnx-mlir import, topology checks      (from model_loader, smollm/onnx.py)
    onnx_prepare.py   fetch / export_graph / verify_export / export_operators        (from core/onnx_source.py)
    torch.py          ResNet .pth state dict                                         (from resnet18/model.py)
  quantize/
    __init__.py       quantize(src, recipe) -> QuantizedModel
    recipe.py         resolve(recipe, cfg): choose_group, validation
    int8.py           QuantizedWeight, quantize_weight, EPS                          (from quant/int8.py)
    plan.py           WeightSpec, apply a weight plan (gather, fuse, transform)      (from core/weight_handler.py)
    awq.py            FoldSpec, hook_for, calibration_ids                            (from quant/awq.py)
    preserve.py       ONNX: keep source quantization, regroup, check fusion legality
    fakequant.py, evaluate.py                                                        (from quant/)
  model_arch/
    __init__.py       ARCHS registry, get(name)
    base.py           Architecture protocol
    smollm.py  smollm_onnx.py  qwen3.py  qwen3_moe.py  qwen3_5.py  resnet18.py
  sources/
    smollm/onnx-source.json                                                          (from graphs/smollm/)
    resnet18/*-source.json, download_*.py, requirements.txt                          (from graphs/resnet18/)
  compilers/          unchanged location; ops/linear.py stops re-exporting quantization
  pipeline.py         export() = compile_graph(to_mlir(quantize(load(...))))
  cli.py              argument parsing; export plus load / quantize / lower subcommands
  graphs/__init__.py  temporary shim: load(name).build(...) -> pipeline (section 6)
```

Removed at the end: `core/`, `quant/`, `graphs/<model>/`.

### Import direction

Imports point one way, enforced by `src/test/tests/test_export_layering.py`
(an AST scan in the style of `test_vendor_boundary.py`):

```
ir  <-  load  <-  quantize  <-  model_arch  ->  compilers/mlir, compilers/onnx_mlir
                                     ^
                               pipeline, cli
```

- `ir` imports nothing above it (only `compilers/mlir/builder` for types).
- `load` imports only `ir`.
- `quantize` imports `ir` and `load` types, never `compilers`.
- `compilers/mlir/ops` imports `ir.quantized` types, never `quantize`.
- `model_arch/<model>.py` modules never import each other, only `model_arch/base.py`.
  This keeps today's "models never import each other" rule.

## 4. Stage contracts

### 4.1 Data types

```python
# ir/source.py
@dataclass
class SourceModel:
    arch: str                    # 'smollm' | 'smollm_onnx' | 'qwen3' | ...
    format: str                  # 'safetensors' | 'onnx' | 'torch'
    source: str                  # as given by the caller; goes to meta['source']
    root: Path                   # directory holding config and tokenizer
    config: dict                 # raw config.json (or {} for ResNet)
    tensors: TensorSource        # lazy name -> np.ndarray; SafeTensors / OnnxWeights / dict
    files: dict[str, Path]       # copied verbatim into the package
    layers: int | None           # requested layer limit (debug)
    provenance: dict             # sha256, onnx import report and nodes, pin

# ir/options.py
@dataclass
class QuantRecipe:
    bits: int = 8                # 8, 4, 32 (float reference); 'preserve' recipes ignore it
    group: int = 128
    embed_bits: int = 8
    awq: bool = False
    awq_samples: list[str] = field(default_factory=list)
    mode: str = 'symmetric'      # 'symmetric' | 'preserve' (source already quantized)

@dataclass
class ShapeOptions:
    chunk: int = 16
    max_seq: int = 512
    attention: str = 'online'
    kv: str = 'f32'
    prefill_lengths: tuple[int, ...] = (36, 40, 42)
    limits: isa.Limits = field(default_factory=isa.Limits)

# ir/quantized.py
@dataclass
class QuantizedModel:
    source: SourceModel
    recipe: QuantRecipe          # resolved: group already chosen
    config: Any                  # the architecture's parsed Config, layers already clipped
    params: dict[str, QTensor]   # flat names, e.g. 'embed', 'l3.qkv', 'l3.ln1', 'norm'
    meta: dict                   # facts quantize learned (fusions, awq, onnx digests)

    def save(self, path: Path) -> None     # params.npz + model.json
    @classmethod
    def load(cls, path: Path) -> 'QuantizedModel'
```

`LLMOptions` stays as a facade with an unchanged constructor, plus a
`split() -> (QuantRecipe, ShapeOptions, layers)` method. `describe()` must
return exactly what it returns today, including the resolved `group`,
because `meta['options']` is part of the golden snapshot.

### 4.2 Architecture protocol

```python
# model_arch/base.py
class Architecture(Protocol):
    name: str
    formats: tuple[str, ...]                                 # accepted source formats

    def parse_config(self, src: SourceModel) -> Config
    def group_dims(self, cfg) -> list[int]                   # K dims choose_group must divide
    def weight_plan(self, cfg, recipe) -> list[WeightSpec]   # what quantize produces
    def awq_spec(self, cfg) -> FoldSpec | None
    def to_mlir(self, q: QuantizedModel, shape: ShapeOptions) -> ModelGraph
```

```python
# quantize/plan.py
@dataclass
class WeightSpec:
    name: str                    # param name, e.g. 'l3.qkv'
    sources: list[str]           # source tensor names, concatenated in order
    kind: str                    # 'linear' | 'embed' | 'float' | 'raw'
    axis: int = 0                # concat axis (torch [out, in] layout)
    transform: Callable | None = None   # e.g. Qwen3.5 q/gate interleave, -exp(A_log)
    layer: int | None = None     # groups specs for the AWQ hook
```

`weight_plan` is pure data about names and fusion. Quantization, AWQ, and
dtype conversion happen only in `quantize/`.

### 4.3 Stage responsibilities

**load(model, source, format='auto', layers=None) -> SourceModel**

- `detect.py` resolves `format` from the path: `.onnx`, `model_int8.onnx`,
  `onnx_asset_dir(..)/model_int8.onnx`, `.pth`, safetensors. Then it maps
  `(model, format)` to an architecture, e.g. `('smollm', 'onnx')` maps to
  `smollm_onnx`. This replaces `cli.py:76-81` and the detection in
  `smollm.build()`.
- safetensors: `resolve_hf`, `config.json`, lazy `SafeTensors`, tokenizer files.
- onnx (`smollm_onnx`): everything in `onnx.py:438-473` that inspects without
  computing. That means the sha256 pin check, `adapter.load_imported`, the
  imported node table, the 210 `MatMulInteger` count, zero weight zero
  points, and the fusion-shares-activation-quantization check. The node table
  and import report go into `provenance`.
- torch: `_find_source` and the state dict for ResNet-18.
- Rejects an unsupported `(model, format)` pair with the message the current
  code raises.

**quantize(src, recipe) -> QuantizedModel**

- `cfg = arch.parse_config(src)`; clip `cfg.layers` when `src.layers` is set.
- `recipe.group = choose_group(recipe.group, arch.group_dims(cfg))`.
- AWQ: when `recipe.awq`, `awq.hook_for(src.root, ...)` with `arch.awq_spec(cfg)`.
- Walk `arch.weight_plan(cfg, recipe)`: gather and concatenate sources, apply
  `transform`, then by `kind`:
  - `linear`: AWQ hook per layer, then `quantize_weight(bits, group)`
  - `embed`: `quantize_weight(embed_bits, group)`
  - `float`: `astype(float32)`
  - `raw`: unchanged
- `mode='preserve'` (`smollm_onnx`): `quantize/preserve.py` builds the fused
  `(int8 matrix, per-column scales)` groups and the dequantized head from
  `onnx.py:480-495`. It never requantizes. It rejects AWQ, `bits != 8`, and
  any other recipe the ONNX contract forbids, with today's error text.

**to_mlir(q, shape) -> ModelGraph**

- `arch.to_mlir` builds the graph from `q.params`, `q.config` and `shape`
  only. It reads no files and quantizes nothing.
- Shape-dependent constants are computed here, e.g. `rope_tables(max_seq, ..)`,
  because they depend on `ShapeOptions`, not on the recipe.
- `smollm_onnx` reads its imported nodes from `q.source.provenance`, and
  checks `kv == 'f32'` and `max_seq == 512` against `ShapeOptions`.
- Sets `meta` and `files` exactly as the current `build()` does.

## 5. Pipeline and CLI

```python
# pipeline.py
def export(graph_or_source, out, targets=None, *, model=None, opts=None, ...):
    if isinstance(graph_or_source, ModelGraph):          # existing graph callers
        return compile_graph(graph_or_source, out, targets, ...)
    recipe, shape, layers = (opts or default_options(model)).split()
    t = time.monotonic(); src = load(model, graph_or_source, source_format, layers);  timings['load'] = ...
    t = time.monotonic(); q = quantize(src, recipe);                                  timings['quantize'] = ...
    t = time.monotonic(); graph = to_mlir(q, shape);                                  timings['to_mlir'] = ...
    return compile_graph(graph, out, targets, ..., timings=timings)
```

- `cli.export` keeps its signature and becomes a re-export of
  `pipeline.export`. Its four callers in `src/` and `examples/` keep working.
- `compile_seconds` gains `load`, `quantize` and `to_mlir`, and keeps
  `frontend` as their sum. No code reads `frontend` today, but recorded board
  evidence (`examples/smollm-safetensors/docs/board-run/data/summary.json`)
  carries it, so later runs stay comparable.
- New subcommands, for debugging and caching:

```
python -m src.export.cli load     smollm --source <dir> --out build/stage/source.json
python -m src.export.cli quantize build/stage/source.json --bits 4 --out build/stage/q
python -m src.export.cli lower    build/stage/q --chunk 16 --out build/stage/model.mlir
python -m src.export.cli export   smollm --out build/pkg/smollm ...      # unchanged
```

  `source.json` records the `load` arguments, not the tensors: the stage is
  replayed, so no weights are copied. `q/` is `QuantizedModel.save`.

## 6. Compatibility

| Caller | Before | After |
|---|---|---|
| `cli.export(...)`, `python -m src.export.cli export` | unchanged | unchanged |
| `src.export.graphs.load(name).build(...)` | frontend module | shim object whose `build()` runs load, quantize, to_mlir and returns `(graph, cfg)` |
| `src.export.graphs.load('resnet18').ResNetOptions` | class | re-exported from `model_arch/resnet18.py` |
| `L.quantize_weight`, `L.QuantizedWeight` | MLIR ops | removed; use `src.export.quantize.int8` (tests updated) |
| `src.export.core.graph.*` | | removed after step 9; imports updated in `src/` and `src/test/` |
| `.github/cd/smollm.py` pin path | `graphs/smollm/onnx-source.json` | `sources/smollm/onnx-source.json` |

The shim and the `examples/` notebooks that need it are removed together in the
examples follow-up.

## 7. Implementation order

Each step is one commit. Every step keeps `make -C src/test lint sim` and
`python -m unittest discover src/test/tests` green, and reproduces the step 0
golden bit for bit.

| # | Commit | Notes |
|---|---|---|
| 0 | `test(export): golden snapshot of every frontend` | See section 8. Runs against the current `build()` before anything moves. |
| 1 | `refactor(export): move graph types to ir/` | `core/graph.py` becomes `ir/graph.py` and `ir/options.py`; add `QuantRecipe`, `ShapeOptions`, `LLMOptions.split()`; temporary re-export in `core/graph.py`. |
| 2 | `refactor(export): move quantization to quantize/` | `quant/*` and `weight_handler` move. Drop the re-export in `ops/linear.py`; ops import `QuantizedWeight` from `ir.quantized`; update `test_compiler`, `test_ops`, frontends. |
| 3 | `refactor(export): move source readers to load/` | `model_loader`, `onnx_source`, `resolve_hf`; add `detect.py`; `cli.py` uses it. |
| 4 | `feat(export): add SourceModel, QuantizedModel and the Architecture protocol` | `ir/source.py`, `ir/quantized.py`, `model_arch/base.py`, `quantize/plan.py`, empty registry. |
| 5 | `refactor(export): split SmolLM safetensors into load, quantize, to_mlir` | `model_arch/smollm.py`; `graphs/smollm/model.py:build` becomes a wrapper. |
| 6 | `refactor(export): split SmolLM ONNX into load, preserve, to_mlir` | `load/onnx.py` checks, `quantize/preserve.py`, `model_arch/smollm_onnx.py`; rewrite `test_auto_onnx_source_uses_common_model_builder` against the stages. |
| 7 | `refactor(export): move Qwen3, Qwen3-MoE, Qwen3.5, ResNet-18 to model_arch/` | One commit per model is fine. Qwen3.5 uses `WeightSpec.transform` for the q/gate interleave and `-exp(A_log)`; MoE plans one `linear` spec per expert. |
| 8 | `feat(export): pipeline with per-stage timings and stage subcommands` | `pipeline.py`; `cli.export` re-exports it; `compile_graph` extracted unchanged; `QuantizedModel.save/load`. |
| 9 | `refactor(export): remove core/, quant/ and per-model graphs/` | Move pins to `sources/`; update `.github/cd/smollm.py`, `test_resnet18_sources.py`, docs outside `docs/human/`; leave the `graphs/__init__.py` shim. |
| 10 | `test(export): enforce stage import direction` | `test_export_layering.py`. |

Suggested pull requests, each merged into `dev`: steps 0-4 (moves, no behavior
change), steps 5-7 (architectures), steps 8-10 (pipeline, cleanup, layering).

Optional, after step 7, only if the golden stays bit-exact: extract the
decoder body that `model_arch/smollm.py` and `model_arch/qwen3.py` share (they differ by
q/k head norms) into `model_arch/base.py` helpers. It is not required for this plan.

## 8. Tests

| Test | Checks |
|---|---|
| `test_export_golden.py` (new) | For each architecture, a tiny seeded synthetic checkpoint (written like `SharedExportEngineTest`) goes through `build()` / the pipeline. sha256 of `module.render()`, of each weight (name, dtype, shape, bytes), of `meta` and of `entry_points` must equal `src/test/vectors/export_golden.json`. Recipes: W8, W4 g8, FP32 reference. `smollm_onnx` runs only when the pinned ONNX and the onnx-mlir import are present, and AWQ only when torch and transformers are installed; otherwise those cases skip. |
| `src/test/vectors/generate_export_golden.py` (new) | Writes the golden file; `--check` mode like `generate_isa_vectors.py`. Run once at step 0, never regenerated during the refactor. |
| `test_load.py` (new) | Format detection for every path form; unsupported pairs; pin mismatch; ONNX topology, zero-point and fusion rejections; `layers` clipping. |
| `test_quantize.py` (new) | `choose_group` via `group_dims`; INT8/INT4/FP32 kinds; AWQ hook called once per layer with the planned matrices; `preserve` rejects AWQ, `bits != 8`, illegal fusion; `QuantizedModel` save/load round trip is bit-exact. |
| `test_pipeline.py` (new) | load, quantize and lower as subcommands produce the same `model.mlir` and weights as `export`; `compile_seconds` has `load`, `quantize`, `to_mlir`, `frontend`. |
| `test_export_layering.py` (new) | Section 3 import directions. |
| Existing (`test_compiler`, `test_ops`, `test_onnx_*`, `test_resnet18_*`, `test_numeric`) | Imports updated, assertions unchanged. |

## 9. Acceptance

1. `generate_export_golden.py --check` passes after every step; the final tree
   produces the step 0 hashes for every architecture that ran.
2. `make -C src/test lint sim` and `python -m unittest discover src/test/tests`
   pass; skipped cases are listed in the pull request with the reason.
3. `python -m src.export.cli export smollm --out build/pkg/smollm` and the
   `--format onnx` variant produce byte-identical `weights.bin`, `programs/`
   and `manifest.json` (except `compile_seconds`) compared with `dev`.
4. No module under `model_arch/` reads a file; no module under `compilers/` imports
   `quantize/`; `test_export_layering.py` enforces both.
5. `examples/` notebooks still run through the shim. If not run in this
   change, the pull request says so.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Moving quantization changes floating-point order (e.g. AWQ before or after concat) | Golden snapshot from step 0; `weight_plan` keeps today's concat-then-quantize order. |
| `opts.group` mutation shows up in `meta['options']` | `LLMOptions.split()` + `describe()` reproduce the resolved group; covered by the golden `meta` hash. |
| ONNX golden cannot run without the pinned asset | Skip with a reason; acceptance item 3 is run locally with the asset and recorded in the PR. |
| Large rename churn breaks `examples/` | Shim in `graphs/__init__.py`; `cli.export` signature unchanged. |
| Docs reference old paths | Update `docs/` (not `docs/human/`) in step 9; `docs/human/` needs separate human confirmation. |

## 11. Implementation notes

Implemented on `npu/issue119-a` in commits `test(export): golden snapshot of
every frontend` through `test(export): enforce stage import direction`. Every
step reproduced the step 0 golden digests bit for bit, including the SmolLM
ONNX case (run locally with the pinned `model_int8.onnx` and a verified
import). Where the code differs from the sections above:

- **Stage entry points live in `src/export/pipeline.py`** (`load`, `quantize`,
  `to_mlir`, `build_graph`), not in `load/__init__.py` and
  `quantize/__init__.py`. Those packages hold generic machinery only; the
  per-model choices are methods of `model_arch/base.py:Architecture`
  (`load`, `config`, `weight_plan`, `quantize`, `to_mlir`, `validate`). This
  keeps `load/` free of model knowledge.
- **`smollm_onnx` is its own architecture.** `model_arch.resolve` maps
  `('smollm', 'onnx')` to it. Its SmolLM-specific topology checks (MatMulInteger
  count, zero points, fusion legality) run in its `load`; `load/onnx.py` holds
  only the generic pin check and ONNX-MLIR import.
- **`WeightSpec` kinds** are `linear`, `embed`, `fp32` (the MoE router),
  `float` and `raw`. `raw` keeps the dtype the source stores, which is what
  Qwen3-MoE and Qwen3.5 did before. The plan executor reads each source tensor
  once per layer, so stacked MoE expert tensors are not re-read per expert.
- **`cli.export()` stays in `cli.py`** and calls `pipeline.build_graph`, then
  `compile_graph()` (the unchanged stage 4 code). Moving it into
  `pipeline.py` waits for the stage 4 plan.
- **`graphs/` and `core/` are gone.** The ResNet download scripts and pins and
  the SmolLM ONNX pin moved to `src/export/sources/<model>/`; `examples/`,
  `.github/` and the tests run them from there. The SmolLM board guides copy
  `src/export/load/safetensors.py` instead of `core/model_loader.py`. There is
  no compatibility shim: callers use `src.export.pipeline`.
- **`QuantizedWeight` and `EPS` moved to `ir/quantized.py`** so the MLIR
  operators no longer import the quantizer. The MoE router reaches
  `ops/moe.py` as an FP32 `QuantizedWeight` prepared by the plan.
- **One deliberate numeric change:** the tied embedding is quantized with the
  resolved group size, like every other INT4 matrix. Before, SmolLM and Qwen3
  used the requested size, so `--embed-bits 4` failed whenever that size did
  not divide the hidden width. INT8 embeddings, the default, are unchanged.
- **Auto-detected formats fall back** to an architecture's only format, so
  ResNet-18 still reads a checkpoint whatever its file name.

Not migrated: the recorded experiment scripts under
`exp/.docs/goals/jit-independent-and-aot-support-resent18/`, which import
`src.export.graphs.resnet18.model`, `src.export.core.graph` and
`src.export.compilers.mlir.ops.linear.quantize_weight` and no longer run
unchanged, and historical records (docs/goals, openspec) that name the old
paths. The examples were updated for the new paths but not rerun on the board.

### Simplification (rev.3)

A follow-up pass, with the shared decoder explicitly left out:

- **Model directories.** A model that carries more than its architecture is a
  directory: `model_arch/smollm/` (`arch.py`, `onnx.py`, `onnx-source.json`)
  and `model_arch/resnet18/` (`arch.py`, `download.py`, the three
  `*-source.json` pins, `requirements.txt`). `src/export/sources/` is gone.
  Qwen3, Qwen3-MoE and Qwen3.5 stay single modules. A model may import its own
  directory; `test_export_layering.py` still forbids imports between models.
  The SmolLM ONNX path reuses `smollm/arch.py:Config` instead of a copy.
- **One ResNet downloader.** `download_model.py`, `download_demo_assets.py`
  and `download_gallery.py` became `resnet18/download.py` with shared
  canonical-record, URL, retry and publish code:
  `python -m src.export.model_arch.resnet18.download {model,demo,gallery}`.
  All three now retry transient HTTP failures and use one exception,
  `SourceError`.
- **`cli.py` split.** `compile_graph()` (ModelGraph -> package) moved to
  `src/export/compile.py`, which no longer imports the front end; `cli.py`
  keeps `export()` (front end, then `compile_graph`) and argument parsing.
- **ONNX-only operators moved.** `native`, `projection` and `multiply` moved
  from `compilers/mlir/ops/linear.py` to `compilers/onnx_mlir/lowering.py`;
  the unused `linear.quantize` was deleted.
- **Deleted or folded.** `quantize/evaluate.py` and `quantize/fakequant.py`
  (no user in `src/`; their only callers import the removed `src.inference`
  package). `load/hf.py` folded into `load/safetensors.py`, `load/detect.py`
  into `load/__init__.py`, and `ir/source.py` with `ir/quantized.py` into
  `ir/data.py`.

