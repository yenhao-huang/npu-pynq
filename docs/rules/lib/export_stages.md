# Export stages: front end, middle end, back end

`src/export/` has three stages. Each step belongs to exactly one; the IR it
consumes and produces decides which.

## Guiding principle: follow IREE

When it is unclear where a step belongs, put it where IREE puts the
equivalent step. A deviation needs a written reason in the plan or PR that
introduces it.

| Our stage | IREE phases (`iree-compile --compile-to=`) | Shared idea |
| --- | --- | --- |
| Front end | importers ahead of `input` (torch-mlir's FX importer) | framework graph → MLIR |
| Middle end | `input`, `flow`, `stream` | tensor-level, target-agnostic: fusion, kernel (dispatch) formation, device affinity and partitioning |
| Back end | `executable-sources`, `executable-targets`, `hal`, `vm` | per-target code generation (bufferization happens here) and the runtime interface |

Known deviation: our schedule stays compiled code in `model.so` instead of
IREE's VM bytecode (plan 2026-1112-004).

| Stage | Path | In → out | Owns | Never |
| --- | --- | --- | --- | --- |
| Front end | `frontend/` (Python) | model → `ir/torch.mlir` | loading, quantization, `torch.export`, import to the torch dialect | MLIR rewrites; anything device- or target-specific |
| Middle end | `middle-end/` (`pynqo-middle-end`) | torch dialect → linalg on tensors: the CPU graph (`ir/model.mlir`) and one module per device (`ir/delegates/<device>.mlir`) | graph-level work: torch rewrites, lowering to linalg, device assignment, partitioning (each device subgraph cut out into its device's module, a bodiless `@__delegate_<id>` called in its place), fusion of the CPU graph | memrefs, loops, LLVM; device code generation; anything that depends on the CPU target |
| Back end | `backend/` (`pynqo-backend`) | the CPU graph and the device modules → package | code generation: each delegate compiles its own device module; per `--cpus` target, delegate calls to `__delegate__(id, in, out)`, bufferization, loops, LLVM, `llc`, link | deciding which device runs an op; graph rewrites that would give every target the same result; CPU code that depends on how a device computes |

## Rules

- **Whole-graph decisions are middle end; per-device compilation is back
  end.** Partitioning and replacing a subgraph with a delegate call need the
  whole graph: where the cuts go, which neighbours share a device, one global
  delegate id per subgraph, and every later graph pass (fusion) must respect
  the cuts. Only the middle end sees the whole graph and every device. A back
  end compiles one device's subgraphs and never needs the whole graph. This is
  the reason IREE (`stream` affinity before `hal`), TVM BYOC (`PartitionGraph`
  before codegen) and ExecuTorch (`to_backend` before `preprocess`) partition
  ahead of code generation.
- **Tensor level is the boundary.** A step on tensors that gives the same
  result for every target belongs to the middle end. Bufferization (it fixes
  the runtime ABI) and everything below it belongs to the back end.
- **The middle end may know which devices exist, never how they compile.** It
  links only a device's match predicate (`src/vendors/<vendor>/`), not its
  dialect lowering or encoder. Rewrites under `middle-end/torch/transforms/`
  stay device-neutral ([register_pass.md](register_pass.md)).
- **The back end decides nothing about placement.** It reads the delegate
  declarations, device modules and attributes the middle end wrote and keeps
  their delegate ids.
- **Devices are delegates.** As in ExecuTorch and LiteRT, a device is a black
  box to the rest of the compiler: the CPU graph sees only a call with the
  subgraph's argument and result, a device's back end compiles only its own
  module and contributes no CPU code, and whatever the device needs on the
  host (packing, summing partial results) lives in its runtime library. This
  deviates from IREE, which keeps device executables in the shared IR until
  `hal`; plan 2026-1113-001 gives the reason.
- **Stages talk through files only.** The front end writes `torch.mlir`, the
  middle end `model.mlir` and `delegates/<device>.mlir`; each is printed with
  locations kept. Only
  `pipeline.py` runs the stages and passes those files on.
- **A new step goes where its input is.** If it needs memrefs or a target, it
  is back end; otherwise it is middle end, even if it is about a device.

Plans `docs/plans/2026-1112-005-middle-end-backend-split.md` (#176, #177)
and `docs/plans/2026-1113-001-delegate-framework.md` (#177) moved fusion,
device assignment, partitioning and the delegate call from the back end to
the middle end.
