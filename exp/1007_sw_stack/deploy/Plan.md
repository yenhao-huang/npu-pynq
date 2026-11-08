# SmolLM2 deployment and acceptance plan

Target: PYNQ-Z1, SmolLM2-135M-Instruct, W8A8 + AWQ, compiled CPU/NPU inference.
Source baseline: `e24397a8f2ff1292e46e63693c5d2a6fa9a0d519` on `npu/issue119-a`.

## Storage

- `downloads/`: original workflow artifacts and extraction workspace.
- `packages/smollm/`: exported model libraries, tokenizer, ISA programs and weights.
- `hardware/overlay/`: matching BIT/HWH and provenance.
- `hardware/reports/`: Vivado implementation and timing evidence.
- `data/reference/`: host simulator outputs and logits for the same package.
- `runs/<run-id>/`: preflight, hashes, GEMM checks, model results, notebook output and logs.
- `scripts/`: repeatable acceptance helpers.
- `smollm_board.ipynb`: sequential board execution with visual progress and metrics.

Generated model, overlay, archives and numeric arrays stay local and are not committed.

## Execution gates

1. **Preflight:** check SSH, ARM architecture, PYNQ Python, libraries, available RAM/CMA, storage and current board activity. Record actual state before loading an overlay.
2. **Artifacts:** recover the successful export and overlay from workflow run `37579935556` if their source is compatible. Check hardware/compiler/runtime differences, paired BIT/HWH, ABI, model manifest, ELF architecture, file hashes and timing reports. Rebuild only incompatible or missing artifacts.
3. **Reference:** verify the ISA/compiler/operator tests actually execute. Use a suitable CPU library for the host: the downloaded ARM library already contains the CPU ISA simulator and can run on the board CPU without root or MMIO. Reuse that identical library through a manifest-only package view; build a different host target from the same IR only if needed. Verify backend 0 and null MMIO, preserve identical weights/layouts/kernels, and use the same fixed prompt and greedy trace. Preserve prompt IDs, generated IDs and logits. Record any missing tool or skipped gate.
4. **Hardware arithmetic:** load the verified ISA overlay and run integer GEMMs from a small tile through representative SmolLM shapes. Require exact equality with the integer reference, no hardware error, and nonzero hardware jobs/cycles. Apply bounded execution deadlines.
5. **Notebook inference:** execute the notebook on the board in order: setup, memory/manifest checks, overlay and model load, prefill, one-token smoke, 24-token generation, host comparison and repeatability. Show streamed text, timing/memory tables and charts, numerical comparison and hardware counters. Save JSON/NPZ evidence as each stage completes. Use an external watchdog for native runtime polling.
6. **Result:** update `Result.md` from observed evidence and exact commands. Diagnose and fix the failing layer, then rerun only affected gates. Finish only when the acceptance requirements below are proved.

## Acceptance requirements

- Full SmolLM2-135M model loads on the physical board.
- Prefill and 24-token greedy generation finish within the configured watchdog deadline without a runtime error.
- Real hardware jobs/cycles establish NPU execution.
- Representative integer GEMMs match exactly.
- Board results match the same quantized host package, or numerical differences have an evidenced explanation and an explicit assessment. Compare logits using predeclared `rtol=1e-3`, `atol=1e-3` as a diagnostic starting point, record the actual errors, and do not weaken thresholds after seeing a failure merely to pass.
- A repeat run proves no state or allocation corruption.
- The notebook executes sequentially on the board, provides visible results, and all artifacts and evidence reside under this deployment directory.

No performance floor is assumed: report measured load time, prefill/decode throughput and memory. Network configuration, board images and credentials are outside this experiment's scope.
