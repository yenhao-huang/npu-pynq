# SmolLM2 board acceptance results

Status: **PASS — physical PYNQ-Z1 acceptance completed on 2026-10-09**.

The complete 30-layer SmolLM2-135M-Instruct W8A8 + AWQ model loads and generates
24 tokens on the physical FPGA/CPU stack. All five representative GEMMs match
the integer reference exactly. The full trace matches the identical package's
CPU ISA simulator with zero logit error, and reset/repeat produces identical
token IDs and logits. The final notebook executes all seven code cells without
errors and contains three metric tables and four plots.

Final evidence: [executed notebook](runs/board-20261009-04/smollm_board.executed.ipynb),
[physical verdict](runs/board-20261009-04/verdict.json),
[independent saved-evidence audit](runs/board-20261009-04/audit.json).

## Direct Jupyter reopening correction

The original run 02 passed through the automated runner, which injected
`NPU_DEPLOY` and `NPU_REPO`. The user then opened the saved notebook under
`runs/board-20261009-02/` directly in Jupyter and reported `ModuleNotFoundError:
No module named 'acceptance'`. The initialization wrongly assumed the kernel cwd
was the deployment root. The original success did not validate this interactive
entry path.

The first cell now finds the scripts/package markers in the current directory
and its ancestors. Run 03 proved the import repair, then failed the monolithic
large GEMM probe with `BO allocation failed: std::bad_alloc`. The probe now uses
the model's 4 MiB column streaming window while still checking all 49,152 outputs
and requiring the full 9,216 jobs for `1x49152x576`.

Run 04 removes both path environment variables and starts the kernel in
`runs/board-20261009-04/`. All seven code cells complete with no error output,
all five GEMMs pass, and full generation/reference/repeat checks pass. The run
log and independent audit explicitly record the nested-cwd validation. Original
run 02 and failed run 03 remain preserved as historical evidence.

Interactive entry: open `exp/1007_sw_stack/deploy/smollm_board.ipynb` on the board
and execute its cells in order. The corrected source notebook is already staged.

## Target and source

SmolLM2-135M-Instruct, W8A8 + AWQ, PYNQ-Z1, source baseline `e24397a8`.
Plan: [Plan.md](Plan.md).

## Gate evidence

| Gate | Status | Evidence |
| --- | --- | --- |
| SSH and resource preflight | Passed; sudo authenticated | [board.json](runs/preflight/board.json): Python 3.10.4, PYNQ 3.1.1, NumPy 1.21.5; ARMv7/glibc 2.35. User authorized sudo and loading `PYNQ_PASSWORD` from workspace `.env`; stdin authentication returned UID 0. Password was not printed or saved. |
| Artifact compatibility | Passed | [provenance.json](runs/artifact-verification/provenance.json): workflow run 37579935556; local BIT/HWH hashes match the manifest; no source differences in hardware/compiler/runtime/models/inference from overlay source `fcb84b55` to baseline. Setup WNS +0.031 ns, hold slack +0.046 ns, no failing endpoints. Board hashes match transferred files. |
| ARM ABI and loader | Passed on physical board | ELF32 little-endian ARM, hard-float flags `0x5000400`; libm/libgcc dependencies and model library load; `_mlir_ciface_prefill` and `_mlir_ciface_decode` exist. This does not prove inference. |
| ISA simulator tests | Passed | [isa.log](runs/host-tests/isa.log): 8 tests pass, including tiled GEMM, column segmentation, faults, and RTL opcode/register contracts. |
| Framework tests | Passed on physical board | [operators.log](runs/board-python-tests/operators.log): safetensors, sampling and tokenizer pass. Three compiled operator tests skip because LLVM is absent on the board. |
| Compiler/operator tests | Passed for the exact baseline on Mac | [baseline-compiler.log](runs/mac-tests/baseline-compiler.log): 4/4 pass; [baseline-operators.log](runs/mac-tests/baseline-operators.log): 6/6 pass, no skips. Tested a local archive of commit `e24397a8` already present in Mac's Git object database; no Windows source/model transfer. |
| Same-package CPU reference | Passed | [provenance.json](data/reference/provenance.json): unprivileged ARM CPU ISA simulator, backend 0, null MMIO, identical model library and weights; full 24-token trace saved in `data/reference/`. |
| Physical ISA/GEMM | Passed | [gemm.json](runs/board-20261009-04/gemm.json): all 5 shapes exact, jobs/cycles positive; version `0x10001`, capabilities `0x7b`. Includes `16x1536x576` prefill contraction and `1x49152x576` lm_head. |
| Sequential SmolLM notebook | Passed | [executed notebook](runs/board-20261009-04/smollm_board.executed.ipynb): all 7 code cells executed in order, no error outputs, 3 tables and 4 plots. Source [smollm_board.ipynb](smollm_board.ipynb) remains output-free. |
| Full 24-token generation and comparison | Passed | [generation.json](runs/board-20261009-04/generation.json): 24 tokens, no EOS or runtime error, prefill 4.617 s, 23 decode calls in 15.934 s (1.443 tokens/s), 957,168 physical decode jobs and 450,169,679 cycles. [comparison.json](runs/board-20261009-04/comparison.json): IDs equal and maximum/mean logit error 0 at the original tolerances. Prefill logits also match. |
| Repeatability | Passed | [repeat.json](runs/board-20261009-04/repeat.json) and [repeat-comparison.json](runs/board-20261009-04/repeat-comparison.json): 24 tokens, no runtime error, IDs equal, maximum/mean logit error 0. |

## Measured performance

| Metric | Final run 04 |
| --- | --- |
| Full model load | 19.268 s |
| Prompt length | 19 tokens |
| Prefill | 4.617 s |
| Generated samples | 24 tokens |
| Decode calls | 23 |
| Decode total | 15.934 s |
| Decode throughput | 1.443 tokens/s |
| Process RSS after generation | 252.5 MiB |
| Process high-water RSS | 296.5 MiB |
| Free CMA after generation | 13.5 MiB |

Prompt: `Explain why the sky is blue in simple words.`

Generated text:

> The sky is blue because the light from the sun reaches our eyes and is scattered by the air molecules in the sky.

Plots: [latency and timing](runs/board-20261009-04/figures/figure-2.png),
[memory snapshots](runs/board-20261009-04/figures/figure-3.png),
[logit comparison](runs/board-20261009-04/figures/figure-4.png).

## Execution details

- Native ISA polling has no built-in timeout; the executed notebook used a 900-second cell timeout and 5400-second external process deadline. Neither deadline was reached in the accepted run.
- Streaming weights were enabled and measured. The complete weights stay memory-mapped; the actual CMA allocation succeeds for I/O, stage and programs.
- After transfer, free CMA briefly fell to 0.8–3 MiB because free-page snapshots include cache effects. Physical execution reclaimed pages and the accepted run completed without allocation failure.
- Board wall clock reports May 2025. No clock configuration was changed; use a host-supplied run ID and monotonic durations.

## Package and storage evidence

- Complete downloaded ARM package: `packages/smollm/`; original workflow download: `downloads/export-009/`.
- Matched overlay: `hardware/overlay/`; implementation reports: `hardware/reports/`.
- Weights: 135,570,176 bytes; SHA-256 `94af3b5511b1f457dc3a15c9ba1b4a7ea3b3940d32b222573ffa09a7398a85aa`.
- ARM library SHA-256: `206dfe9501484012f5e5c0debd3664a0241bc31c9501611e5bd5326ef571990e`.
- Streaming CMA requirements from programs: I/O 599,040 bytes, stage 4,193,280 bytes, ISA words 54,784 bytes. KV state uses 23,592,960 bytes of normal memory.
- The fixed chat prompt encodes to 19 tokens. Tokenizer round-trip passes with `skip_special=False`.
- Remote repository root: `/home/xilinx/jupyter_notebooks/smollm_validation_20261009/`; package, notebook, source and overlay are staged beneath its `exp/1007_sw_stack/deploy/`.

## Experiment decisions

| Hypothesis / command | Observed result | Decision |
| --- | --- | --- |
| Reuse workflow export + overlay from run 37579935556 | Downloads complete; source compatibility and hashes pass | Use these artifacts for physical acceptance. |
| Reuse Mac's `build/pkg/smollm` directly | Mac weights SHA starts `89a2196b`; downloaded AWQ weights start `94af3b55` | Reject it as a direct reference; require the identical weight blob and compatible host library. |
| Run framework tests on Windows, then board | Windows cleanup fails for mapped safetensors; the actual board passes all three framework tests | Keep platform-specific failure visible; run compiled tests on an LLVM host. |
| Load ARM library as unprivileged xilinx | ELF/loader/dependency/symbol checks pass | Proceed to hardware only after authorized root execution. |
| Authenticate sudo using the user-specified `.env` variable | UID 0 confirmed; password sent only over stdin | Run the bounded notebook through the existing PYNQ environment. |
| First physical notebook run | GEMM, load, smoke and 24-token generation pass; comparison stops because reference files are not ready | Preserve run 01, produce the CPU simulator reference, then rerun the complete notebook. |
| Same-library CPU ISA simulation | Backend 0, null MMIO, full trace produced without root; tokens and logits match the FPGA trace | Use the exact package as the arithmetic reference. |
| Full notebook run 04 and saved-evidence audit | All gates pass; repeat is exact; seven code cells execute with no errors | Acceptance complete. |

## Authentication and reference resolution

The initial run was blocked by root authentication. The user subsequently authorized
sudo development and instructed loading `PYNQ_PASSWORD` from `.env`; this works through
stdin without credential disclosure. The original source/model transfer to Mac was
rejected by automatic approval review and was not retried or routed indirectly.
Instead, the identical already-deployed ARM library runs the CPU ISA reference on
the board without root or MMIO. Compiler/operator tests use the requested baseline
already present in Mac's local Git objects; no model or Windows source is transferred.

The repeatable notebook invocation is documented in [README.md](README.md).
Run `python -X utf8 exp/1007_sw_stack/deploy/scripts/audit.py` from the repository
root to independently recheck the saved evidence, package identity, full model,
numeric traces, physical counters, compiler/operator logs and notebook execution.
All generated packages, overlays, raw arrays and executed notebooks remain ignored
under the requested deployment directory; source scripts and documentation are
kept separately. No runtime/compiler/RTL source change was needed for this result.

## Verdict

Every acceptance requirement in [Plan.md](Plan.md) has direct saved evidence and
passes the independent audit. This establishes execution and numeric agreement
for the fixed prompt and representative GEMMs recorded above. No performance
floor was required or introduced.
