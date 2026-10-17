# PYNQ-Z1 runs

Status on 2026-10-09: **compiled official SmolLM2 INT8 physical acceptance
passed; the automated multi-model campaign remains incomplete**.

## Automated compiled-package workflow

GitHub run
[37508256815](https://github.com/yenhao-huang/npu-pynq/actions/runs/37508256815)
(request `006-models-on-board`) is completed with conclusion **cancelled**,
as checked on 2026-10-09. The earlier queued status is historical.

| Job | Runner | State |
| --- | --- | --- |
| export: SmolLM2 (W8A8 + AWQ), Qwen3-0.6B (W4A8 g64 + AWQ), ResNet-18 (W8A8) for the Cortex-A9 | GitHub-hosted, LLVM 22 from apt | PASS (packages 132 MB, 398 MB, 12 MB) |
| build-overlay: ISA front end, 16 x 16 array, 100 MHz | self-hosted Vivado | cancelled; routed build step did not run |
| board: ISA GEMM check, ResNet-18 gallery, SmolLM2 and Qwen3 chat | self-hosted, PYNQ-Z1 | cancelled; no completed board acceptance |

No physical PASS or routed timing is inferred from the successful export job.
Re-running the workflow must return completed board logs, `isa_gemm.json`,
`models.json` and routed timing evidence before its board gates can close.

## Historical official INT8 graph adapter on the physical board

The separate manual run `official-board-20261009-02` on PYNQ-Z1
(`192.168.2.99`) passed all eight notebook cells and 25 independent audit gates.
All 24 generated tokens match the unchanged full ARM ONNX oracle, with maximum
logit error `3.1948089599609375e-05`. CPU integer comparison and reset/repeat
are exact. The run executed 842,400 physical NPU jobs; TTFT was 41.22 s and
TPOT was 17.15 s/token.

This establishes physical execution for the official SmolLM2 INT8 graph adapter
and ISA runtime. Its floating-point operators use ORT; it does not execute the
MLIR/LLVM compiled-package path. The subsequent `src/` relocation passed host
source-only generation and regressions but still needs physical revalidation.
Qwen3 and compiled ResNet-18 acceptance remain open in this campaign.

Evidence, source ownership and remaining gates:
[official-int8.md](official-int8.md).

## Compiled official INT8 physical acceptance: PASS

Run `compiled-board-20261009-04` closes the manual official ONNX-to-MLIR/LLVM
SmolLM2 milestone: eight cells, zero errors, four figures, 32 interactive tokens
and 35/35 independent audit gates. Full unchanged ARM ONNX, separate compiled
ARM ISA-simulator reference and reset/repeat match every token and logit exactly.

Recorded TTFT is 12.860208 s, TPOT 1.543225 s/token; physical counters record
2,880 contractions, 842,400 jobs and 453,076,654 cycles. Native ORT float kernels
remain explicit compiled external calls. A separate directory with only `src/`
and assets outside `exp/` also passed physical CLI generation. The active
Jupyter mapping and named executed notebook hash were verified.

See [CompiledResult.md](../../../../exp/1007_sw_stack/deploy/CompiledResult.md)
for provenance, raw evidence locations, residual buffer fix and exact prompt
length constraints. This does not complete the cancelled workflow's original
W8+AWQ SmolLM, Qwen3, ResNet-18 or routed timing gates.
