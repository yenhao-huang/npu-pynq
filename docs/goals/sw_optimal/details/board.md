# PYNQ-Z1 runs

Status: **pending hardware**. The board (`192.168.2.99`) and the self-hosted
Vivado/board runner (`DESKTOP-54U632L`) have been offline since
2026-10-06 16:30 UTC.

exp-board run 37508256815 (request `006-models-on-board`) is queued:

| Job | Runner | State |
| --- | --- | --- |
| export: SmolLM2 (W8A8 + AWQ), Qwen3-0.6B (W4A8 g64 + AWQ), ResNet-18 (W8A8) for the Cortex-A9 | GitHub-hosted, LLVM 22 from apt | PASS (packages 132 MB, 398 MB, 12 MB) |
| build-overlay: ISA front end, 16 x 16 array, 100 MHz | self-hosted Vivado | queued |
| board: ISA GEMM check, ResNet-18 gallery, SmolLM2 and Qwen3 chat | self-hosted, PYNQ-Z1 | queued |

When it runs, `results/` (board log, `isa_gemm.json`, `models.json`) comes
back as the `exp-board-006-models-on-board` artifact and this page records it.
