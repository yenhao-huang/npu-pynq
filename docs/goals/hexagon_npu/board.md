# PYNQ-Z1 run: heterogeneous overlay with SmolLM2

Status: **PASS**. exp-board run 38184383844 (request `009-hexagon-board`),
2026-10-11, on the PYNQ-Z1 at 192.168.2.99. Overlay: `--hexagon` build of
run 38169804215 (Vivado 2026.1, 16 x 16 INT8 tensor array + `npu_hx_0`,
100 MHz, WNS +0.063 ns, 0 failing endpoints, 0 DRC errors). Runtime and
board scripts from the same branch. Raw results:
[exp/1011_hexagon_npu/board_runs/009-hexagon-board/](../../../exp/1011_hexagon_npu/board_runs/009-hexagon-board/).

The bitstream predates integration review rounds 2-3 (it reports VERSION
`0x00010000`: no CONTROL.STOP, 2-cycle TCM/MMIO responses); the runtime
detects that and uses the older behaviour.

## Self-test (`exp/1011_hexagon_npu/board/hx_selftest.py`)

| Step | Result | Measurement |
| --- | --- | --- |
| Identify | PASS | MAGIC `0x314E5848`, CAPS `0x0010087F` (scalar, vector, L1I, L1D, L2, TCM, DMA; VLEN 64; 1024 TCM words); mailboxes and TCM window read back |
| Scalar program in DDR (sum, xor, signed bubble sort of 64 words, min/max) | PASS: memory image and instruction count equal the ISS | 14,810 instructions in 57,314 cycles = CPI 3.87 (ISS with ideal memory: 56,762); 573 us at 100 MHz; L1I 14,806 hits / 4 misses (99.97 %), L1D 6,157 / 9 (99.85 %), L2 9 / 13 |
| INT8 saturating residual add, 512 elements (DDR -> DMA -> TCM -> vector -> DMA -> DDR) | PASS: bit-exact with `hx_vector` and numpy | kernel 1,702 cycles (3.32 cycles/element, 30 M elements/s at 100 MHz); DMA in 51 us, out 43 us |
| FP16 a*x+y (a = -0.75), 512 elements | PASS: bit-exact | kernel 3,544 cycles (6.92 cycles/element, 14.4 MFLOP-pairs/s) |
| Tensor unit (ISA GEMM checks, `exp/1007_sw_stack/board/isa_gemm.py`) | PASS: every case matches NumPy | e.g. 64 x 1536 x 576 at 2.87 GMAC/s, 197 x 64 x 147 at 2.40 GMAC/s; ABI 1.0 tile path still works before and after ISA programs |

Comparison with the Cortex-A9 (numpy, same 512 elements):

| Operation | NPU vector kernel | numpy on the A9 | Kernel speed-up | End to end from Python |
| --- | ---: | ---: | ---: | ---: |
| INT8 saturating add | 17.0 us | 400 us | 23.5x | 1.79 ms |
| FP16 a*x+y | 35.4 us | 191 us | 5.4x | 1.67 ms |

The kernel itself is 5-23x faster than numpy on the A9, but end to end the
Python driver path (program load, flushes, DMA descriptors, MMIO polling)
costs ~1.6 ms per call and dominates at 512 elements. Integration review
round 3 targets that overhead (resident program, reused buffers); it is
the main item for the next board run.

## SmolLM2-135M-Instruct on the same overlay

| | |
| --- | --- |
| Prompt | "What is the capital of France?" |
| Output (24 tokens, greedy) | "The capital of France is Paris. Paris is a city located in the northern part of the country, and it is known" |
| Package | W8A8 + AWQ, weights streamed, exported by `exp/1011_hexagon_npu/export_smollm.sh` |
| Load | 19.0 s |
| Prefill | 16 tokens in 7.80 s (2.05 tok/s) |
| Decode | 24 tokens in 16.7 s (1.44 tok/s) |
| NPU (tensor unit) | 3,025 calls, 1,040,400 jobs, 4.95 GMAC, 511.7 M cycles |

SmolLM2's GEMMs run on the tensor unit through the ISA front end, exactly as
on the overlay without `--hexagon`; the run shows the added subsystem does
not disturb the tensor path, its runtime or the shared HP0 port.
