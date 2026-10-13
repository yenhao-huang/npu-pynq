# PYNQ-Z1 runs: heterogeneous overlay with SmolLM2

Status: **PASS** on the final RTL.

| Run | Bitstream | Result |
| --- | --- | --- |
| 38329269229 (request `014-hexagon-final-board`, 2026-10-13) | final RTL, run 38205853103 (commit 5471ff4, VERSION `0x00010001`: STOP, 1-cycle TCM/MMIO; WNS +0.113 ns) | self-test PASS, tensor GEMMs PASS, SmolLM2 PASS |
| 38184383844 (request `009-hexagon-board`, 2026-10-11) | first `--hexagon` build, run 38169804215 (VERSION `0x00010000`; WNS +0.063 ns) | self-test PASS, tensor GEMMs PASS, SmolLM2 PASS |

Raw results: [014](../../../exp/1011_hexagon_npu/board_runs/014-hexagon-final-board/),
[009](../../../exp/1011_hexagon_npu/board_runs/009-hexagon-board/).

## Final run (38329269229)

| Step | Result | Measurement |
| --- | --- | --- |
| Identify | PASS | MAGIC `0x314E5848`, VERSION `0x00010001`, CAPS `0x0010087F` |
| Scalar sort/sum program | PASS (vs ISS) | 14,810 instructions in 57,311 cycles, CPI 3.87; L1I 99.97 %, L1D 99.85 % hits |
| INT8 saturating add, 512 elements | PASS, bit-exact (23 calls checked) | kernel 1,447 cycles with the resident `VectorOp` (1,702 in the self-test's one-shot path); 22.3x numpy on the A9 |
| FP16 a*x+y, 512 elements | PASS, bit-exact (23 calls checked) | kernel 3,231 cycles resident (3,525 one-shot); 5.3x numpy |
| Tensor ISA GEMMs | PASS | every case matches NumPy |
| SmolLM2 | PASS | "The capital of France is Paris. Paris is a city located in the northern part of the country, and it is known"; prefill 2.33 tok/s, decode 1.45 tok/s; 3,025 NPU calls |

Per-call time from Python, 512 elements:

| Path | INT8 add | FP16 a*x+y |
| --- | ---: | ---: |
| One-shot (run 009 driver) | 1.79 ms | 1.67 ms |
| One-shot (final driver) | 1.09 ms | 1.08 ms |
| `VectorOp` steady state (median of 20) | 0.86 ms | 0.95 ms |
| of which: copy into the pynq buffer + ARM cache flush | 0.28 ms | 0.28 ms |
| DMA in / NPU run / DMA out + ARM cache invalidate | 0.12 / 0.11 / 0.41 ms | 0.12 / 0.13 / 0.32 ms |
| numpy on the Cortex-A9 | 0.38 ms | 0.19 ms |
| MMIO accesses per steady call | 9 W + 4 R | 9 W + 6 R |

The NPU part of a call (DMA + kernel) is ~0.35 ms; the rest is the A9's
cache maintenance on the pynq buffers. At 512 elements that still loses to
numpy end to end; the kernel itself is 5-22x faster, so larger tensors or
non-cacheable staging buffers (no flush/invalidate) are what make the vector
unit pay off from Python.

## First run (38184383844)

### Self-test (`exp/1011_hexagon_npu/board/hx_selftest.py`)

| Step | Result | Measurement |
| --- | --- | --- |
| Identify | PASS | MAGIC `0x314E5848`, CAPS `0x0010087F` (scalar, vector, L1I, L1D, L2, TCM, DMA; VLEN 64; 1024 TCM words); mailboxes and TCM window read back |
| Scalar program in DDR (sum, xor, signed bubble sort of 64 words, min/max) | PASS: memory image and instruction count equal the ISS | 14,810 instructions in 57,314 cycles = CPI 3.87 (ISS with ideal memory: 56,762); 573 us at 100 MHz; L1I 14,806 hits / 4 misses (99.97 %), L1D 6,157 / 9 (99.85 %), L2 9 / 13 |
| INT8 saturating residual add, 512 elements (DDR -> DMA -> TCM -> vector -> DMA -> DDR) | PASS: bit-exact with `hx_vector` and numpy | kernel 1,702 cycles (3.32 cycles/element, 30 M elements/s at 100 MHz); DMA in 51 us, out 43 us |
| FP16 a*x+y (a = -0.75), 512 elements | PASS: bit-exact | kernel 3,544 cycles (6.92 cycles/element, 14.4 MFLOP-pairs/s) |
| Tensor unit (ISA GEMM checks, `exp/1007_sw_stack/board/isa_gemm.py`) | PASS: every case matches NumPy | e.g. 64 x 1536 x 576 at 2.87 GMAC/s, 197 x 64 x 147 at 2.40 GMAC/s; ABI 1.0 tile path still works before and after ISA programs |

Comparison with the Cortex-A9 (first run) (numpy, same 512 elements):

| Operation | NPU vector kernel | numpy on the A9 | Kernel speed-up | End to end from Python |
| --- | ---: | ---: | ---: | ---: |
| INT8 saturating add | 17.0 us | 400 us | 23.5x | 1.79 ms |
| FP16 a*x+y | 35.4 us | 191 us | 5.4x | 1.67 ms |

The kernel itself is 5-23x faster than numpy on the A9, but end to end the
Python driver path (program load, flushes, DMA descriptors, MMIO polling)
costs ~1.6 ms per call and dominates at 512 elements. Integration review
round 3 targets that overhead (resident program, reused buffers); it is
the main item for the next board run.

### SmolLM2-135M-Instruct on the same overlay

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
