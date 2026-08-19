# MAC NPU Architecture and Roadmap

## Scalar milestone

The current hardware computes:

```text
accumulator_next = wrap_int32(accumulator + int8(a) * int8(b))
```

| Property | Contract |
| --- | --- |
| Operand A/B | signed two's-complement INT8 |
| Product | signed 16-bit |
| Accumulator | signed two's-complement INT32 |
| Overflow | modulo 2^32 wrap |
| Control priority | reset, clear, MAC, idle |
| Pure-core input rate | at most one operand pair per clock |
| Software transport | AXI4-Lite MMIO through Zynq PS M_AXI_GP0 |
| Result readiness | one-cycle core valid converted to sticky AXI status |

The pure core and transport wrapper are separate modules. Preserve that boundary
so arithmetic can be tested without AXI and future stream/DMA wrappers can reuse
the core.

## System boundary

```text
PYNQ Python on ARM
  -> Overlay loads mac_npu.bit + parses mac_npu.hwh
  -> MMIO writes AXI registers
  -> Zynq PS M_AXI_GP0
  -> AXI interconnect
  -> mac_axi_lite
  -> mac_unit
  -> sticky done + accumulator readback
```

This is a memory-mapped accelerator, not an x86 or ARM ISA extension. Calling
the CONTROL register a “MAC instruction” is acceptable only at the software
API level; it is not decoded by the ARM CPU pipeline.

## Why AXI4-Lite comes first

AXI4-Lite is deliberately low-throughput but exposes every control transition,
making it suitable for first board bring-up and signed arithmetic validation.
Do not use repeated MMIO writes as the final matrix data path: CPU/register
overhead will dominate MAC time.

## Growth path and gates

```text
scalar mac_unit
  -> AXI4-Lite board PASS
  -> AXI4-Stream vector protocol
  -> AXI DMA + contiguous PYNQ buffers
  -> vector dot product
  -> parallel processing elements
  -> BRAM-buffered tiled GEMM
  -> stable Python runtime API
  -> optional MLIR/TVM/LLVM lowering
```

For each transition, keep the prior layer testable. Before adding vector/DMA,
record scalar board PASS. Before adding GEMM, prove dot-product shape/length,
backpressure, `TLAST`, DMA cache coherence, and randomized software equivalence.
Before compiler work, expose a stable runtime such as `dot()` or `matmul()` with
well-defined dtype, shape, buffer ownership, timeout, and error behavior.
