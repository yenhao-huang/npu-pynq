# Development Matrix

Select all affected areas. Read existing design-specific references in addition
to this matrix; this file does not replace them.

| Area | Contract to identify before edits | Minimum change evidence |
| --- | --- | --- |
| Numeric model | quantization, rounding, saturation, accumulator width and overflow | focused Python/golden-model tests and consumer parity |
| RTL/interface | clock/reset, signed widths, latency, handshakes, register map | matching testbench or coverage rationale, lint, simulation |
| Constraints/Tcl | part, clocks, pins, IP versions, address map, reproducibility | Tcl parse/build evidence, constraint/timing reports when applicable |
| Export/compiler | input model assumptions, layout, quantization metadata, executable format | deterministic fixture and runtime/golden compatibility test |
| PYNQ runtime | overlay/HWH pairing, IP discovery, MMIO offsets, polling/timeouts, signed readback | host tests plus board test when board-visible behavior can change |
| Examples/docs | public API, reproducible inputs/outputs, safe defaults | notebook or example smoke test; documentation links resolve |
| Deployment/board | approved staging path, matching artifact basenames, target host, rollback | transfer manifest and physical-board PASS evidence |

## Cross-area rules

- If arithmetic or register behavior changes, evaluate numeric model, RTL,
  export, runtime, and tests together even if only one file was requested.
- If Tcl or constraints change, determine whether address, timing, or HWH
  metadata can change and include downstream runtime/board gates accordingly.
- Board validation may be marked not applicable only when the change cannot
  affect overlay loading, MMIO, timing, I/O, or hardware-only behavior. Record
  the rationale in `STATE.md` and the OpenSpec task.
- When a contract is unknown, add its definition to the OpenSpec design/spec
  before implementation. Do not infer a hardware contract from one consumer.
