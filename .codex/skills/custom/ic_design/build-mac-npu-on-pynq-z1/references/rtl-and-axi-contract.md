# RTL and AXI4-Lite Contract

## Contents

1. Arithmetic core interface
2. Width and signedness rules
3. AXI4-Lite register map
4. Command/status timing
5. AXI protocol requirements
6. Test vectors
7. Change checklist

## Arithmetic core interface

`mac_unit` is synchronous to `clk` and uses active-low synchronous reset
`rst_n`. `clear` and `valid` are one-cycle controls. `result_valid` pulses for
an accepted MAC operation; it is not a persistent completion flag.

Priority is:

```text
!rst_n -> clear -> valid -> idle
```

When idle, hold `result` and drive `result_valid` low.

## Width and signedness rules

- Declare both operands and the product as signed.
- Product width is `A_WIDTH + B_WIDTH` (16 bits for INT8 × INT8).
- Sign-extend the product to `ACC_WIDTH` before addition.
- The accumulator assignment naturally keeps the low 32 bits; this defines
  two's-complement wrap rather than saturation.
- Reject parameter combinations where `ACC_WIDTH < A_WIDTH + B_WIDTH`, or keep
  the current known-safe 8/8/32 parameters. Do not silently truncate a wider
  product during sign extension.

## AXI4-Lite register map

| Offset | Name | Access | Bits | Behavior |
| ---: | --- | --- | --- | --- |
| `0x00` | CONTROL | write | bit 0 clear | One-cycle pulse; clears accumulator and done |
| `0x00` | CONTROL | write | bit 1 start | One-cycle pulse; accepts current A/B and clears stale done |
| `0x04` | A | read/write | bits 7:0 | Signed INT8; read sign-extends to 32 bits |
| `0x08` | B | read/write | bits 7:0 | Signed INT8; read sign-extends to 32 bits |
| `0x0C` | STATUS | read | bit 0 done | Sticky until a new clear or start command |
| `0x10` | RESULT | read | bits 31:0 | Signed INT32 accumulator |

CONTROL reads as zero. Undefined addresses read zero and writes have no effect.
Only byte lane zero changes CONTROL/A/B; honor `WSTRB[0]`.

## Command/status timing

Software order for one MAC:

```text
write A
write B
write CONTROL.start
poll STATUS.done == 1
read RESULT
```

On a start write, `start_pulse` reaches `mac_unit` on the following sequential
cycle. `mac_result_valid` then sets sticky done. Software must poll instead of
assuming a fixed CPU delay.

Clear and start should not be asserted in the same CONTROL write. If software
does so, the core's clear priority wins; keep tests and driver on single-command
writes (`1` for clear, `2` for start).

## AXI protocol requirements

- Accept AW and W independently; buffer each until both are present.
- Emit exactly one write response per committed register write.
- Keep `BVALID` asserted until `BREADY`.
- Accept one read address when no prior read response is pending.
- Keep `RVALID` and `RDATA` stable until `RREADY`.
- Return `OKAY` for implemented and undefined offsets in this simple peripheral.
- Reset all pending flags, responses, operands, pulses, and sticky status.

Do not require AWVALID and WVALID in the same cycle. Such a wrapper may pass a
friendly testbench but fail with a real AXI interconnect.

## Test vectors

Use at least:

| Step | A | B | Expected accumulator |
| ---: | ---: | ---: | ---: |
| clear | – | – | `0` |
| start | `2` | `3` | `6` |
| start | `-7` | `6` | `-36` |
| clear | – | – | `0` |
| start | `-128` | `-128` | `16384` |
| start | `127` | `-128` | `128` |

Also verify A/B signed readback, idle result hold, reset, and done clearing.

## Change checklist

When modifying a register offset, bit, width, or timing rule, update all of:

- `mac_axi_lite.sv`
- `tb_mac_axi_lite.sv`
- `mac_mmio.py`
- `test_mac_mmio.py`
- `hardware_smoke_test.py`
- `configs/mac_mvp.json`
- `docs/spec.md`
- this reference and `pynq-integration.md`

Re-run Python, XSIM, Vivado, HWH inspection, and board smoke test in that order.
