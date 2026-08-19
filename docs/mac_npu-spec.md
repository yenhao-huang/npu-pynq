# MAC MVP Specification

At each rising clock edge:

1. If `rst_n == 0`, set `result` and `result_valid` to zero.
2. Otherwise, if `clear == 1`, set `result` and `result_valid` to zero.
3. Otherwise, if `valid == 1`, update `result` with
   `result + signed(a) * signed(b)` and assert `result_valid` for that cycle.
4. Otherwise, preserve `result` and deassert `result_valid`.

`a` and `b` are signed two's-complement INT8 values. `result` is a signed
two's-complement INT32 accumulator. Overflow wraps modulo 2^32. The core has no
AXI dependency and may accept one operand pair per clock.

The MVP is complete when the Python model tests and the self-checking RTL
simulation both pass. AXI4-Lite integration is the next milestone.

## AXI4-Lite register map

| Offset | Name | Access | Meaning |
| ---: | --- | --- | --- |
| `0x00` | CONTROL | W | bit 0: one-cycle clear; bit 1: one-cycle start |
| `0x04` | A | R/W | signed INT8 operand in bits 7:0; read sign-extends |
| `0x08` | B | R/W | signed INT8 operand in bits 7:0; read sign-extends |
| `0x0c` | STATUS | R | bit 0: sticky done; cleared by clear/start |
| `0x10` | RESULT | R | signed INT32 accumulator |

Write operands first, then write `2` to CONTROL. Poll STATUS bit 0 and read
RESULT. Write `1` to CONTROL to clear the accumulator. CONTROL is pulse-based
and reads as zero.
