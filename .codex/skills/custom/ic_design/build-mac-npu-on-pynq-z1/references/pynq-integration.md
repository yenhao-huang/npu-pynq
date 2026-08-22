# PYNQ Python MMIO Integration

## Overlay and metadata

Keep `mac_npu.bit` and `mac_npu.hwh` in the same directory with the same
basename. Load with `pynq.Overlay`; then require `mac_axi_lite_0` in
`overlay.ip_dict` before constructing MMIO.

PYNQ releases may expose address fields as either:

```text
phys_addr + addr_range
base_address + range
```

`load_mac_overlay()` supports both. Fail explicitly if neither schema is
available; do not silently use an arbitrary hard-coded address. The HWH remains
the authoritative normal runtime source, while the known design address is
`0x43C00000` with range `0x10000`.

## Driver API

`MacMMIO` accepts any object with `write(offset, value)` and `read(offset)`,
which permits fake-MMIO unit testing and real `pynq.MMIO` use.

```python
overlay, mac = load_mac_overlay("overlay/mac_npu.bit")
mac.clear()
result = mac.mac(2, 3)
```

Method behavior:

| Method | Required behavior |
| --- | --- |
| `clear()` | write `CONTROL_CLEAR` only |
| `set_operands(a,b)` | validate signed INT8; encode low 8 bits |
| `start()` | write `CONTROL_START` only |
| `wait_done()` | poll STATUS bit 0 until finite monotonic deadline |
| `read_accumulator()` | decode unsigned MMIO word as signed INT32 |
| `mac(a,b)` | set operands, start, wait, return current accumulated result |

`mac()` accumulates; it does not implicitly clear. Callers control accumulation
boundaries explicitly.

## Signed representation

Examples written to operand registers:

```text
 127 -> 0x0000007F
  -1 -> 0x000000FF
-128 -> 0x00000080
```

RTL reads only bits `[7:0]`. RESULT is a full 32-bit two's-complement word;
Python must subtract `2^32` when bit 31 is set.

## Timeout and errors

Use a finite default timeout. A stuck accelerator must raise `TimeoutError`
instead of hanging a Jupyter kernel indefinitely. Raise clear `ValueError`,
`TypeError`, or `KeyError` messages for operands and metadata before issuing
unsafe MMIO access.

## Board smoke test

Run through `deploy_and_test.ps1` or directly on the board:

```bash
cd /home/xilinx/jupyter_notebooks/pynq_z1_repo/mac_npu
python3 hardware_smoke_test.py --bitfile overlay/mac_npu.bit
```

Completion requires the PASS line defined in `SKILL.md`. After the scalar
milestone, replace per-element MMIO with AXI DMA for vector/matrix data while
retaining MMIO for control/status.
