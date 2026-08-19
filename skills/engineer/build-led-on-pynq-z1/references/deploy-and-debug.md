# Deploy and debug

## Check board onboard
"Done" LED should be lightening

## Prepare the overlay files

PYNQ expects the hardware handoff file beside the bitstream with the same basename. For example:

```text
hello_pynq.bit
hello_pynq.hwh
```

If Vivado produced `hello_pynq_wrapper.bit` but `hello_pynq.hwh`, copy them into a deployment folder and rename one so both use one shared basename. Preserve the original generated outputs in the Vivado project.

Upload both files to a board directory such as:

```text
/home/xilinx/jupyter_notebooks/hello_pynq/
```

## Jupyter verification

Load the overlay and first verify that PYNQ discovered AXI GPIO:

```python
from pynq import Overlay

ol = Overlay('/home/xilinx/jupyter_notebooks/hello_pynq/hello_pynq.bit')
print(ol.ip_dict.keys())
```

Then drive LD0-LD3:

```python
leds = ol.axi_gpio_0.channel1
leds.setdirection('out')

leds.write(0b1111, 0xF)  # all four LEDs on
leds.write(0b1010, 0xF)  # LED1 and LED3 on
leds.write(0b0000, 0xF)  # all four LEDs off
```

`write(value, mask)` updates the masked output bits. Bit 0 maps to LD0 and bit 3 maps to LD3.

## Failure diagnosis

### `NSTD-1` / `UCIO-1` at `write_bitstream`

Vivado has external ports without an I/O standard or physical package location. Do not lower these DRC severities. Add `PACKAGE_PIN` and `IOSTANDARD LVCMOS33` constraints for each LED bit. Ensure the name in `get_ports` exactly matches the log's `Problem ports:` value.

### `gpio_rtl_0` versus `gpio_rtl_0_tri_io`

AXI GPIO's wrapper can add `_tri_io`. A constraint that targets `gpio_rtl_0[0]` while the actual wrapper port is `gpio_rtl_0_tri_io[0]` has no effect. Fix the XDC and rebuild.

### "Synthesis is out-of-date"

This is expected after changing the Block Design, wrapper, or XDC. Choose **Yes** to rerun synthesis and implementation before bitstream generation.

### Out-of-context IP warning

Inspect the final lines of `<project>.runs/impl_1/runme.log`. Some individual IP logs can show an early message while ending with `synth_design completed successfully`; the final implementation/bitstream DRC is the deciding result.

References:

- PYNQ AXI GPIO API: https://pynq.readthedocs.io/en/v3.1/pynq_libraries/axigpio.html
- PYNQ overlay loading: https://pynq.readthedocs.io/en/v3.1/pynq_overlays/loading_an_overlay.html
