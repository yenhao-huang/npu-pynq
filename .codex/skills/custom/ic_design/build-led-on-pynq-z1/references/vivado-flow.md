# Vivado block design and constraints

## Build the minimal hardware

1. Create an RTL project and select the PYNQ-Z1 target part.
2. Create a Block Design, e.g. `hello_pynq`.
3. Add `ZYNQ7 Processing System`.
4. Re-customize it under **PS-PL Configuration**:
   - enable `M AXI GP0 interface`;
   - enable `FCLK_CLK0`;
   - set FCLK_CLK0 to 100 MHz if no other clock requirement exists.
5. Add `AXI GPIO`.
6. Re-customize AXI GPIO:
   - set GPIO Width to `4`;
   - configure channel 1 as output (do not use All Inputs);
   - use one channel only for this exercise.
7. Run Connection Automation for AXI GPIO. It normally adds AXI SmartConnect and Processor System Reset. These extra blocks are expected.
8. Right-click AXI GPIO's GPIO interface and select **Make External**.
9. Run **Validate Design**, then save.

Do not click **Generate Block Design** again after edits. That action creates a new Block Design; the required action is to re-open and validate the existing one.

## Create HDL wrapper

In Sources, right-click the `.bd` file and choose **Create HDL Wrapper**. Select **Let Vivado manage wrapper and auto-update**. This wrapper exposes the external GPIO as a real top-level HDL port.

## LED constraints

Create a Constraints XDC file, e.g. `leds.xdc`, through **Add Sources → Add or create constraints → Create File**.

For this project, the wrapper port was `gpio_rtl_0_tri_io[3:0]`; therefore use:

```xdc
# PYNQ-Z1 green LEDs LD0 through LD3
set_property -dict { PACKAGE_PIN R14 IOSTANDARD LVCMOS33 } [get_ports {gpio_rtl_0_tri_io[0]}]
set_property -dict { PACKAGE_PIN P14 IOSTANDARD LVCMOS33 } [get_ports {gpio_rtl_0_tri_io[1]}]
set_property -dict { PACKAGE_PIN N16 IOSTANDARD LVCMOS33 } [get_ports {gpio_rtl_0_tri_io[2]}]
set_property -dict { PACKAGE_PIN M14 IOSTANDARD LVCMOS33 } [get_ports {gpio_rtl_0_tri_io[3]}]
```

The external Block Design interface name can differ from the generated wrapper port. XDC uses the wrapper's top-level port name, not necessarily the label displayed in the Block Design. If bitstream DRC says a different `Problem ports:` name, use that exact name in each `get_ports` expression.

## Build output

After saving the XDC, run **Generate Bitstream**. If Vivado says Synthesis is out-of-date after a change, choose **Yes**; it will re-run synthesis and implementation automatically.

Typical outputs are:

```text
<project>.runs/impl_1/hello_pynq_wrapper.bit
<project>.gen/sources_1/bd/hello_pynq/hw_handoff/hello_pynq.hwh
```

The paths vary with project and Block Design names.
