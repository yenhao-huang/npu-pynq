# PYNQ-Z1 installation selections

The PYNQ-Z1 uses an AMD/Xilinx Zynq-7000 device, specifically `XC7Z020-1CLG400C`.

## Product page

Select **Vivado** as the product to install.

## Vivado customization page

Keep:

- Vivado Design Suite / Vivado
- Vitis HLS (included with the Vivado design tools and useful for developing an accelerator from C/C++)
- Install Cable Drivers (recommended for JTAG/UART work)

Under **Devices**, expand **SoCs** and select only **Zynq-7000**. This provides support for `xc7z020clg400-1` used by the PYNQ-Z1.

Usually leave unselected unless a separate project requires them:

- Vitis Model Composer (MATLAB/Simulink)
- Vitis Embedded Development
- Power Design Manager
- DocNav
- Alveo, Kria, 7 Series, UltraScale, UltraScale+, Versal, and Engineering Sample device families

`Acquire or Manage a License Key` may remain selected if the user needs to import a certificate license after installation.

These choices minimize the download and installed footprint while retaining the hardware-design flow needed for a PYNQ overlay.

Sources: [PYNQ-Z1 reference manual](https://reference.digilentinc.com/_media/reference/programmable-logic/pynq-z1/pynq-rm.pdf), [AMD supported devices](https://docs.amd.com/r/2021.2-English/ug973-vivado-release-notes-install-license/Supported-Devices).
