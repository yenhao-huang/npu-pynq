# Setup and prerequisites

## Host tools

Install AMD Vivado Design Suite on Windows. The installer used in this workflow is the official 2026.1 Windows unified installer:

- https://account.amd.com/zh-tw/forms/downloads/xef.html?filename=FPGAs_AdaptiveSoCs_Unified_SDI_2026.1_0616_1700_Win64.exe

At installer product selection, choose **Vivado**. Include the Zynq-7000 device family; cable drivers are useful for later JTAG work. Vitis, Bootgen, and unrelated device families are not required for this LED overlay.

## License

Create and load **Vivado Basic Tier License, Node Locked License (No Charge)** for the Windows development PC. It is tied to the selected host ID/MAC address. In Vivado use **Help → Manage License → Load/Copy License** to import the downloaded `.lic` file. Confirm that Vivado can obtain a synthesis license before progressing.

Do not select the 60-day Vivado Enterprise evaluation when Basic Tier covers the target device.

## Board and network

Use a PYNQ-Z1 booted from its PYNQ microSD image. Connect via Ethernet and reach Jupyter before attempting deployment. For direct PC-to-board Ethernet, a typical static pair is:

| Device | IPv4 | Mask |
| --- | --- | --- |
| PC Ethernet adapter | `192.168.2.1` | `255.255.255.0` |
| PYNQ-Z1 | `192.168.2.99` | `255.255.255.0` |

Leave gateway/DNS empty for this isolated direct link. A router connection normally supplies an address through DHCP.

## Device choice

PYNQ-Z1 uses a Zynq-7000 XC7Z020 device in a CLG400 package. Select the matching part shown by the installed Vivado board/device database. The current verified project used `xa7z020clg400-1I`; verify the selected package/speed/temperature grade against the physical board or installed PYNQ-Z1 board files before rebuilding an existing project.

Useful official sources:

- PYNQ-Z1 setup: https://pynq.readthedocs.io/en/v2.7.0/getting_started/pynq_z1_setup.html
- PYNQ project and board overlays: https://github.com/Xilinx/PYNQ
