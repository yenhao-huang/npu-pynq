# MAC NPU MVP

This directory contains the first independently verifiable PYNQ-Z1 MAC
milestone: signed INT8 operands with a signed INT32 wrapping accumulator.

```text
src/rtl/mac_npu/mac_unit.sv  synthesizable arithmetic core
sw/mac_npu/mac_reference.py Python golden model
test/tb_mac_unit.sv           self-checking SystemVerilog testbench
test/test_mac_reference.py    dependency-free Python tests
test/run_rtl_sim.ps1          Icarus Verilog simulation entry point
test/run_xsim.ps1             Vivado XSIM AXI4-Lite simulation
configs/mac_mvp.json          machine-readable arithmetic contract
docs/spec.md                  interface and timing specification
```

Run the software model tests from the repository root:

```powershell
python -m unittest discover -s sw/mac_npu -p 'test_*.py' -v
```

If `iverilog` and `vvp` are installed:

```powershell
& .\mount\mac_npu\test\run_rtl_sim.ps1
```

Build the PYNQ-Z1 overlay with the installed Vivado 2026.1:

```powershell
& .\mount\mac_npu\core\api\build_overlay.ps1
```

This creates `overlay/mac_npu.bit` and `overlay/mac_npu.hwh`. On the PYNQ
board, run:

```bash
python3 hardware_smoke_test.py --bitfile overlay/mac_npu.bit
```

From Windows, upload changed files and run that board test in one command:

```powershell
& .\mount\mac_npu\core\api\deploy_and_test.ps1
```

The Python control sequence is:

```python
from mac_mmio import load_mac_overlay

overlay, mac = load_mac_overlay("overlay/mac_npu.bit")
mac.clear()
print(mac.mac(2, 3))   # 6
print(mac.mac(-7, 6))  # -36 (accumulated)
```
