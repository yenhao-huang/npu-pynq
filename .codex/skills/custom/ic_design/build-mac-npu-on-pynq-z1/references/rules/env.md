# Environment Rule

```text
Host OS:           Windows 11 / PowerShell
Primary languages: SystemVerilog, Python, PowerShell, Tcl
Python:            Python 3; MVP code is dependency-free on the host
FPGA tool:         AMD Vivado 2026.1
Vivado path:       C:\AMDDesignTools\2026.1\Vivado\bin
Target part:       xc7z020clg400-1 (PYNQ-Z1)
Simulator:         Vivado XSIM; Icarus Verilog 13.0
Icarus path:       C:\msys64\ucrt64\bin
MSYS2 package:     mingw-w64-ucrt-x86_64-iverilog
Board runtime:     PYNQ Linux with Python 3 and `pynq`
Transport:         OpenSSH/SCP over configured Ethernet
Service manager:   none
```

- Use `src/vivado_tcl/mac_npu/build_overlay.ps1` instead of assuming Vivado is
  on PATH.
- Use `references/installation/icarus-verilog-windows.md` for installation,
  repair, upgrade, PATH, and uninstall procedures. Do not duplicate package
  commands in general workflow references.
- Keep host Python reference/tests dependency-free. Do not install packages
  globally or modify the working Vivado license configuration.
- Use the host's existing valid Vivado Basic license. If sandbox isolation hides
  it, request narrowly scoped execution rather than copying license data.
- Do not terminate an existing interactive Vivado process without user consent.
- Read `configs/pynq-sync.json` for board host/user/root; do not invent network
  settings or store credentials.
- Keep deployable source and overlay artifacts under `mount/mac_npu/`; keep
  Vivado and XSIM work products under repository `results/`.
- Do not change a Windows adapter address, firewall, route, or board image
  without explicit user authorization.
- Never store board passwords, SSH private keys, generated caches, or Vivado
  run products inside the skill directory.
