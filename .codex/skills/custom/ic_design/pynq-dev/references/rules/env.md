# Environment Rules

```text
Host OS:             Windows 11 / PowerShell
Primary languages:  SystemVerilog, Python, Tcl, PowerShell
Target board:        PYNQ-Z1 / Zynq-7020 / xc7z020clg400-1
FPGA tools:          repository-defined AMD Vivado version and valid license
Simulation:          repository Makefile, cocotb and configured HDL simulator
Board runtime:       PYNQ Linux with Python 3 and pynq
OpenSpec:            CLI required; prefer existing executable
Services:            none unless the repository explicitly defines them
```

- Prefer repository dependency files, wrappers, Makefiles, and pinned tool
  versions. Do not install packages globally.
- If `openspec` is unavailable on PATH and Node.js is available, resolve it
  locally with `npx.cmd -y @fission-ai/openspec@latest <command>` on Windows;
  record the resolved version. Do not silently change OpenSpec versions mid-run.
- Do not invent Vivado paths, versions, licenses, simulator settings, board
  addresses, users, passwords, ports, or services. Read repository config and
  specialized skill references.
- Missing tool, license, network, or board access is a blocked validation gate,
  not permission to alter the host or board configuration.
- Do not terminate interactive Vivado processes or change network/firewall,
  routes, adapters, board images, or credentials without explicit user approval.
