# Environment Rules

```text
Primary languages: SystemVerilog, Python, Tcl, PowerShell
Target:            PYNQ-Z1 / Zynq-7020 / xc7z020clg400-1
Build tooling:     GNU Make and repository-relative Tcl
Synthesis:         AMD Vivado, licensed, local or self-hosted runner only
Simulation:        Verilator, Icarus Verilog, optionally cocotb
CI platform:       GitHub Actions
Board runtime:     PYNQ Linux with Python 3 and pynq
```

- Prefer repository Makefiles, Tcl wrappers, dependency files, and pinned tool
  versions. Do not install packages globally for repository work.
- GitHub-hosted runners may run open-source lint, simulation, and Python tests;
  they must not claim to run Vivado synthesis or implementation.
- Vivado jobs require a licensed local installation or a trusted self-hosted
  runner. Keep Tcl paths relative to the repository root.
- Do not invent Vivado versions, paths, licences, board addresses, users,
  passwords, ports, or services. Read the repository and deployment skill.
- Missing tools, licences, network access, or board access are blockers to the
  corresponding validation gate, not permission to modify the environment.
- Do not terminate interactive Vivado processes or change adapters, firewall,
  routes, board images, licences, or credentials without explicit approval.
- On Windows, if Git reports dubious ownership, report the exact repository
  path before changing any global Git configuration.
