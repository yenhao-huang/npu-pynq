# Environment Rules

Primary language: SystemVerilog / Verilog
Build tooling: GNU Make, Tcl
Synthesis: AMD Vivado (Windows or Linux, licensed, local only)
Open-source simulation: Verilator, Icarus Verilog, optionally cocotb
CI platform: GitHub Actions
Package manager: apt on runners; none in the repository

- Vivado is never available on a GitHub-hosted runner. Treat any synthesis,
  implementation, or bitstream step as self-hosted only.
- Verilator and Icarus install from apt in seconds and carry no licence.
- Vivado on Windows is invoked through `vivado.bat`; keep Tcl paths relative to
  the repository root so the same script runs on either platform.
- Git on Windows may report `dubious ownership`. Resolve with
  `git config --global --add safe.directory <path>` before any `git mv`.
