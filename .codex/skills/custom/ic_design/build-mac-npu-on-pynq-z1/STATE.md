# Build MAC NPU on PYNQ-Z1 Skill State

Run ID: install-icarus-and-document-20260813
Instance: npu_repo_in_pynq/.codex/skills/custom/ic_design/build-mac-npu-on-pynq-z1
Started: 2026-08-13
Scope: Install Icarus Verilog on Windows and isolate package-installation workflows under references/installation.

Last updated: 2026-08-13

| Step | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 0. Define Scope | completed | User requested Icarus installation plus an independent `references/installation/` workflow in this skill. | Preserve the existing skill category and implementation layout. |
| 1. Read Relevant Context | completed | Read skill-create, skill-creator, dev conventions, current skill rules/state, RTL runner, Git status, and host tool state. | MSYS2, `iverilog`, and `vvp` were absent; `winget` exists but required execution outside the sandbox. |
| 2. Execute Workflow | completed | Installed MSYS2 2026-06-11 and `mingw-w64-ucrt-x86_64-iverilog` 13.0; added `C:/msys64/ucrt64/bin` to the Windows user PATH; added the directly linked `references/installation/icarus-verilog-windows.md`; updated skill filetree/environment/verification routing; made `run_rtl_sim.ps1` resolve PATH or standard MSYS2 and expose its runtime DLL path. | `vvp` is provided by the same Icarus package; no separate install was performed. |
| 3. Validate Result | completed | `pacman -Q` reports `mingw-w64-ucrt-x86_64-iverilog 1~13.0-2`; absolute `iverilog -V` and `vvp -V` report 13.0 stable; user PATH contains the UCRT64 bin; `run_rtl_sim.ps1` printed `PASS: 263 MAC vectors matched`; generic skill validation, direct links, required layout, and PowerShell parsing passed. | Existing PowerShell windows may need restart; runner fallback works immediately. |
| 4. Handoff Summary | completed | Installation, update, PATH, validation, repository test, repair, upgrade, and uninstall procedures are isolated under `references/installation/`. | Use XSIM separately for the AXI4-Lite wrapper; Icarus test covers the pure MAC core. |
