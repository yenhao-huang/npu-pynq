# Install Icarus Verilog on Windows

## Contents

1. Purpose and installed layout
2. Pre-install inspection
3. Install MSYS2
4. Update MSYS2
5. Install Icarus Verilog
6. Configure Windows PATH
7. Validate the installation
8. Run the repository test
9. Repair and upgrade
10. Uninstall

## Purpose and installed layout

Use this workflow when `Get-Command iverilog,vvp` returns no commands or the
Icarus runner reports that they are absent. Install the UCRT64 package through
MSYS2; `vvp` is part of Icarus Verilog and is not installed separately.

Expected layout:

```text
C:\msys64\usr\bin\bash.exe
C:\msys64\ucrt64\bin\iverilog.exe
C:\msys64\ucrt64\bin\vvp.exe
```

Validated package/version for this host:

```text
MSYS2 installer: 2026-06-11
package:          mingw-w64-ucrt-x86_64-iverilog
Icarus version:   13.0 stable
```

Installation writes outside the repository and downloads packages. Obtain the
required execution approval before running `winget` or `pacman`; do not install
silently without user authorization.

## Pre-install inspection

From PowerShell:

```powershell
Get-Command winget,iverilog,vvp -ErrorAction SilentlyContinue
Test-Path C:\msys64\usr\bin\bash.exe
Test-Path C:\msys64\ucrt64\bin\iverilog.exe
Test-Path C:\msys64\ucrt64\bin\vvp.exe
```

Interpretation:

- Commands found: validate versions before reinstalling.
- Files exist but commands are absent: fix PATH; do not reinstall first.
- `C:\msys64` absent: install MSYS2.
- MSYS2 exists but Icarus files are absent: run the `pacman` package step.

## Install MSYS2

Use the Windows Package Manager package ID `MSYS2.MSYS2`:

```powershell
winget install --id MSYS2.MSYS2 --exact `
  --accept-source-agreements `
  --accept-package-agreements `
  --silent `
  --disable-interactivity
```

Gate: `C:\msys64\usr\bin\bash.exe` exists. If `winget` is unavailable, use
the official MSYS2 installer; do not use an unrelated third-party Icarus binary
bundle.

## Update MSYS2

MSYS2 core runtime upgrades may require the first shell process to exit. Run:

```powershell
& C:\msys64\usr\bin\bash.exe -lc 'pacman -Syu --noconfirm'
```

Then launch a fresh process and run it again so the remaining packages update:

```powershell
& C:\msys64\usr\bin\bash.exe -lc 'pacman -Syu --noconfirm'
```

Do not interrupt `pacman` while it is installing or updating the keyring.

## Install Icarus Verilog

Install the UCRT64 build:

```powershell
& C:\msys64\usr\bin\bash.exe -lc `
  'pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-iverilog'
```

Do not mix the UCRT64 package with paths from MINGW64 or CLANG64. The package
installs both compiler (`iverilog`) and runtime (`vvp`).

## Configure Windows PATH

Required entry:

```text
C:\msys64\ucrt64\bin
```

For the current PowerShell process only:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
```

For the current Windows user, preserve existing entries and add it once:

```powershell
$icarusBin = 'C:\msys64\ucrt64\bin'
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$entries = @($userPath -split ';' | Where-Object { $_ })
if ($entries -notcontains $icarusBin) {
    [Environment]::SetEnvironmentVariable(
        'Path',
        (@($entries) + $icarusBin) -join ';',
        'User'
    )
}
```

Changing the user PATH writes Windows user configuration. Obtain approval when
performing it on behalf of the user. Existing PowerShell processes do not
automatically receive the new PATH; open a new terminal or update `$env:PATH`.

## Validate the installation

First use absolute paths so PATH cannot hide an installation problem:

```powershell
& C:\msys64\ucrt64\bin\iverilog.exe -V
& C:\msys64\ucrt64\bin\vvp.exe -V
```

Then open a new PowerShell and verify command discovery:

```powershell
Get-Command iverilog,vvp
iverilog -V
vvp -V
```

Expected version is Icarus Verilog/runtime 13.0 stable. Version commands may
print license text; require exit code zero and the expected executable paths.

## Run the repository test

From `npu_repo_in_pynq`:

```powershell
& .\mount\mac_npu\test\run_rtl_sim.ps1
```

Required result:

```text
PASS: 263 MAC vectors matched
```

This compiles `mac_unit.sv` and `tb_mac_unit.sv` with SystemVerilog 2012 and
runs the generated `.vvp` program. It does not test AXI, synthesis, bitstream,
or the board; run `run_xsim.ps1` separately for AXI4-Lite verification.

## Repair and upgrade

If executables exist but dependencies or package files are damaged:

```powershell
& C:\msys64\usr\bin\bash.exe -lc `
  'pacman -Syu --noconfirm && pacman -S --noconfirm mingw-w64-ucrt-x86_64-iverilog'
```

If only command discovery fails, inspect PATH and restart PowerShell instead of
reinstalling. Use `pacman -Q mingw-w64-ucrt-x86_64-iverilog` to record the
installed package version.

## Uninstall

Remove only Icarus while keeping MSYS2:

```powershell
& C:\msys64\usr\bin\bash.exe -lc `
  'pacman -Rns --noconfirm mingw-w64-ucrt-x86_64-iverilog'
```

Removing all of MSYS2 or deleting `C:\msys64` is broader and destructive; do
that only when explicitly requested after verifying no other MSYS2 packages or
user files are needed. Remove the PATH entry separately if uninstalling MSYS2.
