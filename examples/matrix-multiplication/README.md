# Matrix multiplication deployment

This example builds the NPU overlay, deploys it to PYNQ-Z1, runs three matrix
tests, and downloads JSON evidence. Run all PowerShell commands from the
repository root.

## Requirements

- Vivado, Python, `ssh`, `scp`, and `tar` are available.
- The board is reachable as `pynq` or `192.168.2.99`.
- The Git worktree is clean before building deployable artifacts:

```powershell
git status --short
```

The command must print nothing. A dirty exploratory build may use
`-tclargs --allow-dirty`, but it deliberately does not publish deployable
artifacts.

## 1. Prepare the board once

```powershell
ssh pynq
sudo usermod -aG render,video xilinx
exit
ssh pynq id
```

The final output should include `render` and `video`. PYNQ also requires root
for `/dev/mem` MMIO, so local deployment will ask for the board sudo password.

## 2. Build and package

```powershell
vivado -mode batch -nojournal -nolog `
  -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl

python examples/matrix-multiplication/package_example.py
```

The package is created under
`mount/matrix-multiplication/local-<commit>`.

## 3. Deploy and test

```powershell
$package = Get-ChildItem mount/matrix-multiplication/local-* -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if (-not $package) { throw 'Build the package first.' }

$deploy = @{
  PackagePath = $package.FullName
  ReleaseTag = 'v0.0.0'
  DeploymentId = Get-Date -Format 'yyyyMMdd-HHmmss'
  EvidencePath = 'build/board/local-evidence.json'
  BoardHost = 'pynq'
}

& examples/matrix-multiplication/deploy_release.ps1 @deploy -DryRun
& examples/matrix-multiplication/deploy_release.ps1 @deploy -InteractiveSudo
```

Enter the board sudo password when prompted. Success prints:

```text
PASS: Phase 1C matrix multiplication example
PASS: deployed v0.0.0 as <deployment-id> and retrieved board evidence
```

Read the evidence:

```powershell
Get-Content build/board/local-evidence.json
```

## Release CD

`.github/workflows/cd.yml` uses the same deployment wrapper without
`-InteractiveSudo`. It therefore uses `sudo -n`; the dedicated board must have
a reviewed non-interactive sudo policy before GitHub Actions CD can pass. Never
store a password or private key in this repository.
