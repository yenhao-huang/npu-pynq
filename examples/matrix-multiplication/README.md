# Matrix multiplication example

This Phase 1C example tiles large matrix multiplication over the Phase 1B
PYNQ runtime.

## Requirements

- Clean Git worktree.
- Vivado installed, licensed, and available as `vivado`.
- PYNQ-Z1 reachable through SSH as `pynq_board` or `192.168.2.99`.
- Python 3, NumPy, and PYNQ installed on the board.

Run every command below from the repository root in PowerShell.

## 1. Build the overlay

```powershell
vivado -mode batch -nojournal -nolog `
  -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl
```

This must create the BIT, HWH, provenance manifest, and build evidence under
`build/vivado/npu_matrix/`.

## 2. Build the standalone package

```powershell
python examples/matrix-multiplication/package_example.py
```

The output is `mount/matrix-multiplication/local-<8-char-commit>`.

## 3. Deploy and test

Prepare one deployment command. Replace `192.168.2.99` if the board uses a
different SSH host. `v0.0.0` is reserved here for local deployment only.

```powershell
$package = Get-ChildItem mount/matrix-multiplication/local-* -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$deploy = @{
  PackagePath = $package.FullName
  ReleaseTag = 'v0.0.0'
  DeploymentId = Get-Date -Format 'yyyyMMdd-HHmmss'
  EvidencePath = 'build/board/local-evidence.json'
  BoardHost = '192.168.2.99'
}
```

Preview the exact paths without connecting to the board:

```powershell
& examples/matrix-multiplication/deploy_release.ps1 @deploy -DryRun
```

If the preview is correct, deploy and execute the board tests:

```powershell
& examples/matrix-multiplication/deploy_release.ps1 @deploy
```

The script uploads the package, verifies the overlay, runs normal,
non-aligned, and repeated matrix cases, then downloads the evidence. Success
prints:

```text
PASS: Phase 1C matrix multiplication example
PASS: deployed v0.0.0 as <deployment-id> and retrieved board evidence
```

Read the result with:

```powershell
Get-Content build/board/local-evidence.json
```

SSH may request host confirmation or a password interactively. Never place a
password or private key in this repository.

## Release deployment

Published stable releases are deployed automatically by `.github/workflows/cd.yml`.
Release CD supplies the version and paths, builds the overlay on the Vivado
runner, and runs the same board validation on the protected PYNQ runner.

Generated overlays, packages, evidence, and credentials are never committed.
Use `matrix_multiplication.ipynb` for interactive testing.
