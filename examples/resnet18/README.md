# ResNet-18 acceptance delivery

This example packages an externally supplied quantized ResNet-18 model and
validation corpus with a trusted PYNQ-Z1 overlay. Generated packages, model
weights, corpora, Vivado reports, bitstreams, and evidence remain outside Git.

Build a deterministic archive from a clean trusted Vivado result:

```powershell
python examples/resnet18/package_example.py `
  --artifact-dir build/vivado/npu_matrix/artifacts `
  --report-dir build/vivado/npu_matrix/reports `
  --descriptor $env:NPU_RESNET18_DESCRIPTOR `
  --output-archive mount/resnet18/npu-resnet18.zip `
  --release-tag v0.0.0 `
  --source-commit (git rev-parse HEAD)
```

Validate deployment without a network operation, then run on the protected
board runner:

```powershell
& examples/resnet18/deploy_release.ps1 `
  -PackageArchive mount/resnet18/npu-resnet18.zip `
  -ReleaseTag v0.0.0 `
  -DeploymentId (Get-Date -Format yyyyMMdd-HHmmss) `
  -EvidencePath build/board/resnet18-evidence.json `
  -DryRun
```

Host fixture output is software-integration evidence only. Phase 2B is not
complete until the trusted Vivado and protected PYNQ-Z1 runs produce the
provenance-bound board evidence.

## Notebook demo

The deterministic archive includes `resnet18.ipynb`. Open it from the extracted
package root on the PYNQ-Z1 and run all cells to validate the packaged overlay,
load the model through the public runtime, execute the acceptance corpus twice,
and inspect accuracy, latency, work, and physical-cycle metrics. The notebook
does not access MMIO or DMA channels directly.
