param(
    [string]$Python = "python",
    [string]$ModelPrefix = "examples/resnet18/model/resnet18",
    [string]$Checkpoint = "examples/resnet18/model/resnet18-f37072fd.pth",
    [string]$Evidence = "examples/resnet18/model/acceptance.json"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
Push-Location $repositoryRoot
try {
    & $Python examples/resnet18/scripts/verify_model.py `
        --model-prefix $ModelPrefix `
        --checkpoint $Checkpoint `
        --output $Evidence
    if ($LASTEXITCODE -ne 0) {
        throw "ResNet-18 real-model host validation failed"
    }
}
finally {
    Pop-Location
}
