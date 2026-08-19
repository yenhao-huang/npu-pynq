$ErrorActionPreference = "Stop"

$vivado = "C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat"
if (-not (Test-Path -LiteralPath $vivado -PathType Leaf)) {
    throw "Vivado 2026.1 was not found at $vivado"
}

$scriptPath = Join-Path $PSScriptRoot "build_overlay.tcl"
& $vivado -mode batch -source $scriptPath -nojournal -nolog
if ($LASTEXITCODE -ne 0) {
    throw "Vivado overlay build failed with exit code $LASTEXITCODE"
}
