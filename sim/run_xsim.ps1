$ErrorActionPreference = "Stop"

$vivadoBin = "C:\AMDDesignTools\2026.1\Vivado\bin"
$xvlog = Join-Path $vivadoBin "xvlog.bat"
$xelab = Join-Path $vivadoBin "xelab.bat"
$xsim = Join-Path $vivadoBin "xsim.bat"

foreach ($tool in @($xvlog, $xelab, $xsim)) {
    if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) {
        throw "Required Vivado simulator tool is missing: $tool"
    }
}

$simDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $simDir
$workDir = Join-Path $simDir "build\xsim"
New-Item -ItemType Directory -Path $workDir -Force | Out-Null

Push-Location $workDir
try {
    & $xvlog --sv (Join-Path $repoRoot "src\rtl\mac_npu\mac_unit.sv") (Join-Path $repoRoot "src\rtl\mac_npu\mac_axi_lite.sv") (Join-Path $repoRoot "src\tb\mac_npu\tb_mac_axi_lite.sv")
    if ($LASTEXITCODE -ne 0) { throw "xvlog failed with exit code $LASTEXITCODE" }
    & $xelab tb_mac_axi_lite -s mac_axi_lite_sim
    if ($LASTEXITCODE -ne 0) { throw "xelab failed with exit code $LASTEXITCODE" }
    & $xsim mac_axi_lite_sim -runall
    if ($LASTEXITCODE -ne 0) { throw "xsim failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
