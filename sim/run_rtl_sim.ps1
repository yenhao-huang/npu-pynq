$ErrorActionPreference = "Stop"

$iverilogCommand = Get-Command iverilog -ErrorAction SilentlyContinue
$vvpCommand = Get-Command vvp -ErrorAction SilentlyContinue
$iverilogPath = if ($iverilogCommand) { $iverilogCommand.Source } else { $null }
$vvpPath = if ($vvpCommand) { $vvpCommand.Source } else { $null }

if (-not $iverilogPath -or -not $vvpPath) {
    $icarusBin = "C:\msys64\ucrt64\bin"
    $fallbackIverilogPath = Join-Path $icarusBin "iverilog.exe"
    $fallbackVvpPath = Join-Path $icarusBin "vvp.exe"
    if ((Test-Path -LiteralPath $fallbackIverilogPath -PathType Leaf) -and
        (Test-Path -LiteralPath $fallbackVvpPath -PathType Leaf)) {
        $iverilogPath = $fallbackIverilogPath
        $vvpPath = $fallbackVvpPath
        $env:PATH = "$icarusBin;$env:PATH"
    }
}

if (-not $iverilogPath -or -not $vvpPath) {
    throw "RTL simulation requires iverilog and vvp on PATH or under C:\msys64\ucrt64\bin. See skills/engineer/build-mac-npu-on-pynq-z1/references/installation/icarus-verilog-windows.md."
}

$simDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $simDir
$rtlPath = Join-Path $repoRoot "src\rtl\mac_npu\mac_unit.sv"
$testbenchPath = Join-Path $repoRoot "src\tb\mac_npu\tb_mac_unit.sv"
$outputPath = Join-Path $env:TEMP "mac_npu_tb.vvp"

& $iverilogPath -g2012 -s tb_mac_unit -o $outputPath $rtlPath $testbenchPath
if ($LASTEXITCODE -ne 0) {
    throw "iverilog compilation failed with exit code $LASTEXITCODE"
}

& $vvpPath $outputPath
if ($LASTEXITCODE -ne 0) {
    throw "RTL simulation failed with exit code $LASTEXITCODE"
}
