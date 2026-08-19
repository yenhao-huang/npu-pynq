[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$controller = Join-Path $repoRoot 'core\service\pynq_sync_controller.ps1'
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("pynq-sync-test-" + [guid]::NewGuid().ToString('N'))

try {
    $source = Join-Path $temporaryRoot 'source'
    $state = Join-Path $temporaryRoot 'state\manifest.json'
    $config = Join-Path $temporaryRoot 'config.json'
    New-Item -ItemType Directory -Path (Join-Path $source 'nested') -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $source 'nested\sample.txt') -Value 'pynq sync smoke test' -Encoding UTF8

    $configuration = [ordered]@{
        localRoot = $source
        remoteHost = '192.168.2.99'
        remoteUser = 'xilinx'
        remoteRoot = '/home/xilinx/jupyter_notebooks/pynq_z1_repo'
        intervalSeconds = 10
        statePath = $state
        exclude = @()
    }
    $configuration | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $config -Encoding UTF8

    $output = & $controller -ConfigPath $config -Once -DryRun -Force 2>&1 | Out-String
    if ($output -notmatch 'nested/sample\.txt') {
        throw "Expected relative file path was not planned:`n$output"
    }
    if ($output -notmatch '/home/xilinx/jupyter_notebooks/pynq_z1_repo/nested/sample\.txt') {
        throw "Expected remote file path was not planned:`n$output"
    }
    if (Test-Path -LiteralPath $state) {
        throw 'Dry run unexpectedly wrote a state manifest.'
    }

    Write-Host 'PYNQ sync controller smoke test: PASS'
}
finally {
    $tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    $resolvedTemporaryRoot = [IO.Path]::GetFullPath($temporaryRoot)
    if ($resolvedTemporaryRoot.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $resolvedTemporaryRoot)) {
        Remove-Item -LiteralPath $resolvedTemporaryRoot -Recurse -Force
    }
}
