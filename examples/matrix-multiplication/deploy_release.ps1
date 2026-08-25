[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackagePath,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^v[0-9]+\.[0-9]+\.[0-9]+$')]
    [string]$ReleaseTag,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$DeploymentId,

    [Parameter(Mandatory = $true)]
    [string]$EvidencePath,

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$BoardHost = 'pynq_board',

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$BoardUser = 'xilinx',

    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string]$RemoteRoot = '/home/xilinx/jupyter_notebooks/npu_matrix',

    [switch]$DryRun,

    [switch]$InteractiveSudo
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-CommandAvailable {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command is unavailable: $Name"
    }
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

$resolvedPackage = (Resolve-Path -LiteralPath $PackagePath -ErrorAction Stop).Path
$packageItem = Get-Item -LiteralPath $resolvedPackage -ErrorAction Stop
if (-not $packageItem.PSIsContainer) {
    throw "PackagePath must be a directory: $PackagePath"
}

$requiredFiles = @(
    'run_on_board.py',
    'package.manifest.json',
    'runtime/matrix_multiplication.py',
    'src/runtime/npu.py',
    'src/runtime/verify_overlay.py',
    'artifacts/npu_matrix.bit',
    'artifacts/npu_matrix.hwh',
    'artifacts/npu_matrix.manifest.json'
)
foreach ($relativePath in $requiredFiles) {
    $candidate = Join-Path $resolvedPackage $relativePath
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Package is missing required file: $relativePath"
    }
}

$resolvedEvidence = [IO.Path]::GetFullPath($EvidencePath)
$remoteRootNormalized = $RemoteRoot.TrimEnd('/')
$remoteStaging = "$remoteRootNormalized/.staging/$ReleaseTag-$DeploymentId"
$remoteVersionRoot = "$remoteRootNormalized/releases/$ReleaseTag"
$remoteDeployment = "$remoteVersionRoot/$DeploymentId"
$remoteEvidence = "$remoteDeployment/board-evidence.json"
$target = "$BoardUser@$BoardHost"

Write-Host "Target: $target"
Write-Host "Package: $resolvedPackage"
Write-Host "Remote staging: $remoteStaging"
Write-Host "Remote deployment: $remoteDeployment"
Write-Host "Evidence: $resolvedEvidence"

if ($DryRun) {
    Write-Host 'Dry run complete; no archive or network command was executed.'
    exit 0
}

Assert-CommandAvailable -Name 'tar'
Assert-CommandAvailable -Name 'ssh'
Assert-CommandAvailable -Name 'scp'

$archiveName = "npu-matrix-$ReleaseTag-$DeploymentId.tar.gz"
$archivePath = Join-Path ([IO.Path]::GetTempPath()) $archiveName
if (Test-Path -LiteralPath $archivePath) {
    throw "Refusing to overwrite existing temporary archive: $archivePath"
}

$evidenceDirectory = Split-Path -Parent $resolvedEvidence
if ($evidenceDirectory) {
    New-Item -ItemType Directory -Force -Path $evidenceDirectory | Out-Null
}

try {
    Invoke-CheckedCommand -Command 'tar' -Arguments @(
        '-czf', $archivePath, '-C', $resolvedPackage, '.'
    )
    Invoke-CheckedCommand -Command 'ssh' -Arguments @(
        $target,
        "set -eu; mkdir -p '$remoteRootNormalized/.staging' '$remoteVersionRoot'; test ! -e '$remoteStaging'; test ! -e '$remoteDeployment'; mkdir '$remoteStaging'"
    )
    Invoke-CheckedCommand -Command 'scp' -Arguments @(
        '--', $archivePath, "${target}:$remoteStaging/package.tar.gz"
    )
    $sudoArguments = if ($InteractiveSudo) { '' } else { '-n ' }
    $validationCommand = "set -eu; tar -xzf '$remoteStaging/package.tar.gz' -C '$remoteStaging'; cd '$remoteStaging'; test -r /etc/profile.d/xrt_setup.sh; source /etc/profile.d/xrt_setup.sh; test -r /etc/profile.d/pynq_venv.sh; source /etc/profile.d/pynq_venv.sh; test -x /usr/local/share/pynq-venv/bin/python3; sudo ${sudoArguments}XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 run_on_board.py --artifact-dir artifacts --release-tag '$ReleaseTag' --evidence board-evidence.json; test -s board-evidence.json; mv '$remoteStaging' '$remoteDeployment'; ln -sfn '$DeploymentId' '$remoteVersionRoot/current'"
    $validationSshArguments = @()
    if ($InteractiveSudo) {
        $validationSshArguments += '-tt'
    }
    $validationSshArguments += @($target, $validationCommand)
    Invoke-CheckedCommand -Command 'ssh' -Arguments $validationSshArguments
    Invoke-CheckedCommand -Command 'scp' -Arguments @(
        '--', "${target}:$remoteEvidence", $resolvedEvidence
    )
}
finally {
    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath -Force
    }
}

Write-Host "PASS: deployed $ReleaseTag as $DeploymentId and retrieved board evidence"
