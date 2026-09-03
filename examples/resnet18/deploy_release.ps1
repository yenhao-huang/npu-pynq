[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackageArchive,

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
    [string]$RemoteRoot = '/home/xilinx/jupyter_notebooks/npu_resnet18',

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$RollbackDeploymentId,

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

$resolvedArchive = (Resolve-Path -LiteralPath $PackageArchive -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $resolvedArchive -PathType Leaf)) {
    throw "PackageArchive must be a file: $PackageArchive"
}
if ([IO.Path]::GetExtension($resolvedArchive) -ne '.zip') {
    throw 'PackageArchive must be a .zip file'
}
$archiveDigest = (Get-FileHash -LiteralPath $resolvedArchive -Algorithm SHA256).Hash.ToLowerInvariant()
$resolvedEvidence = [IO.Path]::GetFullPath($EvidencePath)
$remoteRootNormalized = $RemoteRoot.TrimEnd('/')
$remoteStaging = "$remoteRootNormalized/.staging/$ReleaseTag-$DeploymentId"
$remoteDeployment = "$remoteRootNormalized/releases/$ReleaseTag/$DeploymentId"
$target = "$BoardUser@$BoardHost"

Write-Host "Target: $target"
Write-Host "Archive SHA-256: $archiveDigest"

if ($DryRun) {
    $temporary = Join-Path ([IO.Path]::GetTempPath()) ("npu-resnet18-verify-" + [guid]::NewGuid())
    try {
        Expand-Archive -LiteralPath $resolvedArchive -DestinationPath $temporary
        Invoke-CheckedCommand -Command 'python' -Arguments @(
            (Join-Path $temporary 'run_on_board.py'),
            '--package-root', $temporary,
            '--archive', $resolvedArchive,
            '--archive-sha256', $archiveDigest,
            '--verify-only'
        )
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Recurse -Force
        }
    }
    Write-Host 'Dry run complete; no network command was executed.'
    exit 0
}

Assert-CommandAvailable -Name 'ssh'
Assert-CommandAvailable -Name 'scp'

if ($RollbackDeploymentId) {
    $rollback = "$remoteRootNormalized/releases/$ReleaseTag/$RollbackDeploymentId"
    $command = "set -eu; test -s '$rollback/package/board-evidence.json'; test -f '$rollback/package/package.manifest.json'; ln -sfn '$RollbackDeploymentId' '$remoteRootNormalized/releases/$ReleaseTag/current'"
    Invoke-CheckedCommand -Command 'ssh' -Arguments @($target, $command)
    Write-Host "PASS: selected verified deployment $RollbackDeploymentId"
    exit 0
}

$evidenceDirectory = Split-Path -Parent $resolvedEvidence
if ($evidenceDirectory) {
    New-Item -ItemType Directory -Force -Path $evidenceDirectory | Out-Null
}
$archiveName = 'package.zip'
Invoke-CheckedCommand -Command 'ssh' -Arguments @(
    $target,
    "set -eu; mkdir -p '$remoteRootNormalized/.staging' '$remoteRootNormalized/releases/$ReleaseTag'; test ! -e '$remoteStaging'; test ! -e '$remoteDeployment'; mkdir '$remoteStaging'"
)
Invoke-CheckedCommand -Command 'scp' -Arguments @(
    '--', $resolvedArchive, "${target}:$remoteStaging/$archiveName"
)
$sudoArguments = if ($InteractiveSudo) { '' } else { '-n ' }
$remoteCommand = "set -eu; mkdir '$remoteStaging/package'; python3 -m zipfile -e '$remoteStaging/$archiveName' '$remoteStaging/package'; cd '$remoteStaging/package'; test -r /etc/profile.d/xrt_setup.sh; source /etc/profile.d/xrt_setup.sh; test -r /etc/profile.d/pynq_venv.sh; source /etc/profile.d/pynq_venv.sh; test -x /usr/local/share/pynq-venv/bin/python3; sudo ${sudoArguments}XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 run_on_board.py --package-root . --archive '../$archiveName' --archive-sha256 '$archiveDigest' --evidence board-evidence.json; sudo ${sudoArguments}chmod 0644 board-evidence.json; test -s board-evidence.json; cd '$remoteRootNormalized'; mv '$remoteStaging' '$remoteDeployment'; ln -sfn '$DeploymentId' '$remoteRootNormalized/releases/$ReleaseTag/current'"
$sshArguments = @()
if ($InteractiveSudo) {
    $sshArguments += '-tt'
}
$sshArguments += @($target, $remoteCommand)
Invoke-CheckedCommand -Command 'ssh' -Arguments $sshArguments
Invoke-CheckedCommand -Command 'scp' -Arguments @(
    '--', "${target}:$remoteDeployment/package/board-evidence.json", $resolvedEvidence
)
Write-Host "PASS: deployed $ReleaseTag as immutable $DeploymentId"
