<#
.SYNOPSIS
Deploy one published ResNet-18 release package to the PYNQ-Z1 and accept it.

.DESCRIPTION
This is the automated CD deployer. It transfers exactly the archive that was
published with the Release, proves the deployed tree against that archive's
digest on the board, runs the acceptance corpus on the physical overlay, and
retrieves the evidence. It promotes the immutable deployment directory only
after the board run has written readable evidence.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackageArchive,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^(v[0-9]+\.[0-9]+\.[0-9]+|local-[0-9a-fA-F]{8,64})$')]
    [string]$ReleaseTag,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$DeploymentId,

    [Parameter(Mandatory = $true)]
    [string]$EvidencePath,

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$BoardHost = '192.168.2.99',

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$BoardUser = 'xilinx',

    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string]$RemoteRoot = '/home/xilinx/jupyter_notebooks/npu_resnet18',

    [switch]$DryRun
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

$archive = (Resolve-Path -LiteralPath $PackageArchive).Path
if ([IO.Path]::GetExtension($archive).ToLowerInvariant() -ne '.zip') {
    throw "Release package must be a .zip archive: $archive"
}
$archiveName = [IO.Path]::GetFileName($archive)
if ($archiveName -notmatch '^[A-Za-z0-9._-]+$') {
    throw "Release package filename must be a plain basename: $archiveName"
}
$archiveDigest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
$resolvedEvidence = [IO.Path]::GetFullPath($EvidencePath)

$target = "$BoardUser@$BoardHost"
$remoteBase = $RemoteRoot.TrimEnd('/')
$remoteStaging = "$remoteBase/staging/$DeploymentId"
$remoteDeployment = "$remoteBase/releases/$DeploymentId"

Write-Host "Release package : $archiveName"
Write-Host "Archive SHA-256 : $archiveDigest"
Write-Host "Release tag     : $ReleaseTag"
Write-Host "Deployment      : $target`:$remoteDeployment"

if ($DryRun) {
    Write-Host 'PASS: deployment inputs verified; no network command was executed'
    exit 0
}

Assert-CommandAvailable -Name 'ssh'
Assert-CommandAvailable -Name 'scp'

Invoke-CheckedCommand -Command 'ssh' -Arguments @(
    $target,
    "set -eu; test ! -e '$remoteDeployment'; rm -rf '$remoteStaging'; mkdir -p '$remoteStaging'"
)
Invoke-CheckedCommand -Command 'scp' -Arguments @(
    '--', $archive, "${target}:$remoteStaging/$archiveName"
)

# The board proves the extracted tree against the published archive digest,
# runs the acceptance corpus on the physical overlay, and writes evidence.
$remoteCommand = @(
    "set -eu",
    "cd '$remoteStaging'",
    "python3 -m zipfile -e '$archiveName' package",
    "test -r /etc/profile.d/xrt_setup.sh",
    "source /etc/profile.d/xrt_setup.sh",
    "test -r /etc/profile.d/pynq_venv.sh",
    "source /etc/profile.d/pynq_venv.sh",
    "test -x /usr/local/share/pynq-venv/bin/python3",
    "cd package",
    "sudo -n XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -B run_on_board.py --package-root . --package-archive '../$archiveName' --archive-sha256 '$archiveDigest' --release-tag '$ReleaseTag' --evidence board-evidence.json",
    "sudo -n chmod 0644 board-evidence.json"
) -join '; '
Invoke-CheckedCommand -Command 'ssh' -Arguments @($target, $remoteCommand)

# Promotion is atomic and happens only after readable evidence exists.
Invoke-CheckedCommand -Command 'ssh' -Arguments @(
    $target,
    "set -eu; test -r '$remoteStaging/package/board-evidence.json'; mkdir -p '$remoteBase/releases'; mv '$remoteStaging' '$remoteDeployment'; ln -sfn '$remoteDeployment/package' '$remoteBase/current'"
)

$evidenceDirectory = Split-Path -Parent $resolvedEvidence
if ($evidenceDirectory) {
    New-Item -ItemType Directory -Force -Path $evidenceDirectory | Out-Null
}
Invoke-CheckedCommand -Command 'scp' -Arguments @(
    '--',
    "${target}:$remoteDeployment/package/board-evidence.json",
    $resolvedEvidence
)
Write-Host "PASS [physical-pynq-z1]: board evidence downloaded to $resolvedEvidence"
