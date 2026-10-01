<#
.SYNOPSIS
Deploy one ResNet-18 release package to the PYNQ-Z1.

.DESCRIPTION
This is the automated CD deployer. It transfers exactly the archive CD built,
proves on the board that the extracted tree is that archive (digest and file
list, no FPGA work), and promotes it to releases/<deployment>/package. The
board check is the real-image inference, run next by resnet18_accept_image.ps1
against this deployment.
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

# The board proves the extracted tree against the archive digest. This runs as
# the board user and touches no hardware.
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
    "/usr/local/share/pynq-venv/bin/python3 -B run_on_board.py --package-root . --package-archive '../$archiveName' --archive-sha256 '$archiveDigest' --release-tag '$ReleaseTag' --verify-only",
    "rm -f '../$archiveName'"
) -join '; '
Invoke-CheckedCommand -Command 'ssh' -Arguments @($target, $remoteCommand)

# Promotion is atomic and happens only after the tree was proved.
Invoke-CheckedCommand -Command 'ssh' -Arguments @(
    $target,
    "set -eu; mkdir -p '$remoteBase/releases'; mv '$remoteStaging' '$remoteDeployment'; ln -sfn '$remoteDeployment/package' '$remoteBase/current'"
)
Write-Host "PASS: deployed $ReleaseTag to $target`:$remoteDeployment/package (deploy only)"
