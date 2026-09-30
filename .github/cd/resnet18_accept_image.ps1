<#
.SYNOPSIS
Classify the pinned demo photograph on the physical PYNQ-Z1 and collect evidence.

.DESCRIPTION
Runs against the deployment `resnet18_deploy_and_accept.ps1` already promoted,
so the image acceptance uses exactly the package the board accepted. It never
deploys, never falls back to a host runtime, and retrieves the evidence only
after the board has written it.
#>
[CmdletBinding()]
param(
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

$target = "$BoardUser@$BoardHost"
$remoteBase = $RemoteRoot.TrimEnd('/')
$remotePackage = "$remoteBase/releases/$DeploymentId/package"
$resolvedEvidence = [IO.Path]::GetFullPath($EvidencePath)

Write-Host "Release tag : $ReleaseTag"
Write-Host "Package     : $target`:$remotePackage"

if ($DryRun) {
    Write-Host 'PASS: acceptance inputs verified; no network command was executed'
    exit 0
}

Assert-CommandAvailable -Name 'ssh'
Assert-CommandAvailable -Name 'scp'

$remoteCommand = @(
    "set -eu",
    "cd '$remotePackage'",
    "test -r package.manifest.json",
    "test -r /etc/profile.d/xrt_setup.sh",
    "source /etc/profile.d/xrt_setup.sh",
    "test -r /etc/profile.d/pynq_venv.sh",
    "source /etc/profile.d/pynq_venv.sh",
    "test -x /usr/local/share/pynq-venv/bin/python3",
    "sudo -n XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -B accept_image_on_board.py --package-root . --evidence image-acceptance.json",
    "sudo -n chmod 0644 image-acceptance.json"
) -join '; '
Invoke-CheckedCommand -Command 'ssh' -Arguments @($target, $remoteCommand)

$evidenceDirectory = Split-Path -Parent $resolvedEvidence
if ($evidenceDirectory) {
    New-Item -ItemType Directory -Force -Path $evidenceDirectory | Out-Null
}
Invoke-CheckedCommand -Command 'scp' -Arguments @(
    '--',
    "${target}:$remotePackage/image-acceptance.json",
    $resolvedEvidence
)
Write-Host "PASS [physical-pynq-z1]: image acceptance evidence at $resolvedEvidence"
