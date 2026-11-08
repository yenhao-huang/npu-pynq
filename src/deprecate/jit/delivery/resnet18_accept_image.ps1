<#
.SYNOPSIS
Classify the pinned demo photograph on the physical PYNQ-Z1 and collect evidence.

.DESCRIPTION
Runs against the deployment `resnet18_deploy.ps1` already promoted, so the
image acceptance uses exactly the package CD built. It never deploys, never
falls back to a host runtime, and retrieves the evidence only after the board
has written it.

The inference takes tens of minutes, longer than an idle ssh session reliably
survives, so it starts detached on the board (nohup + setsid) and this script
polls for its exit status with short connections. A dropped connection costs
one poll, not the run.
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

    [ValidateRange(1, 1440)]
    [int]$TimeoutMinutes = 240,

    [ValidateRange(5, 600)]
    [int]$PollSeconds = 30,

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

$sshOptions = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4')
$remoteDeployment = "$remoteBase/releases/$DeploymentId"
$log = "$remoteDeployment/image-acceptance.log"
$exitFile = "$remoteDeployment/image-acceptance.exit"

# The job script lives beside the package, not in it, so the package tree stays
# exactly what was built. Base64 avoids every layer of quoting.
$job = @(
    'set -u'
    "cd '$remotePackage'"
    'source /etc/profile.d/xrt_setup.sh'
    'source /etc/profile.d/pynq_venv.sh'
    'code=0'
    'sudo -n XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -B accept_image_on_board.py --package-root . --evidence image-acceptance.json || code=$?'
    'if [ -f image-acceptance.json ]; then sudo -n chmod 0644 image-acceptance.json || code=1; fi'
    "echo `$code > '$exitFile.tmp' && mv '$exitFile.tmp' '$exitFile'"
) -join "`n"
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($job + "`n"))
$start = @(
    'set -eu'
    "cd '$remotePackage'"
    'test -r package.manifest.json'
    'test -r /etc/profile.d/xrt_setup.sh'
    'test -r /etc/profile.d/pynq_venv.sh'
    'test -x /usr/local/share/pynq-venv/bin/python3'
    "test ! -e '$exitFile'"
    "echo $encoded | base64 -d > '$remoteDeployment/image-acceptance.sh'"
    # The trailing '&' already ends the command, so no '; ' may follow it.
    "nohup setsid bash '$remoteDeployment/image-acceptance.sh' > '$log' 2>&1 < /dev/null & echo started"
) -join '; '
Invoke-CheckedCommand -Command 'ssh' -Arguments ($sshOptions + @($target, $start))

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$failures = 0
$status = $null
$lastReport = Get-Date
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds $PollSeconds
    $answer = & ssh @sshOptions $target "cat '$exitFile' 2>/dev/null || echo running" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $failures += 1
        Write-Host "poll failed (ssh exit $LASTEXITCODE); $failures in a row"
        if ($failures -ge 20) {
            throw "lost the board for $failures polls in a row"
        }
        continue
    }
    $failures = 0
    $answer = "$answer".Trim()
    if ($answer -ne 'running') {
        $status = $answer
        break
    }
    if (((Get-Date) - $lastReport).TotalMinutes -ge 5) {
        $lastReport = Get-Date
        & ssh @sshOptions $target "tail -n 3 '$log'" 2>$null | ForEach-Object { Write-Host "  board: $_" }
    }
}
if ($null -eq $status) {
    throw "image acceptance did not finish within $TimeoutMinutes minutes"
}

# The board log's tail is what a failure annotation shows.
& ssh @sshOptions $target "tail -n 40 '$log'" 2>$null | ForEach-Object { Write-Host $_ }
if ($status -ne '0') {
    throw "image acceptance failed on the board (exit $status)"
}

$evidenceDirectory = Split-Path -Parent $resolvedEvidence
if ($evidenceDirectory) {
    New-Item -ItemType Directory -Force -Path $evidenceDirectory | Out-Null
}
Invoke-CheckedCommand -Command 'scp' -Arguments @(
    '-o', 'BatchMode=yes',
    '--',
    "${target}:$remotePackage/image-acceptance.json",
    $resolvedEvidence
)
Write-Host "PASS [physical-pynq-z1]: image acceptance evidence at $resolvedEvidence"
