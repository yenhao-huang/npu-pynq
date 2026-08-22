[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BitPath,

    [Parameter(Mandatory = $true)]
    [string]$HwhPath,

    [string[]]$ExtraPath = @(),

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$OverlayName,

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$BoardHost = 'pynq_board',

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$BoardUser = 'xilinx',

    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string]$RemoteRoot = '/home/xilinx/jupyter_notebooks',

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-InputFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedExtension
    )

    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    $item = Get-Item -LiteralPath $resolved -ErrorAction Stop
    if (-not $item.PSIsContainer -and $item.Extension -ieq $ExpectedExtension) {
        return $item.FullName
    }

    throw "Expected a $ExpectedExtension file: $Path"
}

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

$resolvedBit = Resolve-InputFile -Path $BitPath -ExpectedExtension '.bit'
$resolvedHwh = Resolve-InputFile -Path $HwhPath -ExpectedExtension '.hwh'

if (-not $OverlayName) {
    $OverlayName = [IO.Path]::GetFileNameWithoutExtension($resolvedBit) -replace '_wrapper$', ''
}
if ($OverlayName -notmatch '^[A-Za-z0-9._-]+$') {
    throw 'OverlayName may contain only letters, digits, dot, underscore, and hyphen.'
}

$resolvedExtras = foreach ($path in $ExtraPath) {
    $resolved = (Resolve-Path -LiteralPath $path -ErrorAction Stop).Path
    $item = Get-Item -LiteralPath $resolved -ErrorAction Stop
    if ($item.PSIsContainer) {
        throw "ExtraPath must be a file: $path"
    }
    if ($item.Name -notmatch '^[A-Za-z0-9._-]+$') {
        throw "Extra filename contains unsupported characters: $($item.Name)"
    }
    $item.FullName
}

$remoteDirectory = "$($RemoteRoot.TrimEnd('/'))/$OverlayName"
$target = "$BoardUser@$BoardHost"
$transfers = @(
    [pscustomobject]@{ Source = $resolvedBit; Destination = "$remoteDirectory/$OverlayName.bit" }
    [pscustomobject]@{ Source = $resolvedHwh; Destination = "$remoteDirectory/$OverlayName.hwh" }
)
foreach ($extra in $resolvedExtras) {
    $transfers += [pscustomobject]@{
        Source = $extra
        Destination = "$remoteDirectory/$([IO.Path]::GetFileName($extra))"
    }
}

Write-Host "Target: $target"
Write-Host "Remote directory: $remoteDirectory"
foreach ($transfer in $transfers) {
    Write-Host "Upload: $($transfer.Source) -> $($transfer.Destination)"
}

if ($DryRun) {
    Write-Host 'Dry run complete; no network commands were executed.'
    exit 0
}

Assert-CommandAvailable -Name 'ssh'
Assert-CommandAvailable -Name 'scp'

Invoke-CheckedCommand -Command 'ssh' -Arguments @(
    $target,
    "mkdir -p -- '$remoteDirectory'"
)
foreach ($transfer in $transfers) {
    Invoke-CheckedCommand -Command 'scp' -Arguments @(
        '--',
        $transfer.Source,
        "${target}:$($transfer.Destination)"
    )
}
Invoke-CheckedCommand -Command 'ssh' -Arguments @(
    $target,
    "ls -lh -- '$remoteDirectory'"
)

Write-Host "Upload verified: ${target}:$remoteDirectory"
