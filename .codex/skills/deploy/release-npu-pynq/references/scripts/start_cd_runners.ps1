<#
.SYNOPSIS
Bring this host's GitHub Actions self-hosted runners online for a CD release.

.DESCRIPTION
Starts every `actions.runner.*` Windows service on this host and waits for the
repository to report the runners online with the labels the CD jobs request.

It starts services only. It never registers, reconfigures, or removes a runner,
and it never handles a registration token. A runner that was configured to run
interactively has no service; the script reports the `run.cmd` it found so a
person can launch it.

.EXAMPLE
& start_cd_runners.ps1 -WhatIf
Shows which services would be started, and changes nothing.

.EXAMPLE
& start_cd_runners.ps1
Starts the services and waits for the runners to report online.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    # Labels the CD jobs request. A release needs all of them present online.
    [string[]]$RequiredLabels = @('vivado', 'pynq-z1'),

    # Seconds to wait for the runners to report online after starting them.
    [ValidateRange(0, 600)]
    [int]$TimeoutSeconds = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-RunnerServices {
    Get-Service -Name 'actions.runner.*' -ErrorAction SilentlyContinue
}

function Get-RunnerState {
    <#
    Returns the repository's runners, or $null when the state cannot be read.
    Reading runner state needs repository administration access; a 403 is a
    permission answer, not a missing runner.
    #>
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        Write-Warning 'gh is unavailable; cannot confirm runner state from the repository.'
        return $null
    }
    $raw = & gh api 'repos/:owner/:repo/actions/runners' 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $raw) {
        Write-Warning 'Cannot read runner state (repository administration access is needed).'
        return $null
    }
    try {
        return ($raw | ConvertFrom-Json).runners
    }
    catch {
        Write-Warning "Runner state was not valid JSON: $($_.Exception.Message)"
        return $null
    }
}

function Show-RunnerState {
    param($Runners)
    if (-not $Runners) {
        Write-Host '  (none reported)'
        return
    }
    foreach ($runner in $Runners) {
        $labels = ($runner.labels | ForEach-Object { $_.name }) -join ', '
        $busy = if ($runner.busy) { 'busy' } else { 'idle' }
        Write-Host "  $($runner.name): $($runner.status), $busy [$labels]"
    }
}

function Test-LabelsOnline {
    param($Runners, [string[]]$Labels)
    if (-not $Runners) { return $false }
    $online = $Runners | Where-Object { $_.status -eq 'online' }
    if (-not $online) { return $false }
    $available = @()
    foreach ($runner in $online) {
        $available += ($runner.labels | ForEach-Object { $_.name })
    }
    foreach ($label in $Labels) {
        if ($available -notcontains $label) { return $false }
    }
    return $true
}

Write-Host 'Runner state before starting:'
Show-RunnerState -Runners (Get-RunnerState)

$services = Get-RunnerServices
if (-not $services) {
    Write-Warning 'No actions.runner.* service exists on this host.'
    $interactive = Get-ChildItem -Path $HOME, 'C:\' -Filter 'run.cmd' -Depth 3 `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.DirectoryName -match 'actions-runner' } |
        Select-Object -First 3
    if ($interactive) {
        Write-Host 'Found an interactively configured runner. Launch it in its own window:'
        foreach ($entry in $interactive) {
            Write-Host "  $($entry.FullName)"
        }
        Write-Host 'It stays online only while that window is open.'
    }
    else {
        Write-Host 'No runner is installed here. references/runners.md has the one-time setup.'
    }
    exit 1
}

foreach ($service in $services) {
    if ($service.Status -eq 'Running') {
        Write-Host "Already running: $($service.Name)"
        continue
    }
    if ($PSCmdlet.ShouldProcess($service.Name, 'Start runner service')) {
        Start-Service -Name $service.Name
        Write-Host "Started: $($service.Name)"
    }
}

if ($WhatIfPreference) {
    Write-Host 'WhatIf: no service was started and no state was changed.'
    exit 0
}

# A service reports Running before the runner has registered as online, so wait
# on the repository's view rather than on the service.
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$runners = Get-RunnerState
while (-not (Test-LabelsOnline -Runners $runners -Labels $RequiredLabels)) {
    if ($null -eq $runners) {
        # State is unreadable; the services are running, which is all this host
        # can prove. Report it rather than blocking the release.
        Write-Warning 'Runner services are running, but their online state could not be confirmed.'
        Write-Warning 'Check Settings -> Actions -> Runners before pushing the release branch.'
        exit 0
    }
    if ((Get-Date) -gt $deadline) {
        Write-Host 'Runner state at timeout:'
        Show-RunnerState -Runners $runners
        throw "Runners did not report online with all of: $($RequiredLabels -join ', ')"
    }
    Start-Sleep -Seconds 5
    $runners = Get-RunnerState
}

Write-Host 'Runner state now:'
Show-RunnerState -Runners $runners
Write-Host "PASS: runners are online with $($RequiredLabels -join ', ')"
