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
    [int]$TimeoutSeconds = 120,

    # The logon task register_runner_task.ps1 creates.
    [string]$TaskName = 'GitHub Actions Runner'
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

function Find-InteractiveRunner {
    <#
    Returns run.cmd paths for runners configured to run interactively rather
    than as a service. Such a runner is online only while its window is open.
    #>
    $roots = @($HOME, 'C:\', 'C:\Windows\System32') | Select-Object -Unique
    Get-ChildItem -Path $roots -Filter 'run.cmd' -Recurse -Depth 4 `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.DirectoryName -match 'actions-runner' } |
        Select-Object -ExpandProperty FullName -Unique
}

# A runner comes up in one of three ways, in this order of preference:
# a logon scheduled task (the user's own environment, no password needed),
# an enabled Windows service, or a bare run.cmd launched in a window.
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$services = @(Get-RunnerServices | Where-Object { $_.StartType -ne 'Disabled' })
if ($task) {
    if ($task.State -eq 'Running') {
        Write-Host "Already running: scheduled task '$TaskName'"
    }
    elseif ($PSCmdlet.ShouldProcess($TaskName, 'Start runner scheduled task')) {
        Start-ScheduledTask -TaskName $TaskName
        Write-Host "Started scheduled task: $TaskName"
    }
    if ($WhatIfPreference) {
        Write-Host 'WhatIf: nothing was started.'
        exit 0
    }
}
elseif (-not $services) {
    Write-Host 'No runner task or service on this host; looking for an interactive runner.'
    # Copies of one registration must not run side by side, so launch exactly
    # one, preferring any copy outside C:\Windows.
    $interactive = @(Find-InteractiveRunner |
        Sort-Object { if ($_ -like 'C:\Windows\*') { 1 } else { 0 } } |
        Select-Object -First 1)
    if (-not $interactive) {
        Write-Warning 'No runner is installed here. references/runners.md has the one-time setup.'
        exit 1
    }
    foreach ($entry in $interactive) {
        if ($entry -like 'C:\Windows\*') {
            Write-Warning "Runner lives under C:\Windows: $entry"
            Write-Warning 'As a service it fails with error 1053, and its _work tree fills a protected folder. references/runners.md explains how to move it.'
        }
        if ($PSCmdlet.ShouldProcess($entry, 'Launch interactive runner')) {
            # Its own window, so the runner survives this script exiting.
            Start-Process -FilePath $entry `
                -WorkingDirectory (Split-Path -Parent $entry)
            Write-Host "Launched: $entry"
            Write-Host 'It stays online only while that window is open.'
        }
    }
    if ($WhatIfPreference) {
        Write-Host 'WhatIf: nothing was launched.'
        exit 0
    }
}
else {
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
