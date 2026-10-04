<#
.SYNOPSIS
One-time setup: start this host's runner at every logon, as the signed-in user.

.DESCRIPTION
Use this instead of a Windows service when the account has no password (a
service cannot log on as a passwordless account), or when the CD jobs need the
user's own environment: the PATH that finds vivado, python and pwsh, the Vivado
licence, and the SSH keys that reach the PYNQ-Z1.

It registers a scheduled task that runs run.cmd at logon, restarts it if it
exits, and never times it out. If a runner service exists it is stopped and
disabled first, because two listeners for one registration fight over the
session. Run it once, from an elevated PowerShell. It never registers the runner
with GitHub and never handles a token.

.EXAMPLE
& register_runner_task.ps1 -RunnerRoot C:\actions-runner -WhatIf
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$RunnerRoot = 'C:\actions-runner',
    [string]$TaskName = 'GitHub Actions Runner',
    [string]$User = "$env:USERDOMAIN\$env:USERNAME"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$runCmd = Join-Path $RunnerRoot 'run.cmd'
if (-not (Test-Path -LiteralPath $runCmd -PathType Leaf)) {
    throw "No run.cmd under $RunnerRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $RunnerRoot '.runner') -PathType Leaf)) {
    throw "The runner under $RunnerRoot is not registered; run config.cmd first"
}
if ($RunnerRoot -like 'C:\Windows\*') {
    throw "Move the runner out of $RunnerRoot first: services and tasks started there fail with error 1053"
}

foreach ($service in @(Get-Service -Name 'actions.runner.*' -ErrorAction SilentlyContinue)) {
    if ($PSCmdlet.ShouldProcess($service.Name, 'Stop and disable the runner service')) {
        if ($service.Status -ne 'Stopped') { Stop-Service -Name $service.Name -Force }
        Set-Service -Name $service.Name -StartupType Disabled
        Write-Host "Disabled service: $($service.Name)"
    }
}

$action = New-ScheduledTaskAction -Execute $runCmd -WorkingDirectory $RunnerRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
# Scheduled tasks stop after 72 hours by default; a runner must not.
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

if ($PSCmdlet.ShouldProcess($TaskName, "Register logon task for $User")) {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Registered and started '$TaskName' for $User."
    Write-Host 'Leave the runner window open; minimize it. It returns at every logon.'
}
