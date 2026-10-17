<# Deploy immutable SmolLM assets, run detached physical acceptance, retain evidence. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ModelArchive,
    [Parameter(Mandatory)][string]$OverlayDirectory,
    [Parameter(Mandatory)][ValidatePattern('^[0-9a-f]{40}$')][string]$SourceCommit,
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9._-]+$')][string]$ReleaseTag,
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9._-]+$')][string]$DeploymentId,
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9._-]+$')][string]$BoardHost,
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9._-]+$')][string]$BoardUser,
    [Parameter(Mandatory)][ValidatePattern('^/[A-Za-z0-9._/-]+$')][string]$RemoteRoot,
    [Parameter(Mandatory)][string]$EvidencePath,
    [ValidateRange(1,240)][int]$TimeoutMinutes = 60,
    [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Checked([string]$Command, [string[]]$Arguments) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed ($LASTEXITCODE)" }
}
$repo = (Resolve-Path "$PSScriptRoot/../..").Path
$model = (Resolve-Path -LiteralPath $ModelArchive).Path
$overlay = (Resolve-Path -LiteralPath $OverlayDirectory).Path
foreach ($name in 'npu_matrix.bit','npu_matrix.hwh','npu_matrix.manifest.json') {
    if (-not (Test-Path -LiteralPath "$overlay/$name" -PathType Leaf)) { throw "Missing $name" }
}
Checked python @("$repo/src/runtime/verify_overlay.py", $overlay)
$remote = $RemoteRoot.TrimEnd('/') + "/releases/$ReleaseTag-$DeploymentId"
if ($remote -match '/\.\.?(/|$)') { throw 'Remote root cannot contain traversal' }
Write-Host "SmolLM $SourceCommit -> ${BoardUser}@${BoardHost}:$remote"
if ($DryRun) { Write-Host 'PASS: inputs verified; no deployment started'; exit 0 }
$target = "$BoardUser@$BoardHost"
$ssh = @('-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4')
$stage = "$repo/build/smollm-deploy/$DeploymentId"
if (Test-Path -LiteralPath $stage) { throw "Stage already exists: $stage" }
New-Item -ItemType Directory -Path "$stage/overlay" -Force | Out-Null
Copy-Item -LiteralPath $model -Destination "$stage/model.tgz"
Copy-Item -LiteralPath "$overlay/npu_matrix.bit","$overlay/npu_matrix.hwh","$overlay/npu_matrix.manifest.json" -Destination "$stage/overlay"
Checked tar @('-czf',"$stage/payload.tgz",'-C',$repo,'--exclude=__pycache__','--exclude=*.pyc',
    'src','.github/cd/smollm.py','-C',$stage,'model.tgz','overlay')
$hash = (Get-FileHash -LiteralPath "$stage/payload.tgz" -Algorithm SHA256).Hash.ToLowerInvariant()
# No existing deployment is overwritten and no active pointer is changed.
Checked ssh ($ssh + @($target,"mkdir -p '$RemoteRoot/releases' && mkdir '$remote'"))
Checked scp ($ssh + @("$stage/payload.tgz","${target}:$remote/payload.tgz"))
$job = @(
    'set -u'
    "cd '$remote'"
    'code=0'
    "sudo -n XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 -B .github/cd/smollm.py accept --package model --overlay overlay/npu_matrix.bit --commit '$SourceCommit' --tag '$ReleaseTag' --deployment-id '$DeploymentId' --evidence interactive.json || code=`$?"
    'if [ -f interactive.json ]; then sudo -n chmod 0644 interactive.json || code=1; fi'
    'echo "$code" > exit.tmp && mv exit.tmp exit'
) -join "`n"
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($job + "`n"))
$start = @(
    'set -eu'
    "cd '$remote'"
    "echo '$hash  payload.tgz' | sha256sum --check --strict"
    'tar -xzf payload.tgz'
    'mkdir model && tar -xzf model.tgz -C model'
    "/usr/local/share/pynq-venv/bin/python3 -B .github/cd/smollm.py verify --package model --commit '$SourceCommit' --tag '$ReleaseTag'"
    "echo '$encoded' | base64 -d > job.sh"
    'nohup setsid bash job.sh > board.log 2>&1 < /dev/null & echo started'
) -join '; '
Checked ssh ($ssh + @($target,$start))
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$status = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 20
    $answer = & ssh @ssh $target "cat '$remote/exit' 2>/dev/null || echo running" 2>$null
    if ($LASTEXITCODE -ne 0) { continue }
    if ("$answer".Trim() -ne 'running') { $status = "$answer".Trim(); break }
}
$evidence = [IO.Path]::GetFullPath($EvidencePath)
New-Item -ItemType Directory -Force -Path (Split-Path $evidence -Parent) | Out-Null
& ssh @ssh $target "tail -n 40 '$remote/board.log'" 2>$null | ForEach-Object { Write-Host $_ }
& scp @ssh "${target}:$remote/interactive.json" $evidence
$retrieved = $LASTEXITCODE -eq 0
if ($null -eq $status) { throw 'SmolLM acceptance timed out; inspect the retained board job' }
if ($status -ne '0' -or -not $retrieved) { throw "SmolLM acceptance failed (exit $status)" }
$record = Get-Content -LiteralPath $evidence -Raw | ConvertFrom-Json
if (-not $record.success -or $record.source_commit -ne $SourceCommit -or $record.deployment_id -ne $DeploymentId) {
    throw 'SmolLM evidence identity or success gate failed'
}
Write-Host "PASS [physical-pynq-z1]: 32 token IDs and text match; evidence $evidence"
