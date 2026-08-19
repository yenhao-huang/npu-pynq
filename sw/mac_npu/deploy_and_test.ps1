$ErrorActionPreference = "Stop"

# Stage the board-runtime payload into mount/, then sync and smoke test.
# mount/ is the deploy staging tree mirrored to the board; sources live in sw/.

$swDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $swDir)
$configPath = Join-Path $repoRoot "configs\pynq-sync.json"
$syncController = Join-Path $repoRoot "core\service\pynq_sync_controller.ps1"
$stageDir = Join-Path $repoRoot "mount\mac_npu"

$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
$target = "$($config.remoteUser)@$($config.remoteHost)"
$remoteMacRoot = "$($config.remoteRoot.TrimEnd('/'))/mac_npu"

New-Item -ItemType Directory -Path $stageDir -Force | Out-Null
foreach ($name in @("mac_mmio.py", "hardware_smoke_test.py", "mac_npu_demo.ipynb")) {
    Copy-Item -LiteralPath (Join-Path $swDir $name) -Destination $stageDir -Force
}

& $syncController -Once
if ($LASTEXITCODE -ne 0) {
    throw "PYNQ sync failed with exit code $LASTEXITCODE"
}

& ssh $target "cd '$remoteMacRoot' && python3 hardware_smoke_test.py --bitfile overlay/mac_npu.bit"
if ($LASTEXITCODE -ne 0) {
    throw "PYNQ hardware smoke test failed with exit code $LASTEXITCODE"
}
