[CmdletBinding()]
param(
    [string]$ConfigPath,
    [switch]$Once,
    [switch]$DryRun,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot 'configs\pynq-sync.json'
}

function Resolve-ConfigPath {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$BasePath
    )

    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $BasePath $Value))
}

function ConvertTo-PosixSingleQuoted {
    param([Parameter(Mandatory = $true)][string]$Value)

    $singleQuote = [string][char]39
    $doubleQuote = [string][char]34
    $replacement = $singleQuote + $doubleQuote + $singleQuote + $doubleQuote + $singleQuote
    return $singleQuote + $Value.Replace($singleQuote, $replacement) + $singleQuote
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

function Write-SyncMessage {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
}

function Test-IsExcluded {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$Patterns
    )

    foreach ($pattern in $Patterns) {
        if ($RelativePath -like $pattern -or [IO.Path]::GetFileName($RelativePath) -like $pattern) {
            return $true
        }
    }
    return $false
}

function Read-Manifest {
    param([Parameter(Mandatory = $true)][string]$Path)

    $hashes = @{}
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $hashes
    }

    $state = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    if ($null -ne $state.files) {
        foreach ($property in $state.files.PSObject.Properties) {
            $hashes[$property.Name] = [string]$property.Value
        }
    }
    return $hashes
}

function Write-Manifest {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][hashtable]$Hashes,
        [Parameter(Mandatory = $true)][string]$LocalRoot,
        [Parameter(Mandatory = $true)][string]$Remote
    )

    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    $orderedHashes = [ordered]@{}
    foreach ($key in ($Hashes.Keys | Sort-Object)) {
        $orderedHashes[$key] = $Hashes[$key]
    }

    $document = [ordered]@{
        version = 1
        updatedAt = (Get-Date).ToString('o')
        localRoot = $LocalRoot
        remote = $Remote
        files = $orderedHashes
    }
    $json = $document | ConvertTo-Json -Depth 5
    $temporaryPath = "$Path.tmp"
    [IO.File]::WriteAllText($temporaryPath, $json, [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporaryPath -Destination $Path -Force
}

$resolvedConfigPath = (Resolve-Path -LiteralPath $ConfigPath -ErrorAction Stop).Path
$config = Get-Content -Raw -LiteralPath $resolvedConfigPath | ConvertFrom-Json

$requiredFields = @('localRoot', 'remoteHost', 'remoteUser', 'remoteRoot', 'intervalSeconds', 'statePath')
foreach ($field in $requiredFields) {
    if (-not $config.PSObject.Properties.Name.Contains($field)) {
        throw "Missing configuration field: $field"
    }
}

if ([string]$config.remoteHost -notmatch '^[A-Za-z0-9._-]+$') {
    throw 'remoteHost contains unsupported characters.'
}
if ([string]$config.remoteUser -notmatch '^[A-Za-z0-9._-]+$') {
    throw 'remoteUser contains unsupported characters.'
}

$remoteRoot = ([string]$config.remoteRoot).TrimEnd('/')
if (-not $remoteRoot.StartsWith('/') -or $remoteRoot -eq '/' -or $remoteRoot.Split('/') -contains '..') {
    throw 'remoteRoot must be a specific absolute path without parent traversal.'
}

$intervalSeconds = [int]$config.intervalSeconds
if ($intervalSeconds -lt 2) {
    throw 'intervalSeconds must be at least 2.'
}

$localRoot = Resolve-ConfigPath -Value ([string]$config.localRoot) -BasePath $repoRoot
$statePath = Resolve-ConfigPath -Value ([string]$config.statePath) -BasePath $repoRoot
if (-not (Test-Path -LiteralPath $localRoot -PathType Container)) {
    throw "Local sync root does not exist: $localRoot"
}

$excludePatterns = @()
if ($null -ne $config.exclude) {
    $excludePatterns = @($config.exclude | ForEach-Object { [string]$_ })
}

$target = "$($config.remoteUser)@$($config.remoteHost)"
$remoteLabel = "${target}:$remoteRoot"

if (-not $DryRun) {
    foreach ($command in @('ssh', 'scp')) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "Required command is unavailable: $command"
        }
    }
}

function Invoke-SyncPass {
    $previousHashes = Read-Manifest -Path $statePath
    $currentHashes = @{}
    $changed = [System.Collections.Generic.List[object]]::new()

    $files = Get-ChildItem -LiteralPath $localRoot -Recurse -File | Sort-Object FullName
    foreach ($file in $files) {
        $relativePath = $file.FullName.Substring($localRoot.Length).TrimStart('\', '/')
        $relativePath = $relativePath.Replace('\', '/')
        if (Test-IsExcluded -RelativePath $relativePath -Patterns $excludePatterns) {
            continue
        }

        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $currentHashes[$relativePath] = $hash
        if ($Force -or -not $previousHashes.ContainsKey($relativePath) -or $previousHashes[$relativePath] -ne $hash) {
            $changed.Add([pscustomobject]@{
                FullName = $file.FullName
                RelativePath = $relativePath
                Hash = $hash
            })
        }
    }

    if ($changed.Count -eq 0) {
        Write-SyncMessage "No changed files in $localRoot"
        if (-not $DryRun) {
            Write-Manifest -Path $statePath -Hashes $currentHashes -LocalRoot $localRoot -Remote $remoteLabel
        }
        return
    }

    Write-SyncMessage "Planning $($changed.Count) upload(s) to $remoteLabel"
    $createdRemoteDirectories = @{}
    foreach ($item in $changed) {
        $remoteFile = "$remoteRoot/$($item.RelativePath)"
        $remoteDirectory = $remoteRoot
        $relativeDirectory = [IO.Path]::GetDirectoryName($item.RelativePath)
        if ($relativeDirectory) {
            $remoteDirectory = "$remoteRoot/$($relativeDirectory.Replace('\', '/'))"
        }

        Write-SyncMessage "$($item.RelativePath) -> $remoteFile"
        if ($DryRun) {
            continue
        }

        if (-not $createdRemoteDirectories.ContainsKey($remoteDirectory)) {
            Invoke-CheckedCommand -Command 'ssh' -Arguments @(
                $target,
                "mkdir -p -- $(ConvertTo-PosixSingleQuoted $remoteDirectory)"
            )
            $createdRemoteDirectories[$remoteDirectory] = $true
        }

        Invoke-CheckedCommand -Command 'scp' -Arguments @(
            '--',
            $item.FullName,
            "${target}:$(ConvertTo-PosixSingleQuoted $remoteFile)"
        )
    }

    if ($DryRun) {
        Write-SyncMessage 'Dry run complete; no network commands or state writes were performed.'
        return
    }

    Write-Manifest -Path $statePath -Hashes $currentHashes -LocalRoot $localRoot -Remote $remoteLabel
    Write-SyncMessage "Sync pass complete; manifest updated at $statePath"
}

Write-SyncMessage "Controller source: $localRoot"
Write-SyncMessage "Controller target: $remoteLabel"
Write-SyncMessage "Mode: $(if ($DryRun) { 'dry-run' } elseif ($Once) { 'once' } else { "watch every $intervalSeconds seconds" })"

do {
    Invoke-SyncPass
    if ($Once -or $DryRun) {
        break
    }
    Start-Sleep -Seconds $intervalSeconds
} while ($true)
