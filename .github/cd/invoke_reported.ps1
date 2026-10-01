<#
.SYNOPSIS
Run a CD script and, when it fails, publish its last output lines as an annotation.

.DESCRIPTION
The tail of ssh, scp and remote Python output is where a board failure explains
itself ("Could not resolve hostname", "a password is required", a traceback).
Raw job logs are not always readable where a release is driven from, but
annotations are, so a failed step names its cause rather than only its exit code.

.EXAMPLE
& .github/cd/invoke_reported.ps1 -Title 'board: matrix example' `
    -Script examples/matrix-multiplication/deploy_release.ps1 `
    -ArgumentList @('-PackagePath', 'build/cd-package', '-ReleaseTag', 'v1.0.6')
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Title,
    [Parameter(Mandatory = $true)][string]$Script,
    [string[]]$ArgumentList = @(),
    [ValidateRange(1, 200)][int]$TailLines = 15
)

$log = [IO.Path]::GetTempFileName()
try {
    # A child pwsh, so every stream, including native stderr, reaches the log.
    & pwsh -NoProfile -File $Script @ArgumentList *>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        $tail = Get-Content -LiteralPath $log -Tail $TailLines | ForEach-Object {
            ($_ -replace '%', '%25') -replace "`r", ''
        }
        Write-Host "::error title=$Title::$($tail -join '%0A')"
    }
    exit $code
}
finally {
    Remove-Item -LiteralPath $log -ErrorAction SilentlyContinue
}
