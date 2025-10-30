[CmdletBinding()]
param(
    [switch]$AllowNetwork
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot/common.ps1"

$repoRoot = Get-RepoRoot
$logDir = Join-Path $repoRoot "artifacts/logs"
Ensure-Directory $logDir
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logDir "test-$timestamp.log"
Start-Transcript -Path $logPath -Append | Out-Null

try {
    $venvPython = Get-VenvPython
    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment missing. Run scripts/bootstrap.ps1 first."
    }

    if ($AllowNetwork) {
        [Environment]::SetEnvironmentVariable('ALLOW_NETWORK', '1')
    }

    Write-Step "Running pytest"
    Push-Location $repoRoot
    try {
        & $venvPython -m pytest -q
    }
    finally {
        Pop-Location
    }

    Write-Step "Tests completed"
}
finally {
    Stop-Transcript | Out-Null
}
