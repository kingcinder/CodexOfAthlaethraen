[CmdletBinding()]
param(
    [int]$Tail = 50
)

. "$PSScriptRoot/common.ps1"

$repoRoot = Get-RepoRoot
$logFile = Join-Path $repoRoot "artifacts/logs/genie_lamp.log.jsonl"
if (-not (Test-Path $logFile)) {
    Write-Host "No structured log file found at $logFile yet." -ForegroundColor Yellow
    return
}

Write-Step "Last $Tail log lines"
Get-Content -Path $logFile -Tail $Tail
