[CmdletBinding()]
param(
    [string]$Url = "http://127.0.0.1:7860/health",
    [int]$Retries = 15,
    [int]$DelaySeconds = 2,
    [int]$TimeoutSeconds = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    if ($MyInvocation.MyCommand.Path) {
        $scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
        return (Resolve-Path (Join-Path $scriptDirectory ".."))
    }
    return (Get-Location)
}

$repoRoot = (Get-RepoRoot).Path
$logDir = Join-Path $repoRoot "artifacts/logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logDir "health-$timestamp.log"
Start-Transcript -Path $logPath -Append | Out-Null

try {
    Write-Host "[health] Probing $Url"
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            $response = Invoke-RestMethod -Method Get -Uri $Url -TimeoutSec $TimeoutSeconds
            if ($null -ne $response -and $response.ok) {
                Write-Host "[health] Service healthy on attempt $attempt"
                return
            }
            Write-Host "[health] Unexpected response on attempt $attempt: $response" -ForegroundColor Yellow
        }
        catch {
            Write-Host "[health] Attempt $attempt failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
        if ($attempt -lt $Retries) {
            Start-Sleep -Seconds $DelaySeconds
        }
    }
    throw "Health check failed after $Retries attempts for $Url"
}
finally {
    Stop-Transcript | Out-Null
}
