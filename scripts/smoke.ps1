[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:7860",
    [int]$TimeoutSeconds = 10
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
$logPath = Join-Path $logDir "smoke-$timestamp.log"
Start-Transcript -Path $logPath -Append | Out-Null

try {
    $healthUrl = "$BaseUrl/health"
    $diagUrl = "$BaseUrl/diag"

    Write-Host "[smoke] Checking $healthUrl"
    $health = Invoke-RestMethod -Method Get -Uri $healthUrl -TimeoutSec $TimeoutSeconds
    if (-not ($health.ok -eq $true)) {
        throw "Health endpoint did not return ok=true"
    }
    Write-Host "[smoke] Health endpoint returned ok"

    Write-Host "[smoke] Checking $diagUrl"
    $diag = Invoke-RestMethod -Method Get -Uri $diagUrl -TimeoutSec $TimeoutSeconds

    $requiredKeys = @("python", "platform")
    foreach ($key in $requiredKeys) {
        if (-not $diag.ContainsKey($key)) {
            throw "Diagnostics payload missing required key '$key'"
        }
    }

    if (-not ($diag.ContainsKey("torch") -or $diag.ContainsKey("torch_error"))) {
        throw "Diagnostics payload must include torch version or torch_error"
    }
    if (-not ($diag.ContainsKey("transformers") -or $diag.ContainsKey("transformers_error"))) {
        throw "Diagnostics payload must include transformers version or transformers_error"
    }

    Write-Host "[smoke] Diagnostics payload validated"
}
finally {
    Stop-Transcript | Out-Null
}
