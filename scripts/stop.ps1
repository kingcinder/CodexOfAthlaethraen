[CmdletBinding()]
param(
    [int]$Port = 7860
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
$logPath = Join-Path $logDir "stop-$timestamp.log"
Start-Transcript -Path $logPath -Append | Out-Null

try {
    Write-Host "[stop] Inspecting TCP connections on port $Port"
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction Stop
    }
    catch {
        throw "Unable to query TCP connections. Ensure the NetTCPIP module is available."
    }

    if (-not $connections) {
        Write-Host "[stop] No processes are bound to port $Port"
        return
    }

    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($pid in $pids) {
        try {
            $process = Get-Process -Id $pid -ErrorAction Stop
            Write-Host "[stop] Terminating process $($process.ProcessName) (PID $pid)"
            Stop-Process -Id $pid -ErrorAction Stop
        }
        catch {
            Write-Host "[stop] Failed to terminate PID $pid: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    Write-Host "[stop] Completed"
}
finally {
    Stop-Transcript | Out-Null
}
