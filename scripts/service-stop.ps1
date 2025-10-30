[CmdletBinding()]
param()

. "$PSScriptRoot/common.ps1"

$repoRoot = Get-RepoRoot
$pidPath = Join-Path $repoRoot "artifacts/run/uvicorn.pid"
if (-not (Test-Path $pidPath)) {
    return
}

try {
    $pid = Get-Content -Path $pidPath | Select-Object -First 1
    if ($pid) {
        $process = Get-Process -Id [int]$pid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Step "Stopping Genie Lamp process (PID $pid)"
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        }
    }
} catch {
    # ignore errors
}
Remove-Item -Path $pidPath -ErrorAction SilentlyContinue
