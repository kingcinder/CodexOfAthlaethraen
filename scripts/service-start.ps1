[CmdletBinding()]
param(
    [switch]$UseGpu
)

. "$PSScriptRoot/common.ps1"

$repoRoot = Get-RepoRoot
Import-DotEnv (Join-Path $repoRoot ".env")
[Environment]::SetEnvironmentVariable('GENIE_LAMP_HOME', $repoRoot)
if ($UseGpu) {
    [Environment]::SetEnvironmentVariable('GENIE_LAMP_USE_GPU', '1')
}

$pidPath = Join-Path $repoRoot "artifacts/run/uvicorn.pid"
if (Test-Path $pidPath) {
    try {
        $existingPid = Get-Content -Path $pidPath | Select-Object -First 1
        if ($existingPid) {
            $proc = Get-Process -Id [int]$existingPid -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Step "Existing Genie Lamp process detected (PID $existingPid). Stopping first."
                & (Join-Path $repoRoot "scripts/service-stop.ps1")
            }
        }
    } catch {
        # ignore stale pid entries
    }
}

$settings = Get-ServiceSettings
$venvPython = Get-VenvPython
$stdoutLog = Join-Path $repoRoot "artifacts/logs/uvicorn.stdout.log"
$stderrLog = Join-Path $repoRoot "artifacts/logs/uvicorn.stderr.log"
Ensure-Directory (Split-Path $stdoutLog)

$arguments = @('-m', 'genie_lamp', 'serve', '--host', $settings.Host, '--port', $settings.Port.ToString())
if ($UseGpu) {
    $arguments += '--use-gpu'
}

Write-Step "Starting Genie Lamp service on $($settings.Host):$($settings.Port)"
$process = Start-Process -FilePath $venvPython -ArgumentList $arguments -WorkingDirectory $repoRoot -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
Set-Content -Path $pidPath -Value $process.Id
Start-Sleep -Seconds 3
