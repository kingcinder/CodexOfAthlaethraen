[CmdletBinding()]
param(
    [Alias("AllowDownloadModels")]
    [switch]$AllowNetworkModels,
    [switch]$UseGpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $repoRoot) {
    $repoRoot = Get-Location
}
$repoRoot = (Resolve-Path $repoRoot).Path
. (Join-Path $repoRoot "scripts/common.ps1")

$logDir = Join-Path $repoRoot "artifacts/logs"
Ensure-Directory $logDir
$transcript = Join-Path $logDir "run.ps1.log"
Start-Transcript -Path $transcript -Append | Out-Null

try {
    Write-Step "Step 1/6: Environment diagnostics"
    & (Join-Path $repoRoot "scripts/doctor.ps1")

    Write-Step "Step 2/6: Installation"
    $installArgs = @()
    if ($AllowNetworkModels) { $installArgs += "-AllowNetworkModels" }
    if ($UseGpu) { $installArgs += "-UseGpu" }
    & (Join-Path $repoRoot "scripts/install.ps1") @installArgs

    Write-Step "Step 3/6: Build artifacts"
    & (Join-Path $repoRoot "scripts/build.ps1")

    Write-Step "Step 4/6: Launch service"
    $serviceArgs = @()
    if ($UseGpu) { $serviceArgs += "-UseGpu" }
    & (Join-Path $repoRoot "scripts/service-start.ps1") @serviceArgs

    Write-Step "Step 5/6: Health verification"
    & (Join-Path $repoRoot "scripts/health.ps1")

    Write-Step "Step 6/6: Smoke tests"
    & (Join-Path $repoRoot "scripts/smoke.ps1")

    Write-Step "Collecting tail of structured logs"
    & (Join-Path $repoRoot "scripts/logs.ps1") -Tail 20
}
finally {
    & (Join-Path $repoRoot "scripts/service-stop.ps1")
    Stop-Transcript | Out-Null
}
