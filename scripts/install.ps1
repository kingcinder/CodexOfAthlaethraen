[CmdletBinding()]
param(
    [Alias("AllowDownloadModels")]
    [switch]$AllowNetworkModels,
    [switch]$UseGpu
)

. "$PSScriptRoot/common.ps1"

$repoRoot = Get-RepoRoot
Import-DotEnv (Join-Path $repoRoot ".env")

Write-Step "Preparing Python virtual environment"
$venvPath = Get-VenvPath
if (-not (Test-Path $venvPath)) {
    & python -m venv $venvPath
}

$venvPython = Get-VenvPython
Write-Step "Upgrading packaging tools"
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Step "Installing Genie Lamp in editable mode"
Push-Location $repoRoot
& $venvPython -m pip install -e .
& $venvPython -m pip install pytest==8.3.4
Pop-Location

if ($UseGpu) {
    Write-Step "GPU flag detected. Install torch with CUDA manually if your hardware supports it:"
    Write-Host "    $venvPython -m pip install torch==2.9.0+cu121 --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Yellow
}

$modelArgs = @()
if ($AllowNetworkModels) {
    $modelArgs += "-AllowNetwork"
}
& (Join-Path $repoRoot "scripts/fetch-model.ps1") @modelArgs

Ensure-Directory (Join-Path $repoRoot "artifacts/logs")
Ensure-Directory (Join-Path $repoRoot "artifacts/models")
Ensure-Directory (Join-Path $repoRoot "artifacts/run")

Write-Step "Installation completed"
