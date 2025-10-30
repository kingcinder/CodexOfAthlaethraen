[CmdletBinding()]
param()

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
$logPath = Join-Path $logDir "repair-$timestamp.log"
Start-Transcript -Path $logPath -Append | Out-Null

$success = $false
try {
    Write-Host "[repair-venv] Repository root: $repoRoot"

    $python = Get-Command python -ErrorAction Stop
    Write-Host "[repair-venv] Using python at $($python.Path)"

    $venvPath = Join-Path $repoRoot ".venv"
    if (Test-Path $venvPath) {
        Write-Host "[repair-venv] Removing existing virtual environment at $venvPath"
        Remove-Item -Path $venvPath -Recurse -Force
    } else {
        Write-Host "[repair-venv] No existing virtual environment detected"
    }

    Write-Host "[repair-venv] Creating fresh virtual environment"
    & $python.Path -m venv $venvPath

    $venvPython = Join-Path $venvPath "Scripts/python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment python not found at $venvPython"
    }

    Write-Host "[repair-venv] Upgrading embedded packaging tools via ensurepip"
    & $venvPython -m ensurepip --upgrade

    Write-Host "[repair-venv] Upgrading pip/setuptools/wheel"
    & $venvPython -m pip install --upgrade pip setuptools wheel

    Write-Host "[repair-venv] Installing genie-lamp in editable mode"
    Push-Location $repoRoot
    try {
        & $venvPython -m pip install -e .
    }
    finally {
        Pop-Location
    }

    $success = $true
    Write-Host "PASS: Virtual environment repaired successfully." -ForegroundColor Green
    Write-Host "Next steps: run scripts/run.ps1 or run.ps1 to start the service." -ForegroundColor Green
}
catch {
    Write-Error "FAIL: $($_.Exception.Message)"
    Write-Host "Check the log at $logPath for details and rerun scripts/bootstrap.ps1 if the issue persists." -ForegroundColor Yellow
    exit 1
}
finally {
    Stop-Transcript | Out-Null
    if ($success) {
        exit 0
    }
}
