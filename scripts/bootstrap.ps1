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
$logPath = Join-Path $logDir "bootstrap-$timestamp.log"
Start-Transcript -Path $logPath -Append | Out-Null

try {
    Write-Host "[bootstrap] Repository root: $repoRoot"

    $python = Get-Command python -ErrorAction Stop
    Write-Host "[bootstrap] Using python at $($python.Path)"

    $venvPath = Join-Path $repoRoot ".venv"
    if (-not (Test-Path $venvPath)) {
        Write-Host "[bootstrap] Creating virtual environment at $venvPath"
        & $python.Path -m venv $venvPath
    } else {
        Write-Host "[bootstrap] Virtual environment already exists at $venvPath"
    }

    $venvPython = Join-Path $venvPath "Scripts/python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment python not found at $venvPython"
    }

    Write-Host "[bootstrap] Upgrading pip/setuptools/wheel"
    & $venvPython -m pip install --upgrade pip setuptools wheel

    Write-Host "[bootstrap] Installing genie-lamp in editable mode"
    Push-Location $repoRoot
    try {
        & $venvPython -m pip install -e .
    }
    finally {
        Pop-Location
    }

    Write-Host "[bootstrap] Completed successfully"
}
finally {
    Stop-Transcript | Out-Null
}
