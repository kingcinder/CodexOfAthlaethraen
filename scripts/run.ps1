[CmdletBinding()]
param(
    [string]$Host = "127.0.0.1",
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
$venvPath = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts/python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run scripts/bootstrap.ps1 first."
}

$env:PYTHONPATH = (Resolve-Path $repoRoot).Path
$env:GENIE_LAMP_HOME = $repoRoot

Write-Host "[run] PYTHONPATH=$env:PYTHONPATH"
Write-Host "[run] Launching uvicorn on $Host:$Port"

$arguments = @(
    "-m", "uvicorn", "genie_lamp.main:app",
    "--host", $Host,
    "--port", $Port.ToString(),
    "--reload",
    "--reload-dir", ".."
)

Push-Location $repoRoot
try {
    & $venvPython $arguments
}
finally {
    Pop-Location
}
