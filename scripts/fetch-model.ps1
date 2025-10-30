[CmdletBinding()]
param(
    [switch]$AllowNetwork
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot/common.ps1"

$repoRoot = Get-RepoRoot
Import-DotEnv (Join-Path $repoRoot ".env")
$modelRoot = Join-Path $repoRoot "artifacts/models/sentence-transformers"
$modelName = "sentence-transformers/all-MiniLM-L6-v2"
$targetDir = Join-Path $modelRoot "all-MiniLM-L6-v2"
Ensure-Directory $modelRoot

function Test-ModelPresent {
    param([string]$Path)
    return (Test-Path (Join-Path $Path "config.json"))
}

if (Test-ModelPresent -Path $targetDir) {
    Write-Step "Sentence-Transformer cache already present at $targetDir"
    return
}

if (-not $AllowNetwork) {
    Write-Host "Model artifacts missing at $targetDir." -ForegroundColor Yellow
    Write-Host "Populate the directory manually with the contents of $modelName to remain offline." -ForegroundColor Yellow
    Write-Host "Re-run with -AllowNetwork to download from Hugging Face (requires internet access)." -ForegroundColor Yellow
    return
}

$venvPython = Get-VenvPython
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found. Run scripts/bootstrap.ps1 first."
}

Write-Step "Downloading $modelName to $modelRoot"
Push-Location $repoRoot
try {
    & $venvPython - <<'PY'
from pathlib import Path
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
cache = Path("artifacts/models/sentence-transformers")
cache.mkdir(parents=True, exist_ok=True)
SentenceTransformer(model_name, cache_folder=str(cache))
PY
}
finally {
    Pop-Location
}
