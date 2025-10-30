[CmdletBinding()]
param()

. "$PSScriptRoot/common.ps1"

$repoRoot = Get-RepoRoot
Import-DotEnv (Join-Path $repoRoot ".env")
Write-Step "Verifying Python interpreter"
$pythonCmd = Get-Command python -ErrorAction Stop
$pythonVersion = & $pythonCmd.Path - <<'PY'
import sys
print(sys.version)
PY
if (-not $pythonVersion.StartsWith('3.11')) {
    throw "Python 3.11.x is required. Found: $pythonVersion"
}
Write-Step "Python $pythonVersion detected"

Write-Step "Ensuring curl availability"
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl is required for health checks. Install curl or ensure it is on PATH."
}

Write-Step "Environment check completed"
