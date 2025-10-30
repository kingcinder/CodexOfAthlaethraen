[CmdletBinding()]
param()

. "$PSScriptRoot/common.ps1"

$repoRoot = Get-RepoRoot
$venvPython = Get-VenvPython
Ensure-Directory (Join-Path $repoRoot "artifacts/dist")

Write-Step "Building wheel artifact"
Push-Location $repoRoot
& $venvPython -m pip wheel . -w artifacts/dist --no-deps
Pop-Location

Write-Step "Wheel available under artifacts/dist"
