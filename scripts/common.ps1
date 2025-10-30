Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    param()
    return (Resolve-Path (Join-Path $PSScriptRoot ".."))
}

function Get-VenvPath {
    param()
    return (Join-Path (Get-RepoRoot) ".venv")
}

function Get-VenvPython {
    param()
    $venvPath = Get-VenvPath
    return (Join-Path $venvPath "Scripts/python.exe")
}

function Ensure-Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Import-DotEnv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    if (-not (Test-Path $Path)) {
        return
    }
    foreach ($line in Get-Content -Path $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line.TrimStart().StartsWith('#')) { continue }
        $parts = $line -split '=', 2
        if ($parts.Count -ne 2) { continue }
        $key = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($key) {
            [Environment]::SetEnvironmentVariable($key, $value)
        }
    }
}

function Write-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )
    Write-Host "[+] $Message"
}

function Get-ServiceSettings {
    param()
    $host = [Environment]::GetEnvironmentVariable('GENIE_LAMP_HOST')
    if (-not $host) { $host = '127.0.0.1' }
    $portValue = [Environment]::GetEnvironmentVariable('GENIE_LAMP_PORT')
    if (-not $portValue -or -not ($portValue -as [int])) { $portValue = '7860' }
    return @{ Host = $host; Port = [int]$portValue }
}

