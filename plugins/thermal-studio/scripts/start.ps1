param(
    [int]$Port = 8765,
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Device = 'auto',
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$pluginRoot = Split-Path -Parent $PSScriptRoot
$configuredRoot = $env:THERMAL_STUDIO_ROOT
if ($configuredRoot) {
    $appRoot = (Resolve-Path -LiteralPath $configuredRoot).Path
} else {
    $appRoot = (Resolve-Path -LiteralPath (Join-Path $pluginRoot '..\..')).Path
}

$entrypoint = Join-Path $appRoot 'Start.ps1'
if (-not (Test-Path -LiteralPath $entrypoint)) {
    throw "Thermal Studio root was not found at '$appRoot'. Set THERMAL_STUDIO_ROOT to the directory containing Start.ps1."
}

if ($NoBrowser) {
    & $entrypoint -Port $Port -Device $Device -NoBrowser
} else {
    & $entrypoint -Port $Port -Device $Device
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
