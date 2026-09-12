param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$url = "http://127.0.0.1:$Port"
try { $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 3 } catch { Write-Output 'Thermal Studio is not responding on this port; no process was stopped.'; exit 0 }
if ($health.app -ne 'Thermal Studio Local') { throw 'This port belongs to another application.' }
$pidFile = Join-Path $PSScriptRoot '.runtime\server.pid'
if (-not (Test-Path -LiteralPath $pidFile)) { throw 'Server pid file missing; no process was stopped.' }
$pidFromFile = [int](Get-Content -LiteralPath $pidFile)
$serverProcessId = [int]$health.pid
if ($pidFromFile -ne $serverProcessId) {
    $fileProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$pidFromFile" -ErrorAction SilentlyContinue
    if (-not $fileProcess -or $fileProcess.CommandLine -notlike '*server.py*') { throw 'Server identity mismatch; no process was stopped.' }
}
$library = Invoke-RestMethod -Uri "$url/api/bootstrap"
foreach ($job in $library.jobs) {
    if ($job.status.phase -in @('running','pending')) { Invoke-RestMethod -Method Post -Uri "$url/api/jobs/$($job.id)/cancel" | Out-Null }
}
Stop-Process -Id $serverProcessId -Force -ErrorAction SilentlyContinue
if ($pidFromFile -ne $serverProcessId) { Stop-Process -Id $pidFromFile -Force -ErrorAction SilentlyContinue }
Write-Output 'Thermal Studio stopped. Models, configurations and completed results are preserved.'
