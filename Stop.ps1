param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$url = "http://127.0.0.1:$Port"
try { $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 3 } catch { Write-Output 'Thermal Studio is not responding on this port; no process was stopped.'; exit 0 }
if ($health.app -ne 'Thermal Studio Local') { throw 'This port belongs to another application.' }
$pidFile = Join-Path $PSScriptRoot '.runtime\server.pid'
if (-not (Test-Path -LiteralPath $pidFile)) { throw 'Server pid file missing; no process was stopped.' }
$serverProcessId = [int](Get-Content -LiteralPath $pidFile)
if ($serverProcessId -ne [int]$health.pid) { throw 'Server identity mismatch; no process was stopped.' }
$library = Invoke-RestMethod -Uri "$url/api/bootstrap"
foreach ($job in $library.jobs) {
    if ($job.status.phase -in @('running','pending')) { Invoke-RestMethod -Method Post -Uri "$url/api/jobs/$($job.id)/cancel" | Out-Null }
}
Stop-Process -Id $serverProcessId
Write-Output 'Thermal Studio stopped. Models, configurations and completed results are preserved.'
