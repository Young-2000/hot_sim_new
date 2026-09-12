param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$appDirectory = $PSScriptRoot
$runtimeDirectory = Join-Path $appDirectory '.runtime'
New-Item -ItemType Directory -Force -Path $runtimeDirectory | Out-Null
$pythonCandidates = @((Join-Path $appDirectory '.venv\Scripts\python.exe'), 'D:\Users\zongtianyu\anaconda3\python.exe')
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) { $pythonCandidates += $pythonCommand.Source }
$pythonPath = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $pythonPath) { throw 'Python not found. Install Python 3.11 or later, then run Install.ps1.' }
$url = "http://127.0.0.1:$Port"
try { $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 2 } catch { $health = $null }
if ($health -and $health.app -eq 'Thermal Studio Local') {
    Write-Output "Thermal Studio is running at $url"
    if (-not $NoBrowser) { Start-Process $url }
    exit 0
}
if ($health) { throw "Port $Port is in use. Choose another port with -Port." }
$process = Start-Process -FilePath $pythonPath -ArgumentList @('server.py', '--port', "$Port") -WorkingDirectory $appDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDirectory 'server.log') -RedirectStandardError (Join-Path $runtimeDirectory 'server-error.log')
$process.Id | Set-Content -LiteralPath (Join-Path $runtimeDirectory 'server.pid')
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    Start-Sleep -Milliseconds 300
    if ($process.HasExited) { throw "Server stopped. Read .runtime/server-error.log." }
    try {
        $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 1
        if ($health.app -eq 'Thermal Studio Local') { Write-Output "Thermal Studio is running at $url"; if (-not $NoBrowser) { Start-Process $url }; exit 0 }
    } catch { }
}
throw 'Startup timed out. Read .runtime/server-error.log.'
