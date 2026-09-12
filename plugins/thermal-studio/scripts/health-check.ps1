param([int]$Port = 8765)

$ErrorActionPreference = 'Stop'
$url = "http://127.0.0.1:$Port/api/health"
try {
    $health = Invoke-RestMethod -Uri $url -TimeoutSec 5
} catch {
    throw "Thermal Studio health check failed at $url. Start the service first. $($_.Exception.Message)"
}

$health | ConvertTo-Json -Depth 8
