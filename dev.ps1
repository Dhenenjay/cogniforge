# CogniForge Development Server Runner (PowerShell)
# Run with: .\dev.ps1
# Or with parameters: .\dev.ps1 -Port 8080 -NoReload

param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload,
    [int]$Workers = 1,
    [string]$LogLevel = "info",
    [switch]$Production,
    [switch]$Debug
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "         COGNIFORGE API - DEVELOPMENT SERVER" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

# Check if uvicorn is installed
$uvicornCheck = pip show uvicorn 2>$null
if (-not $uvicornCheck) {
    Write-Host "❌ Uvicorn not found. Installing..." -ForegroundColor Red
    pip install uvicorn[standard]
}

# Check if FastAPI is installed
$fastapiCheck = pip show fastapi 2>$null
if (-not $fastapiCheck) {
    Write-Host "❌ FastAPI not found. Installing..." -ForegroundColor Red
    pip install fastapi
}

# Set log level based on flags
if ($Debug) {
    $LogLevel = "debug"
}

# Display configuration
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  • Host:        $Host" -ForegroundColor White
Write-Host "  • Port:        $Port" -ForegroundColor White
Write-Host "  • Hot Reload:  $(if ($NoReload) {'Disabled'} else {'Enabled'})" -ForegroundColor White
Write-Host "  • Log Level:   $LogLevel" -ForegroundColor White
Write-Host "  • Workers:     $Workers" -ForegroundColor White

Write-Host ""
Write-Host "------------------------------------------------------------" -ForegroundColor Gray
Write-Host "Available Endpoints:" -ForegroundColor Green
Write-Host "  • API Docs:     http://${Host}:${Port}/docs" -ForegroundColor White
Write-Host "  • ReDoc:        http://${Host}:${Port}/redoc" -ForegroundColor White
Write-Host "  • Health:       http://${Host}:${Port}/health" -ForegroundColor White
Write-Host "  • Training:     http://${Host}:${Port}/api/v1/train" -ForegroundColor White
Write-Host "  • Code Gen:     http://${Host}:${Port}/api/v1/generate" -ForegroundColor White
Write-Host "  • Demo:         http://${Host}:${Port}/api/v1/demo" -ForegroundColor White
Write-Host "------------------------------------------------------------" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Build uvicorn command
$uvicornArgs = @(
    "main:app",
    "--host", $Host,
    "--port", $Port,
    "--log-level", $LogLevel
)

if (-not $NoReload) {
    $uvicornArgs += "--reload"
}

if ($Production) {
    $uvicornArgs += "--workers", $Workers
}

# Run uvicorn
try {
    uvicorn @uvicornArgs
}
catch {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Server stopped" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Cyan
}