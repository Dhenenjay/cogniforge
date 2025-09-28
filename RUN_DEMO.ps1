# CogniForge Complete Demo Runner
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "           CogniForge - AI Robot Control Demo         " -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Kill any existing Python processes
Write-Host "[1/4] Cleaning up existing processes..." -ForegroundColor Green
Stop-Process -Name python -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 2

# Start API Server
Write-Host "[2/4] Starting API Server..." -ForegroundColor Green
$apiProcess = Start-Process -FilePath "C:\Users\Dhenenjay\miniconda3\envs\pybullet_env\python.exe" `
    -ArgumentList "C:\Users\Dhenenjay\cogniforge\api_server.py" `
    -WorkingDirectory "C:\Users\Dhenenjay\cogniforge" `
    -PassThru -WindowStyle Minimized

Write-Host "    API Server PID: $($apiProcess.Id)" -ForegroundColor Gray

# Wait for API to be ready
Write-Host "[3/4] Waiting for API to initialize..." -ForegroundColor Green
Start-Sleep -Seconds 3

# Test API
try {
    $apiTest = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -ErrorAction Stop
    Write-Host "    API is running at http://localhost:8000" -ForegroundColor Green
} catch {
    Write-Host "    API failed to start!" -ForegroundColor Red
}

# Start Frontend Web Server
Write-Host "[4/4] Starting Frontend Server..." -ForegroundColor Green
$frontendProcess = Start-Process -FilePath "C:\Users\Dhenenjay\miniconda3\envs\pybullet_env\python.exe" `
    -ArgumentList "-m", "http.server", "8080" `
    -WorkingDirectory "C:\Users\Dhenenjay\cogniforge\frontend" `
    -PassThru -WindowStyle Minimized

Write-Host "    Frontend Server PID: $($frontendProcess.Id)" -ForegroundColor Gray
Write-Host "    ✓ Frontend running at http://localhost:8080" -ForegroundColor Green

Start-Sleep -Seconds 2

# Open Browser
Write-Host ""
Write-Host "Opening browser..." -ForegroundColor Yellow
Start-Process "http://localhost:8080"

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "                  SYSTEM READY!                       " -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Frontend: http://localhost:8080" -ForegroundColor White
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "In the browser:" -ForegroundColor Yellow
Write-Host "1. Type: 'Pick up the blue cube and place it on the green platform'" -ForegroundColor White
Write-Host "2. Click: Execute" -ForegroundColor White
Write-Host "3. Watch the real-time progress!" -ForegroundColor White
Write-Host ""

# Ask about PyBullet demo
Write-Host "Do you want to run PyBullet visualization? (Y/N): " -ForegroundColor Cyan -NoNewline
$response = Read-Host

if ($response -eq 'Y' -or $response -eq 'y') {
    Write-Host ""
    Write-Host "Starting PyBullet Demo..." -ForegroundColor Green
    Start-Process -FilePath "C:\Users\Dhenenjay\miniconda3\envs\pybullet_env\python.exe" `
        -ArgumentList "C:\Users\Dhenenjay\cogniforge\integrated_demo.py" `
        -WorkingDirectory "C:\Users\Dhenenjay\cogniforge" `
        -WindowStyle Normal
    Write-Host "PyBullet window opened!" -ForegroundColor Green
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Press any key to stop all services..." -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Read-Host

# Cleanup
Write-Host "Stopping services..." -ForegroundColor Red
Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $frontendProcess.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Name python -Force -ErrorAction SilentlyContinue

Write-Host "Demo stopped. Thank you!" -ForegroundColor Green