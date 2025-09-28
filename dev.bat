@echo off
REM CogniForge Development Server Runner (Windows Batch)
REM Run with: dev.bat
REM Or with parameters: dev.bat 8080

setlocal

REM Default values
set HOST=127.0.0.1
set PORT=8000
set LOG_LEVEL=info

REM Check for port argument
if not "%1"=="" set PORT=%1

REM Check for host argument
if not "%2"=="" set HOST=%2

REM Check for log level argument
if not "%3"=="" set LOG_LEVEL=%3

echo ============================================================
echo          COGNIFORGE API - DEVELOPMENT SERVER
echo ============================================================
echo.
echo Starting server at http://%HOST%:%PORT%
echo Hot reload: ENABLED
echo Log level: %LOG_LEVEL%
echo.
echo ------------------------------------------------------------
echo Available Endpoints:
echo   - API Documentation: http://%HOST%:%PORT%/docs
echo   - Alternative Docs:  http://%HOST%:%PORT%/redoc
echo   - Health Check:      http://%HOST%:%PORT%/health
echo   - Training API:      http://%HOST%:%PORT%/api/v1/train
echo   - Code Generation:   http://%HOST%:%PORT%/api/v1/generate
echo   - Demo API:          http://%HOST%:%PORT%/api/v1/demo
echo ------------------------------------------------------------
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

REM Run uvicorn with hot reload
uvicorn main:app --reload --host %HOST% --port %PORT% --log-level %LOG_LEVEL%

echo.
echo ============================================================
echo Server stopped
echo ============================================================

endlocal