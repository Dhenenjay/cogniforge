@echo off
REM CogniForge Demo Recorder - Batch File Wrapper
REM Simple wrapper to run the demo recording scripts

echo ============================================================
echo          COGNIFORGE DEMO RECORDER
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Select recording method:
echo   1. Python recorder (recommended - creates visualizations)
echo   2. PowerShell recorder (screen capture)
echo   3. Quick demo (Python, no prompts)
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting Python demo recorder...
    echo This will create animated visualizations of CogniForge features.
    echo.
    python "%~dp0record_demo.py"
) else if "%choice%"=="2" (
    echo.
    echo Starting PowerShell demo recorder...
    echo This will capture your screen while running demos.
    echo.
    powershell -ExecutionPolicy Bypass -File "%~dp0record_demo.ps1"
) else if "%choice%"=="3" (
    echo.
    echo Running quick demo recording...
    python -c "from record_demo import DemoRecorder; r = DemoRecorder(); r.run_full_demo(); print('Done!')"
) else (
    echo Invalid choice. Please run again and select 1, 2, or 3.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Recording complete! Check the 'recordings' folder for output.
echo ============================================================
pause