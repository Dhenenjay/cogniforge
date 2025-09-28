@echo off
echo ==============================================================
echo                    CogniForge Full Demo
echo     Natural Language Robot Control with Vision & Learning
echo ==============================================================
echo.

echo [1/3] Starting API Server...
start "CogniForge API" /MIN cmd /k "C:\Users\Dhenenjay\miniconda3\envs\pybullet_env\python.exe api_server.py"

echo [2/3] Waiting for API to initialize...
timeout /t 5 /nobreak > nul

echo [3/3] Opening Frontend...
start "" "C:\Users\Dhenenjay\cogniforge\frontend\index.html"

echo.
echo ==============================================================
echo                       DEMO READY!
echo ==============================================================
echo.
echo FRONTEND: Browser opened with control interface
echo BACKEND:  API running at http://localhost:8000
echo.
echo TO USE:
echo 1. In the browser, type your command in the text box
echo 2. Click "Execute" to start the pipeline
echo 3. Watch the real-time progress and metrics
echo.
echo Example command: 
echo "Pick up the blue cube and place it on the green platform"
echo.
echo ==============================================================
echo.

echo Press Y to also run PyBullet visualization demo, or N to skip...
choice /C YN /N /M "Run PyBullet Demo? (Y/N): "

if errorlevel 2 goto end
if errorlevel 1 goto pybullet

:pybullet
echo.
echo Starting PyBullet Integrated Demo...
start "PyBullet Demo" cmd /k "C:\Users\Dhenenjay\miniconda3\envs\pybullet_env\python.exe integrated_demo.py"
echo.
echo PyBullet window opened - watch the robot execute the task!
echo.

:end
echo ==============================================================
echo All components are running. Press any key to stop all services...
echo ==============================================================
pause > nul

echo.
echo Stopping services...
taskkill /FI "WindowTitle eq CogniForge API*" /F 2>nul
taskkill /FI "WindowTitle eq PyBullet Demo*" /F 2>nul

echo.
echo Demo stopped. Thank you!
pause