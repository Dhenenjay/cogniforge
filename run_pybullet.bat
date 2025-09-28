@echo off
REM Batch script to run Python scripts with PyBullet environment

if "%1"=="" (
    echo Usage: run_pybullet.bat script.py
    echo.
    echo This will run your Python script in the PyBullet conda environment.
    exit /b 1
)

echo Running %1 with PyBullet environment...
C:\Users\Dhenenjay\miniconda3\Scripts\conda.exe run -n pybullet_env python %*