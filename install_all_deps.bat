@echo off
echo ====================================================
echo Installing ALL Cogniforge Dependencies
echo This is CRITICAL for 100% functionality
echo ====================================================

REM Activate conda environment
call C:\Users\Dhenenjay\miniconda3\Scripts\activate.bat pybullet_env

echo.
echo Step 1: Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Step 2: Installing Core Dependencies...
python -m pip install numpy scipy matplotlib

echo.
echo Step 3: Installing Machine Learning packages...
python -m pip install torch torchvision scikit-learn

echo.
echo Step 4: Installing OpenAI and AI packages...
python -m pip install openai

echo.
echo Step 5: Installing Computer Vision...
python -m pip install opencv-python pillow imageio imageio-ffmpeg

echo.
echo Step 6: Installing Data Processing...
python -m pip install pandas

echo.
echo Step 7: Installing Environment and Configuration...
python -m pip install python-dotenv pyyaml

echo.
echo Step 8: Installing Web/API packages...
python -m pip install fastapi uvicorn pydantic httpx requests aiofiles

echo.
echo Step 9: Installing RL packages...
python -m pip install gymnasium stable-baselines3 wandb

echo.
echo Step 10: Installing Utilities...
python -m pip install tqdm rich structlog colorama watchfiles prometheus-client

echo.
echo Step 11: Installing CMA-ES optimizer...
python -m pip install cma

echo.
echo Step 12: Installing Testing Tools...
python -m pip install pytest pytest-asyncio pytest-cov black flake8 mypy

echo.
echo Step 13: Installing Additional ML tools...
python -m pip install plotly dash tensorboard

echo.
echo Step 14: Verifying critical packages...
python -c "import pybullet; print(f'PyBullet: {pybullet.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

echo.
echo ====================================================
echo ALL Dependencies Installation COMPLETE!
echo ====================================================
pause