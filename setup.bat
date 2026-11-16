@echo off
setlocal
cd /d "%~dp0"
echo ===== Face Mask Detection : setup =====
echo Install location: %CD%
echo (Tip: keep this folder on D: or E: - drive C: is almost full.)
echo.

where python >nul 2>&1
if errorlevel 1 goto :nopython

if not exist ".tmp" mkdir ".tmp"
set "TMP=%CD%\.tmp"
set "TEMP=%CD%\.tmp"
set "PIP_NO_CACHE_DIR=1"

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto :fail
)
set "PY=%CD%\.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

nvidia-smi >nul 2>&1
if errorlevel 1 goto :cpu
echo NVIDIA GPU detected - installing CUDA build of PyTorch
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :fail
goto :rest

:cpu
echo No NVIDIA GPU found - installing CPU build of PyTorch
"%PY%" -m pip install torch torchvision
if errorlevel 1 goto :fail

:rest
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
"%PY%" -c "import torch,cv2,ultralytics;print('OK  torch',torch.__version__,'| CUDA available:',torch.cuda.is_available(),'| ultralytics',ultralytics.__version__)"
if errorlevel 1 goto :fail
rmdir /s /q ".tmp" >nul 2>&1
echo.
echo Setup complete.  Next: run quickstart.bat   (or see README.md)
pause
exit /b 0

:nopython
echo Python was not found. Install Python 3.10 - 3.13 from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during installation, then run setup.bat again.
pause
exit /b 1

:fail
echo.
echo Setup FAILED - read the messages above.
pause
exit /b 1
