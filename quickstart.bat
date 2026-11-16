@echo off
setlocal
cd /d "%~dp0"
set "TMP=%CD%\.tmp"
set "TEMP=%CD%\.tmp"
if not exist ".tmp" mkdir ".tmp"

if not exist ".venv\Scripts\python.exe" (
    echo Environment not set up yet - running setup first...
    call setup.bat
)
if not exist ".venv\Scripts\python.exe" exit /b 1

if exist "weights\best.pt" goto :run

echo.
echo === Step 1/3: preparing the dataset ===
if not exist "datasets\face_mask\data.yaml" (
    ".venv\Scripts\python.exe" prepare_dataset.py
    if errorlevel 1 (
        echo Dataset step failed - see the instructions above.
        pause
        exit /b 1
    )
)
echo.
echo === Step 2/3: fine-tuning YOLO (this takes a while, GPU strongly recommended) ===
".venv\Scripts\python.exe" train.py
if errorlevel 1 (
    echo Training failed - see messages above.
    pause
    exit /b 1
)

:run
echo.
echo === Step 3/3: starting webcam detection (press q in the window to quit) ===
".venv\Scripts\python.exe" detect.py --source 0
pause
