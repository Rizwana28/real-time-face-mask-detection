@echo off
rem One-click demo: runs the trained model on the sample photos and opens the annotated results.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Not set up yet - run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" detect.py --source samples --out outputs\demo
pause
