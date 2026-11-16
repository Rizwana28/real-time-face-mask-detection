@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Not set up yet - run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" detect.py --source 0 %*
pause
