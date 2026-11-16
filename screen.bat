@echo off
rem Live mask detection on your screen (Zoom, YouTube, any window). Press q in the preview window to quit.
rem Examples:  screen.bat            screen.bat --source screen:2        screen.bat --source screen:0,0,960,540
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Not set up yet - run setup.bat first.
    pause
    exit /b 1
)
if "%~1"=="" (".venv\Scripts\python.exe" detect.py --source screen) else (".venv\Scripts\python.exe" detect.py %*)
pause
