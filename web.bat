@echo off
rem Starts the MaskSense web app and opens http://127.0.0.1:7860 in your browser.
rem Tabs: Webcam | Screen | Video | Photo.  Press Ctrl+C in this window to stop.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Not set up yet - run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py %*
pause
