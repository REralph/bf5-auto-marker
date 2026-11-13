@echo off
rem Starts BF5 Auto Marker without a console window. Run setup.bat once first.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo The Python environment is missing - run setup.bat first.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" main.py
