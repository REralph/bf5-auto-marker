@echo off
rem One-time setup: makes a private Python environment in .venv and installs the libraries the app needs.
rem Needs Python 3 from https://www.python.org/downloads/ (the app is developed on Python 3.13).
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
    echo Python was not found. Install it from https://www.python.org/downloads/ and run setup.bat again.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python environment in .venv ...
    %PY% -m venv .venv || goto failed
)
echo Installing the libraries (this can take a few minutes the first time) ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto failed
echo.
echo Setup finished. Start BF5 Auto Marker with run.bat
pause
exit /b 0

:failed
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
