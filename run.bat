@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem  GestureOS launcher.  Prefers a local virtual environment, passes every
rem  argument straight through to main.py, and stays open on failure so that
rem  double-clicking it does not hide the error.
rem
rem    run.bat                        start the full UI
rem    run.bat --check                verify the installation
rem    run.bat --demo --frames 300    scripted demo, no webcam needed
rem ---------------------------------------------------------------------------

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
if exist "venv\Scripts\python.exe"  set "PYTHON=venv\Scripts\python.exe"

where py >nul 2>nul
if errorlevel 1 goto :run
if "%PYTHON%"=="python" set "PYTHON=py -3"

:run
echo Starting GestureOS with: %PYTHON%
%PYTHON% main.py %*
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo GestureOS exited with code %EXITCODE%.
    echo Troubleshooting:
    echo   %PYTHON% main.py --check       show missing dependencies
    echo   %PYTHON% main.py --demo        run without a webcam
    echo   type logs\gestureos.log        last log lines
    echo.
    pause
)
exit /b %EXITCODE%
