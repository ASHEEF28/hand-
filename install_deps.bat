@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem  GestureOS dependency installer.
rem
rem    install_deps.bat          runtime dependencies
rem    install_deps.bat --dev    runtime + test dependencies (pytest)
rem ---------------------------------------------------------------------------

set "PYTHON=python"
where py >nul 2>nul && set "PYTHON=py -3"

echo Using: %PYTHON%
%PYTHON% --version
if errorlevel 1 (
    echo Python was not found. Install Python 3.10-3.12 from https://python.org
    echo and tick "Add python.exe to PATH" during setup.
    pause
    exit /b 1
)

echo.
echo [1/3] Upgrading pip...
%PYTHON% -m pip install --upgrade pip

echo.
echo [2/3] Installing runtime dependencies...
rem MediaPipe is pinned to ^<1.0 in requirements.txt: 1.x dropped the bundled
rem hand-landmark model and the mp.solutions API this project prefers.
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed. Common fixes:
    echo   - Use Python 3.10, 3.11 or 3.12 ^(MediaPipe has no wheels for 3.13+^)
    echo   - %PYTHON% -m pip install "mediapipe>=0.10.14,^<1.0"
    echo   - %PYTHON% -m pip install opencv-python-headless
    pause
    exit /b 1
)

if "%~1"=="--dev" (
    echo.
    echo [2b/3] Installing development dependencies...
    %PYTHON% -m pip install -r requirements-dev.txt
)

echo.
echo [3/3] Verifying the installation...
%PYTHON% main.py --check

echo.
echo Done! Start GestureOS with:  run.bat   ^(or: %PYTHON% main.py^)
echo No webcam handy? Try:        run.bat --demo --frames 300
pause
exit /b 0
