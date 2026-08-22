@echo off
cd /d "%~dp0"
echo Installing GestureOS dependencies...
pip install -r requirements.txt
echo Done! Run with: run.bat or python main.py