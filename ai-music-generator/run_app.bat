@echo off
title Antigravity // AI Music Generator Launcher
echo ============================================================
echo   ANTIGRAVITY // Deep Learning LSTM Music Generator
echo ============================================================
echo.
echo [1/2] Checking and installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to verify/install python dependencies.
    echo Please make sure Python and pip are correctly added to your environment variables.
    pause
    exit /b %errorlevel%
)

echo [SUCCESS] Dependencies verified.
echo.
echo [2/2] Launching Flask Server on http://127.0.0.1:5000...
echo.
python main.py serve
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Web Server stopped with errors.
    pause
)
