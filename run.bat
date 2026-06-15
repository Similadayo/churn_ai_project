@echo off
REM ============================================================
REM  Customer Churn Prediction - One-Click Launcher (Windows)
REM  Just DOUBLE-CLICK this file. It will:
REM    1) Find Python
REM    2) Create a private environment (first run only)
REM    3) Install everything needed (first run only)
REM    4) Open the dashboard in your web browser
REM ============================================================

cd /d "%~dp0"
title Customer Churn Dashboard

echo.
echo ============================================================
echo   Starting the Customer Churn Prediction Dashboard
echo ============================================================
echo.

REM --- 1. Locate a Python interpreter -------------------------
set "PY_CMD="
where py >nul 2>nul && set "PY_CMD=py -3.11"
if not defined PY_CMD (
    where python >nul 2>nul && set "PY_CMD=python"
)

if not defined PY_CMD (
    echo [ERROR] Python was not found on this computer.
    echo.
    echo Please install Python 3.11 from:
    echo     https://www.python.org/downloads/release/python-3119/
    echo.
    echo IMPORTANT: During install, tick the box
    echo     "Add python.exe to PATH"
    echo Then double-click this file again.
    echo.
    pause
    exit /b 1
)

REM --- 2. Create the virtual environment (first run only) -----
if not exist "venv\Scripts\python.exe" (
    echo [setup] Creating a private Python environment ^(first run, please wait^)...
    %PY_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Could not create the environment. See messages above.
        pause
        exit /b 1
    )
)

REM --- 3. Install dependencies --------------------------------
echo [setup] Installing required packages ^(first run can take a few minutes^)...
call "venv\Scripts\python.exe" -m pip install --upgrade pip >nul
call "venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Installing packages failed. See messages above.
    pause
    exit /b 1
)

REM --- 4. Launch the dashboard --------------------------------
echo.
echo ============================================================
echo   All set! Opening the dashboard in your browser...
echo   ^(Keep this black window open while you use the app.^)
echo   To stop: close this window.
echo ============================================================
echo.
call "venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py

pause
