@echo off
setlocal enabledelayedexpansion
title PhasePix - SEM Phase Analyzer

:: Set current directory to the folder containing this .bat file
cd /d "%~dp0"

echo ==================================================
echo   Starting PhasePix (SEM Phase Analyzer)...
echo ==================================================
echo.

:: Check for Python virtual environment (.venv or venv)
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment (.venv)...
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment (venv)...
    call "venv\Scripts\activate.bat"
) else (
    echo [WARNING] Virtual environment not found in .venv or venv.
    echo Attempting to run using system Python / Streamlit...
)

echo.
echo Launching PhasePix in browser via Streamlit...
echo.

:: Run Streamlit via python -m streamlit (ensures correct python environment is used)
python -m streamlit run app.py

:: Keep console window open if application terminates or encounters an error
if %ERRORLEVEL% neq 0 (
    echo.
    echo ==================================================
    echo [ERROR] PhasePix encountered an issue (Exit Code: %ERRORLEVEL%).
    echo Please review the error log above.
    echo ==================================================
)

echo.
echo Press any key to close this window...
pause >nul
