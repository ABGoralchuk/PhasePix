@echo off
setlocal EnableExtensions
title PhasePix - Native Windows Launcher
cd /d "%~dp0"

echo ==================================================
echo   PhasePix - SEM Phase Analyzer - Native Launcher
echo ==================================================
echo.

if not exist "app.py" goto :no_app

rem -----------------------------------------------------------------
rem 1. Locate a Python interpreter: local venv first, then py, then python
rem -----------------------------------------------------------------
set "PY_EXE="
set "PY_ARGS="
if exist ".venv\Scripts\python.exe" set "PY_EXE=.venv\Scripts\python.exe"
if not defined PY_EXE if exist "venv\Scripts\python.exe" set "PY_EXE=venv\Scripts\python.exe"
if defined PY_EXE goto :venv_found

where py >nul 2>&1
if errorlevel 1 goto :try_python
py -3 -c "import sys" >nul 2>&1
if errorlevel 1 goto :try_python
set "PY_EXE=py"
set "PY_ARGS=-3"
echo Using the Windows Python launcher: py -3
goto :check_streamlit

:try_python
where python >nul 2>&1
if errorlevel 1 goto :no_python
set "PY_EXE=python"
set "PY_ARGS="
echo Using the python interpreter found on PATH.
goto :check_streamlit

:venv_found
echo Using virtual environment: %PY_EXE%

rem -----------------------------------------------------------------
rem 2. Verify Streamlit is installed for that interpreter
rem -----------------------------------------------------------------
:check_streamlit
"%PY_EXE%" %PY_ARGS% -c "import streamlit" >nul 2>&1
if errorlevel 1 goto :missing_streamlit

echo.
echo Launching PhasePix - the dashboard opens at http://localhost:8501
echo Press Ctrl+C in this window to stop the server.
echo.

"%PY_EXE%" %PY_ARGS% -m streamlit run app.py
if errorlevel 1 goto :run_failed

echo.
echo PhasePix was closed. No errors were reported.
goto :end_pause

rem -----------------------------------------------------------------
rem Error paths - every one of them ends at :end_pause
rem -----------------------------------------------------------------
:run_failed
echo.
echo ==================================================
echo [ERROR] PhasePix exited with an error.
echo Review the traceback printed above.
echo.
echo If a package is missing, run:
echo   "%PY_EXE%" %PY_ARGS% -m pip install -r requirements.txt
echo ==================================================
echo.
goto :end_pause

:missing_streamlit
echo.
echo ==================================================
echo [ERROR] Streamlit is not installed for this Python.
echo.
echo Install the project dependencies with:
echo   "%PY_EXE%" %PY_ARGS% -m pip install -r requirements.txt
echo.
echo Or skip Python entirely and use Docker:
echo   run_windows_docker.bat
echo ==================================================
echo.
goto :end_pause

:no_python
echo.
echo ==================================================
echo [ERROR] No Python interpreter was found on this PC.
echo.
echo Option A - install Python 3.10 or newer:
echo   https://www.python.org/downloads/windows/
echo   Tick "Add python.exe to PATH" during installation.
echo.
echo Option B - skip Python entirely and use Docker:
echo   run_windows_docker.bat
echo ==================================================
echo.
goto :end_pause

:no_app
echo.
echo ==================================================
echo [ERROR] app.py was not found in this folder:
echo   %~dp0
echo Extract the complete PhasePix package and run this
echo launcher from the project root folder.
echo ==================================================
echo.
goto :end_pause

:end_pause
echo.
echo --------------------------------------------------
echo This window stays open so you can read the text above.
echo Press any key to close it.
echo --------------------------------------------------
pause >nul
endlocal
exit /b 0
