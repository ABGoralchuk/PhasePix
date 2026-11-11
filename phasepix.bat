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
rem 0. One-click auto-update: bring this folder to origin/main.
rem    -c safe.directory="%CD%" avoids the Windows "dubious ownership"
rem    error on copied folders and USB sticks. Preset JSONs are no longer
rem    tracked in Git and any local copy is backed up + restored here, so a
rem    lab PC can never be blocked by its own saved recipes.
rem -----------------------------------------------------------------
if not exist ".git" goto :skip_update
where git >nul 2>&1
if errorlevel 1 goto :skip_update
echo Checking for PhasePix updates...
if exist "parameter_presets.json" copy /y "parameter_presets.json" "parameter_presets.json.upd" >nul 2>&1
if exist "light_calibration_presets.json" copy /y "light_calibration_presets.json" "light_calibration_presets.json.upd" >nul 2>&1
if exist "clsm_calibration_presets.json" copy /y "clsm_calibration_presets.json" "clsm_calibration_presets.json.upd" >nul 2>&1
git -c safe.directory="%CD%" fetch --all --quiet 2>nul
if errorlevel 1 echo Update check skipped - offline or git unavailable.
if errorlevel 1 goto :restore_user_presets
git -c safe.directory="%CD%" merge-base --is-ancestor HEAD origin/main >nul 2>&1
if not errorlevel 1 goto :fast_forward_reset
git -c safe.directory="%CD%" pull --ff-only origin main 2>nul
goto :restore_user_presets
:fast_forward_reset
git -c safe.directory="%CD%" reset --hard origin/main >nul 2>&1
:restore_user_presets
if exist "parameter_presets.json.upd" copy /y "parameter_presets.json.upd" "parameter_presets.json" >nul 2>&1
if exist "parameter_presets.json.upd" del /q "parameter_presets.json.upd" >nul 2>&1
if exist "light_calibration_presets.json.upd" copy /y "light_calibration_presets.json.upd" "light_calibration_presets.json" >nul 2>&1
if exist "light_calibration_presets.json.upd" del /q "light_calibration_presets.json.upd" >nul 2>&1
if exist "clsm_calibration_presets.json.upd" copy /y "clsm_calibration_presets.json.upd" "clsm_calibration_presets.json" >nul 2>&1
if exist "clsm_calibration_presets.json.upd" del /q "clsm_calibration_presets.json.upd" >nul 2>&1
echo.
:skip_update

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

rem ---- Free port 8501: a stale server from a previous session would keep
rem      serving the OLD code and make the update look like it "did nothing".
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do taskkill /f /pid %%P >nul 2>&1
ping -n 2 127.0.0.1 >nul

echo.
echo Launching PhasePix - the dashboard opens at http://localhost:8501
echo Press Ctrl+C in this window to stop the server.
echo.

"%PY_EXE%" %PY_ARGS% -m streamlit run app.py --server.port=8501 --server.headless=false --browser.gatherUsageStats=false
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
echo Or ask your PhasePix administrator for the zero-install
echo PhasePix_Portable folder and run PhasePix_Start.bat
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
echo Option B - ask your PhasePix administrator for the zero-install
echo PhasePix_Portable folder and run PhasePix_Start.bat
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
