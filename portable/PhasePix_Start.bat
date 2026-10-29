@echo off
setlocal EnableExtensions
title PhasePix Portable - SEM Microstructure Analyzer
cd /d "%~dp0"

rem ===================================================================
rem  PhasePix Portable launcher - zero-install Windows entry point.
rem  Every failure path ends at :end_pause so the window never closes.
rem ===================================================================

set "ROOT=%~dp0"
set "PY_EXE=%ROOT%python\python.exe"
set "GIT_EXE=%ROOT%git\cmd\git.exe"
set "APP_DIR=%ROOT%app"
set "MODELS_DIR=%APP_DIR%\models_cache"

rem ---- Portable environment: bundled Python, bundled Git, local weights ----
set "PATH=%ROOT%python;%ROOT%python\Scripts;%ROOT%git\cmd;%PATH%"
set "CELLPOSE_LOCAL_MODELS_PATH=%MODELS_DIR%"
set "TORCH_HOME=%MODELS_DIR%"
set "STREAMLIT_SERVER_HEADLESS=false"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
rem Never let a credential prompt freeze an unattended double-click launch.
set "GIT_TERMINAL_PROMPT=0"

echo ==================================================================
echo   PhasePix Portable - SEM Microstructure Analyzer
echo   Everything needed is inside this folder. Nothing is installed.
echo ==================================================================
echo.

if not exist "%PY_EXE%" goto :no_python
if not exist "%APP_DIR%\app.py" goto :no_app

rem -------------------------------------------------------------------
rem [1/3] Non-destructive auto-update from GitHub.
rem User-edited preset JSON files are backed up first and restored after
rem the hard reset, so lab recipes are never lost.
rem -------------------------------------------------------------------
echo [1/3] Checking for PhasePix updates...
if not exist "%GIT_EXE%" goto :no_git
if not exist "%APP_DIR%\.git" goto :no_git

cd /d "%APP_DIR%"
if exist "parameter_presets.json" copy /y "parameter_presets.json" "parameter_presets.json.tmp" >nul 2>&1
if exist "light_calibration_presets.json" copy /y "light_calibration_presets.json" "light_calibration_presets.json.tmp" >nul 2>&1

rem A stalled connection aborts after 60 s of no data instead of hanging the launch.
"%GIT_EXE%" -c safe.directory="%APP_DIR%" -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --all --quiet
if errorlevel 1 goto :offline

"%GIT_EXE%" -c safe.directory="%APP_DIR%" reset --hard origin/main >nul 2>&1
if errorlevel 1 goto :try_master
goto :restore_presets

:try_master
"%GIT_EXE%" -c safe.directory="%APP_DIR%" reset --hard origin/master >nul 2>&1
if errorlevel 1 goto :update_failed

:restore_presets
if not exist "parameter_presets.json.tmp" goto :restore_light
copy /y "parameter_presets.json.tmp" "parameter_presets.json" >nul 2>&1
del /q "parameter_presets.json.tmp" >nul 2>&1

:restore_light
if not exist "light_calibration_presets.json.tmp" goto :updated
copy /y "light_calibration_presets.json.tmp" "light_calibration_presets.json" >nul 2>&1
del /q "light_calibration_presets.json.tmp" >nul 2>&1

:updated
echo Code synchronised with GitHub:
"%GIT_EXE%" log -1 --oneline 2>nul
goto :req_check

:offline
echo [INFO] Offline mode - starting current local version.
goto :restore_presets_offline

:update_failed
echo [WARNING] GitHub was reachable but the update failed.
echo [INFO] Offline mode - starting current local version.

:restore_presets_offline
if not exist "parameter_presets.json.tmp" goto :restore_light_offline
copy /y "parameter_presets.json.tmp" "parameter_presets.json" >nul 2>&1
del /q "parameter_presets.json.tmp" >nul 2>&1

:restore_light_offline
if not exist "light_calibration_presets.json.tmp" goto :req_check
copy /y "light_calibration_presets.json.tmp" "light_calibration_presets.json" >nul 2>&1
del /q "light_calibration_presets.json.tmp" >nul 2>&1
goto :req_check

:no_git
echo [WARNING] Portable Git or the .git folder is missing.
echo [INFO] Skipping auto-update and starting the local version.

rem -------------------------------------------------------------------
rem [2/3] Install dependencies only when requirements.txt changed.
rem -------------------------------------------------------------------
:req_check
echo.
echo [2/3] Verifying Python dependencies...
cd /d "%APP_DIR%"
if not exist ".portable_req_stamp" goto :req_install
fc /b ".portable_req_stamp" "requirements.txt" >nul 2>&1
if not errorlevel 1 goto :req_ready

:req_install
echo requirements.txt changed - installing dependencies...
"%PY_EXE%" -m pip install --prefer-binary -r "requirements.txt"
if errorlevel 1 goto :req_failed
copy /y "requirements.txt" ".portable_req_stamp" >nul 2>&1
goto :req_ready

:req_failed
echo [WARNING] Dependency installation failed - trying to start anyway.
echo Check the internet connection and run this launcher again.

rem -------------------------------------------------------------------
rem [3/3] Launch the Streamlit dashboard.
rem -------------------------------------------------------------------
:req_ready
echo.
echo [3/3] Starting PhasePix...
echo The dashboard opens automatically at http://localhost:8501
echo Keep this window open while you work. Press Ctrl+C to stop.
echo.
"%PY_EXE%" -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false
if errorlevel 1 goto :run_failed

echo.
echo PhasePix was closed. No errors were reported.
goto :end_pause

rem -------------------------------------------------------------------
rem Error paths - every one of them ends at :end_pause
rem -------------------------------------------------------------------
:run_failed
echo.
echo ==================================================================
echo [ERROR] PhasePix exited with an error.
echo Review the traceback printed above.
echo.
echo Useful checks:
echo   1. Antivirus did not quarantine python\python.exe
echo   2. The folder is on a local drive, not a network share
echo   3. Free disk space in app\models_cache
echo   4. Run PhasePix_Start.bat again
echo ==================================================================
echo.
goto :end_pause

:no_python
echo.
echo ==================================================================
echo [ERROR] The bundled Python interpreter is missing:
echo   %PY_EXE%
echo.
echo Re-run build_portable_windows.bat on a build PC, then copy the
echo complete PhasePix_Portable folder again.
echo ==================================================================
echo.
goto :end_pause

:no_app
echo.
echo ==================================================================
echo [ERROR] The application folder is incomplete - app.py not found in:
echo   %APP_DIR%
echo.
echo Re-run build_portable_windows.bat on a build PC, then copy the
echo complete PhasePix_Portable folder again.
echo ==================================================================
echo.
goto :end_pause

:end_pause
echo.
echo ------------------------------------------------------------------
echo This window stays open so you can read the messages above.
echo Press any key to close it.
echo ------------------------------------------------------------------
pause >nul
endlocal
exit /b 0
