@echo off
setlocal EnableExtensions
title PhasePix Portable Builder
cd /d "%~dp0"

rem ===================================================================
rem  PhasePix Portable Distribution Builder - Windows x64
rem
rem  Produces a self-contained, zero-install folder:
rem      dist\PhasePix_Portable
rem
rem  Contents: portable CPython 3.11, portable MinGit, a clean clone of
rem  this repository, all pip dependencies, pre-downloaded AI weights,
rem  and the end-user launchers built from the portable\ templates.
rem
rem  Flow control is goto-based on purpose: no nested parenthesised
rem  blocks, so a stray character can never break the script parser.
rem  Every exit path ends at :end_pause.
rem ===================================================================

set "ROOT=%~dp0"
set "DIST=%ROOT%dist\PhasePix_Portable"
set "CACHE=%ROOT%dist\_downloads"
set "PY_DIR=%DIST%\python"
set "GIT_DIR=%DIST%\git"
set "APP_DIR=%DIST%\app"
set "PY_EXE=%PY_DIR%\python.exe"
set "GIT_EXE=%GIT_DIR%\cmd\git.exe"
set "TEMPLATES=%ROOT%portable"
set "PY_URL=https://github.com/indygreg/python-build-standalone/releases/download/20240726/cpython-3.11.9%%2B20240726-x86_64-pc-windows-msvc-shared-install_only.tar.gz"
set "GIT_URL=https://github.com/git-for-windows/git/releases/download/v2.47.0.windows.1/MinGit-2.47.0-64-bit.zip"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
set "REPO_URL=https://github.com/ABGoralchuk/PhasePix.git"

echo ==================================================================
echo   PhasePix Portable Builder - Windows x64
echo ==================================================================
echo   Output folder : %DIST%
echo   Python        : CPython 3.11 portable, indygreg build
echo   Git           : MinGit 2.47.0 64-bit
echo   PyTorch       : CUDA 12.1 wheels, CPU fallback built in
echo ==================================================================
echo   This build downloads several gigabytes and can take 20-40 min.
echo ==================================================================
echo.

if not exist "%ROOT%app.py" goto :missing_project
if not exist "%ROOT%requirements.txt" goto :missing_project
if not exist "%TEMPLATES%\PhasePix_Start.bat" goto :missing_project
if not exist "%TEMPLATES%\Create_Desktop_Shortcut.bat" goto :missing_project

where powershell >nul 2>&1
if errorlevel 1 goto :no_powershell
where tar >nul 2>&1
if errorlevel 1 goto :no_tar

rem -------------------------------------------------------------------
rem [0/7] Prepare the output folders and the download cache
rem -------------------------------------------------------------------
echo [0/7] Preparing output folders...
if not exist "%ROOT%dist" mkdir "%ROOT%dist"
if not exist "%DIST%" mkdir "%DIST%"
if not exist "%CACHE%" mkdir "%CACHE%"
echo Output folders ready.
echo.

rem -------------------------------------------------------------------
rem [1/7] Portable CPython 3.11 - shared install_only build
rem -------------------------------------------------------------------
echo [1/7] Downloading portable CPython 3.11...
if exist "%PY_EXE%" goto :python_ready
if exist "%CACHE%\python.tar.gz" del /q "%CACHE%\python.tar.gz"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12;$ProgressPreference='SilentlyContinue';Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%CACHE%\python.tar.gz' -UseBasicParsing"
if errorlevel 1 goto :python_download_failed

echo Extracting portable CPython into %PY_DIR% ...
tar -xzf "%CACHE%\python.tar.gz" -C "%DIST%"
if errorlevel 1 goto :python_extract_failed
if not exist "%PY_EXE%" goto :python_extract_failed
del /q "%CACHE%\python.tar.gz" >nul 2>&1
echo Portable CPython is ready.
goto :python_done

:python_ready
echo Portable CPython already present - download skipped.

:python_done
echo.

rem -------------------------------------------------------------------
rem [2/7] Portable MinGit 2.47.0 - no Git installation required
rem -------------------------------------------------------------------
echo [2/7] Downloading portable MinGit 2.47.0...
if exist "%GIT_EXE%" goto :git_ready
if exist "%CACHE%\mingit.zip" del /q "%CACHE%\mingit.zip"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12;$ProgressPreference='SilentlyContinue';Invoke-WebRequest -Uri '%GIT_URL%' -OutFile '%CACHE%\mingit.zip' -UseBasicParsing"
if errorlevel 1 goto :git_download_failed

if not exist "%GIT_DIR%" mkdir "%GIT_DIR%"
echo Extracting portable MinGit into %GIT_DIR% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%CACHE%\mingit.zip' -DestinationPath '%GIT_DIR%' -Force"
if errorlevel 1 goto :git_extract_failed
if not exist "%GIT_EXE%" goto :git_extract_failed
del /q "%CACHE%\mingit.zip" >nul 2>&1
echo Portable MinGit is ready.
goto :git_done

:git_ready
echo Portable MinGit already present - download skipped.

:git_done
echo.

rem -------------------------------------------------------------------
rem [3/7] Clean clone of the PhasePix repository - .git is preserved so
rem      the portable launcher can auto-update from GitHub later.
rem -------------------------------------------------------------------
echo [3/7] Cloning PhasePix into %APP_DIR% ...
if exist "%APP_DIR%\.git" goto :clone_ready
if not exist "%APP_DIR%" goto :clone_now
echo Removing the incomplete app folder from a previous build...
rmdir /s /q "%APP_DIR%"

:clone_now
"%GIT_EXE%" clone "%REPO_URL%" "%APP_DIR%"
if errorlevel 1 goto :clone_failed
echo Repository cloned from GitHub.
goto :clone_done

:clone_ready
echo Repository clone already present - clone skipped.

:clone_done
echo.

rem -------------------------------------------------------------------
rem [4/7] Install the scientific + deep learning stack
rem -------------------------------------------------------------------
echo [4/7] Upgrading pip, setuptools and wheel...
"%PY_EXE%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :pip_bootstrap_failed

echo.
echo Installing PyTorch and Torchvision with CUDA 12.1 wheels...
echo The CUDA build still runs on CPU-only PCs, so one folder serves both.
"%PY_EXE%" -m pip install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 goto :torch_failed

echo.
echo Installing PhasePix requirements...
"%PY_EXE%" -m pip install --prefer-binary -r "%APP_DIR%\requirements.txt"
if errorlevel 1 goto :requirements_failed
echo.
echo All Python dependencies are installed.
echo.

rem -------------------------------------------------------------------
rem [5/7] Custom application icon
rem -------------------------------------------------------------------
echo [5/7] Copying the PhasePix icon...
if not exist "%ROOT%Icon.ico" goto :icon_missing
copy /y "%ROOT%Icon.ico" "%DIST%\Icon.ico" >nul
echo Icon.ico copied into the portable folder.
goto :icon_done

:icon_missing
echo Icon.ico not found in the repository root - the launcher shortcut
echo falls back to the default Windows icon.

:icon_done
echo.

rem -------------------------------------------------------------------
rem [6/7] Pre-download the base AI checkpoints
rem -------------------------------------------------------------------
echo [6/7] Pre-downloading SAM vit_b and Cellpose weights...
if not exist "%APP_DIR%\models_cache" mkdir "%APP_DIR%\models_cache"
set "CELLPOSE_LOCAL_MODELS_PATH=%APP_DIR%\models_cache"
set "TORCH_HOME=%APP_DIR%\models_cache"
"%PY_EXE%" -c "import sys;sys.path.insert(0,r'%APP_DIR%');import cellpose_model as C;C.ensure_cellpose_weights('cyto2');C.ensure_sam_checkpoint('vit_b')"
if errorlevel 1 goto :prefetch_warning
echo Base checkpoints are cached inside app\models_cache.
goto :prefetch_done

:prefetch_warning
echo [WARNING] The checkpoint pre-download did not finish.
echo PhasePix downloads the missing weights automatically on first run.

:prefetch_done
echo.

rem -------------------------------------------------------------------
rem [7/7] End-user launchers and first-run folders
rem -------------------------------------------------------------------
echo [7/7] Installing the end-user launchers...
copy /y "%TEMPLATES%\PhasePix_Start.bat" "%DIST%\PhasePix_Start.bat" >nul
if errorlevel 1 goto :launcher_failed
copy /y "%TEMPLATES%\Create_Desktop_Shortcut.bat" "%DIST%\Create_Desktop_Shortcut.bat" >nul
if errorlevel 1 goto :launcher_failed
if exist "%TEMPLATES%\PORTABLE_README.txt" copy /y "%TEMPLATES%\PORTABLE_README.txt" "%DIST%\PORTABLE_README.txt" >nul

if not exist "%APP_DIR%\models_cache" mkdir "%APP_DIR%\models_cache"
if not exist "%APP_DIR%\images" mkdir "%APP_DIR%\images"
if not exist "%APP_DIR%\images\sem" mkdir "%APP_DIR%\images\sem"
if not exist "%APP_DIR%\images\light" mkdir "%APP_DIR%\images\light"
copy /y "%APP_DIR%\requirements.txt" "%APP_DIR%\.portable_req_stamp" >nul
echo Launchers installed.
echo.

if exist "%CACHE%" rmdir /s /q "%CACHE%"

echo ==================================================================
echo   BUILD COMPLETE
echo ==================================================================
echo   Portable folder : %DIST%
echo.
echo   Next steps:
echo     1. Test it: double-click PhasePix_Start.bat
echo     2. Optional: run Create_Desktop_Shortcut.bat for a Desktop icon
echo     3. Copy the whole PhasePix_Portable folder to a USB stick or
echo        a network share and hand it to your colleagues
echo     4. They double-click PhasePix_Start.bat - no installs needed
echo.
echo   Updates are pulled from GitHub automatically on every launch.
echo ==================================================================
echo.
goto :end_pause

rem -------------------------------------------------------------------
rem Error paths - every one of them ends at :end_pause
rem -------------------------------------------------------------------
:missing_project
echo.
echo ==================================================================
echo [ERROR] Required project files are missing from this folder:
echo   %ROOT%
echo Expected next to this script:
echo   app.py
echo   requirements.txt
echo   portable\PhasePix_Start.bat
echo   portable\Create_Desktop_Shortcut.bat
echo Run this builder from the PhasePix project root.
echo ==================================================================
echo.
goto :end_pause

:no_powershell
echo.
echo ==================================================================
echo [ERROR] Windows PowerShell was not found.
echo PowerShell ships with Windows 10 and 11 and is required to download
echo the portable Python, MinGit and to create the Desktop shortcut.
echo ==================================================================
echo.
goto :end_pause

:no_tar
echo.
echo ==================================================================
echo [ERROR] The built-in tar.exe was not found.
echo Windows 10 version 1803 or newer is required for the portable build.
echo ==================================================================
echo.
goto :end_pause

:python_download_failed
echo.
echo ==================================================================
echo [ERROR] Downloading the portable CPython archive failed.
echo Useful checks:
echo   1. Internet access to github.com
echo   2. Antivirus or proxy blocking the download
echo   3. Free disk space in dist
echo Then run build_portable_windows.bat again.
echo ==================================================================
echo.
goto :end_pause

:python_extract_failed
echo.
echo ==================================================================
echo [ERROR] Extracting the portable CPython archive failed.
echo Expected file after extraction:
echo   %PY_EXE%
echo Delete the dist\PhasePix_Portable\python folder and re-run the builder.
echo ==================================================================
echo.
goto :end_pause

:git_download_failed
echo.
echo ==================================================================
echo [ERROR] Downloading the portable MinGit archive failed.
echo Useful checks:
echo   1. Internet access to github.com
echo   2. Free disk space in dist
echo Then run build_portable_windows.bat again.
echo ==================================================================
echo.
goto :end_pause

:git_extract_failed
echo.
echo ==================================================================
echo [ERROR] Extracting the portable MinGit archive failed.
echo Expected file after extraction:
echo   %GIT_EXE%
echo Delete the dist\PhasePix_Portable\git folder and re-run the builder.
echo ==================================================================
echo.
goto :end_pause

:clone_failed
echo.
echo ==================================================================
echo [ERROR] Cloning the PhasePix repository failed.
echo Useful checks:
echo   1. Internet access to github.com
echo   2. The portable MinGit folder was extracted correctly
echo   3. Delete dist\PhasePix_Portable\app and re-run the builder
echo ==================================================================
echo.
goto :end_pause

:pip_bootstrap_failed
echo.
echo ==================================================================
echo [ERROR] Upgrading pip, setuptools or wheel failed.
echo Check internet access to pypi.org, then re-run the builder.
echo ==================================================================
echo.
goto :end_pause

:torch_failed
echo.
echo ==================================================================
echo [ERROR] Installing PyTorch or Torchvision failed.
echo Useful checks:
echo   1. Internet access to download.pytorch.org
echo   2. Free disk space - the CUDA wheels need about 3 GB
echo   3. Antivirus blocking python.exe
echo ==================================================================
echo.
goto :end_pause

:requirements_failed
echo.
echo ==================================================================
echo [ERROR] Installing requirements.txt failed.
echo Scroll up to see which package failed, then re-run the builder -
echo packages already installed are simply skipped.
echo ==================================================================
echo.
goto :end_pause

:launcher_failed
echo.
echo ==================================================================
echo [ERROR] Copying the launcher templates failed.
echo Expected templates:
echo   %TEMPLATES%\PhasePix_Start.bat
echo   %TEMPLATES%\Create_Desktop_Shortcut.bat
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
