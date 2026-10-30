@echo off
setlocal EnableExtensions
title PhasePix Portable Builder
echo ========================================================
echo Building PhasePix Portable (This will take 5-15 minutes)
echo ========================================================

set "DEST=%~dp0dist\PhasePix_Portable"
if exist "%DEST%" rmdir /s /q "%DEST%"
mkdir "%DEST%"

echo.
echo [1/7] Downloading Portable Python 3.11...
curl -L -o "%DEST%\python.tar.gz" "https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.11.7+20240107-x86_64-pc-windows-msvc-shared-install_only.tar.gz"
tar -xzf "%DEST%\python.tar.gz" -C "%DEST%"
del "%DEST%\python.tar.gz"

echo.
echo [2/7] Downloading Portable Git...
curl -L -o "%DEST%\mingit.zip" "https://github.com/git-for-windows/git/releases/download/v2.47.0.windows.1/MinGit-2.47.0-64-bit.zip"
mkdir "%DEST%\git"
tar -xf "%DEST%\mingit.zip" -C "%DEST%\git"
del "%DEST%\mingit.zip"

echo.
echo [3/7] Cloning PhasePix Application directly from GitHub...
"%DEST%\git\cmd\git.exe" clone https://github.com/ABGoralchuk/PhasePix.git "%DEST%\app"

echo.
echo [4/7] Creating Launchers and Convenience Folders...
if exist "%~dp0Icon.ico" copy /Y "%~dp0Icon.ico" "%DEST%\Icon.ico" >nul

(
echo @echo off
echo explorer.exe "%%~dp0app\images"
echo exit /b 0
) > "%DEST%\Open_Images_Folder.bat"

(
echo @echo off
echo setlocal
echo title PhasePix Portable
echo set "PATH=%%~dp0python;%%~dp0python\Scripts;%%~dp0git\cmd;%%PATH%%"
echo set "CELLPOSE_LOCAL_MODELS_PATH=%%~dp0app\models_cache"
echo set "TORCH_HOME=%%~dp0app\models_cache"
echo echo [1/2] Checking for PhasePix updates...
echo cd /d "%%~dp0app"
echo if exist parameter_presets.json copy /Y parameter_presets.json parameter_presets.tmp ^>nul
echo git fetch --all
echo git reset --hard origin/main ^|^| git reset --hard origin/master
echo if exist parameter_presets.tmp move /Y parameter_presets.tmp parameter_presets.json ^>nul
echo echo [2/2] Starting PhasePix...
echo "%%~dp0python\python.exe" -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false
echo pause
) > "%DEST%\PhasePix_Start.bat"

(
echo @echo off
echo set "SCRIPT_DIR=%%~dp0"
echo set "TARGET=%%SCRIPT_DIR%%PhasePix_Start.bat"
echo set "ICON=%%SCRIPT_DIR%%Icon.ico"
echo set "SHORTCUT=%%USERPROFILE%%\Desktop\PhasePix.lnk"
echo powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%%SHORTCUT%%'); $s.TargetPath = '%%TARGET%%'; $s.WorkingDirectory = '%%SCRIPT_DIR%%'; if (Test-Path '%%ICON%%') { $s.IconLocation = '%%ICON%%' }; $s.Save()"
echo echo [OK] Shortcut created on Desktop!
echo pause
) > "%DEST%\Create_Desktop_Shortcut.bat"

echo.
echo [5/7] Copying Cached AI Models...
if exist "%~dp0models_cache" xcopy /E /I /Y "%~dp0models_cache" "%DEST%\app\models_cache" >nul

echo.
echo [6/7] Installing PyTorch and Libraries (This takes the longest)...
cd /d "%DEST%\python"
python.exe -m pip install --upgrade pip setuptools wheel

echo.
echo   -- Installing the CUDA 12.1 build of PyTorch --
python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo   -- Installing the remaining PhasePix libraries --
echo      torch / torchvision are unconstrained in requirements.txt, so pip
echo      keeps the +cu121 wheels that were just installed.
python.exe -m pip install -r "%DEST%\app\requirements.txt" --prefer-binary --upgrade-strategy only-if-needed

echo.
echo [7/7] Verifying that the CUDA PyTorch wheels are still installed...
python.exe -c "import sys, torch; print('PyTorch', torch.__version__, 'CUDA build:', torch.version.cuda or 'CPU-only'); sys.exit(0 if torch.version.cuda else 1)" || goto cuda_broken
echo [OK] CUDA-enabled PyTorch verified.
goto build_done

:cuda_broken
echo [WARN] The CUDA PyTorch wheels were replaced by CPU wheels.
echo        Reinstalling the CUDA 12.1 build...
python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
python.exe -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)" || goto cuda_failed
echo [OK] CUDA-enabled PyTorch restored.

:build_done
echo.
echo ========================================================
echo Build Complete!
echo The portable folder is located at: dist\PhasePix_Portable
echo ========================================================
pause
exit /b 0

:cuda_failed
echo ========================================================
echo [ERROR] Could not restore CUDA-enabled PyTorch.
echo         PhasePix still works, but this portable build will
echo         run in CPU mode only. Check the internet connection
echo         and re-run build_portable_windows.bat.
echo ========================================================
pause
exit /b 1
