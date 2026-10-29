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
echo [1/6] Downloading Portable Python 3.11...
curl -L -o "%DEST%\python.tar.gz" "https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.11.7+20240107-x86_64-pc-windows-msvc-shared-install_only.tar.gz"
tar -xzf "%DEST%\python.tar.gz" -C "%DEST%"
del "%DEST%\python.tar.gz"

echo.
echo [2/6] Downloading Portable Git...
curl -L -o "%DEST%\mingit.zip" "https://github.com/git-for-windows/git/releases/download/v2.47.0.windows.1/MinGit-2.47.0-64-bit.zip"
mkdir "%DEST%\git"
tar -xf "%DEST%\mingit.zip" -C "%DEST%\git"
del "%DEST%\mingit.zip"

echo.
echo [3/6] Copying PhasePix Application...
"%DEST%\git\cmd\git.exe" clone "%~dp0." "%DEST%\app"

echo.
echo [4/6] Copying Icon...
if exist "%~dp0Icon.ico" (
    copy /Y "%~dp0Icon.ico" "%DEST%\Icon.ico"
) else if exist "%~dp0icon.ico" (
    copy /Y "%~dp0icon.ico" "%DEST%\Icon.ico"
)

echo.
echo [5/6] Creating Launchers...
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
echo [6/6] Installing PyTorch and Libraries (This takes the longest)...
cd /d "%DEST%\python"
python.exe -m pip install --upgrade pip setuptools wheel
python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python.exe -m pip install -r "%DEST%\app\requirements.txt" --prefer-binary

echo.
echo ========================================================
echo Build Complete!
echo The portable folder is located at: dist\PhasePix_Portable
echo ========================================================
pause