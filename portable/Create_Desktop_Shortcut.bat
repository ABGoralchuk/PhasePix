@echo off
setlocal EnableExtensions
title PhasePix - Create Desktop Shortcut
cd /d "%~dp0"

rem ===================================================================
rem  Creates a Desktop shortcut to PhasePix_Start.bat.
rem  Icon.ico is used automatically when it sits next to this script.
rem ===================================================================

set "ROOT=%~dp0"
set "TARGET=%ROOT%PhasePix_Start.bat"
set "ICON=%ROOT%Icon.ico"

echo ==================================================================
echo   PhasePix - Desktop Shortcut Creator
echo ==================================================================
echo.

if not exist "%TARGET%" goto :no_target

if not exist "%ICON%" goto :no_icon

echo Icon found: %ICON%
echo Creating the shortcut on your Desktop with the PhasePix icon...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop');$w=New-Object -ComObject WScript.Shell;$s=$w.CreateShortcut([IO.Path]::Combine($d,'PhasePix.lnk'));$s.TargetPath='%TARGET%';$s.WorkingDirectory='%ROOT%';$s.Description='PhasePix Portable - SEM Microstructure Analyzer';$s.IconLocation='%ICON%,0';$s.WindowStyle=1;$s.Save()"
if errorlevel 1 goto :failed
goto :done

:no_icon
echo Icon.ico was not found next to this script.
echo Creating the shortcut with the default Windows icon...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop');$w=New-Object -ComObject WScript.Shell;$s=$w.CreateShortcut([IO.Path]::Combine($d,'PhasePix.lnk'));$s.TargetPath='%TARGET%';$s.WorkingDirectory='%ROOT%';$s.Description='PhasePix Portable - SEM Microstructure Analyzer';$s.WindowStyle=1;$s.Save()"
if errorlevel 1 goto :failed
goto :done

:done
echo.
echo ==================================================================
echo [OK] Shortcut created on your Desktop: PhasePix
echo Double-click it to start PhasePix at any time.
echo ==================================================================
echo.
goto :end_pause

:failed
echo.
echo ==================================================================
echo [ERROR] PowerShell could not create the shortcut.
echo.
echo Workaround: right-click PhasePix_Start.bat, choose
echo "Send to - Desktop - create shortcut", then set the icon manually.
echo ==================================================================
echo.
goto :end_pause

:no_target
echo.
echo ==================================================================
echo [ERROR] PhasePix_Start.bat was not found next to this script:
echo   %TARGET%
echo.
echo Keep both files inside the same PhasePix_Portable folder.
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
