@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo ============================================================
echo  Hammy - GUI Launcher
echo ============================================================
echo.

set "ROOT=%~dp0"
set "APP=!ROOT!gui_app\app.py"

if exist "!APP!" goto :find_python
echo [ERROR] GUI entry not found: !APP!
pause
exit /b 1

:find_python
set "PY_CMD="
set "PY_IS_PORTABLE="

if exist "!ROOT!python\python.exe" (
    set "PY_CMD=!ROOT!python\python.exe"
    set "PY_IS_PORTABLE=1"
    echo [INFO] Using portable python: !PY_CMD!
    goto :check_python
)
if exist "!ROOT!python\pythonw.exe" (
    set "PY_CMD=!ROOT!python\pythonw.exe"
    set "PY_IS_PORTABLE=1"
    echo [WARN] python.exe not found, using pythonw.exe: !PY_CMD!
    goto :check_python
)

echo [INFO] Portable python not found, searching system PATH...
for /f "delims=" %%I in ('where python 2^>nul') do if not defined PY_CMD set "PY_CMD=%%I"
if not defined PY_CMD for /f "delims=" %%I in ('where python3 2^>nul') do if not defined PY_CMD set "PY_CMD=%%I"
if not defined PY_CMD goto :no_python
set "PY_IS_PORTABLE=0"

:check_python
"!PY_CMD!" -c "import sys" >nul 2>nul
if errorlevel 1 goto :bad_python
echo [INFO] Python ready: !PY_CMD!
goto :run

:no_python
echo [ERROR] Python not found.
echo        Please install Python 3.8+ or place a portable python
echo        at !ROOT!python\python.exe
echo.
pause
exit /b 1

:bad_python
echo [ERROR] Python interpreter is not usable: !PY_CMD!
echo        It may be the Microsoft Store stub. Install a real Python,
echo        or place a portable python at !ROOT!python\python.exe
echo.
pause
exit /b 1

:run
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set "PYTHONNET_RUNTIME="

echo Starting GUI (logs will appear in this window)...
echo.

"!PY_CMD!" "!APP!"
set "EXITCODE=!ERRORLEVEL!"

echo.
echo ============================================================
if "!EXITCODE!"=="0" (
    echo  GUI exited normally.
    exit /b 0
)
echo  [ERROR] GUI exited with code !EXITCODE!
echo  Check crash log: %TEMP%\video_rename_gui_crash.log
echo.
echo Press any key to close...
pause >nul
exit /b !EXITCODE!
