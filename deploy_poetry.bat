@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"

echo ============================================================
echo [Deploy] Downloading latest code from GitHub...
echo ============================================================
git pull origin main
if errorlevel 1 exit /b 1

set "POETRY_CMD=poetry"
where poetry > nul 2>&1
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\poetry.exe" (
        set "POETRY_CMD=%USERPROFILE%\.local\bin\poetry.exe"
    ) else (
        echo [Error] Poetry was not found.
        echo Install it with: python -m pipx install poetry
        exit /b 1
    )
)

echo.
echo [Deploy] Synchronizing Poetry dependencies...
call "%POETRY_CMD%" install --no-root --sync
if errorlevel 1 exit /b 1

echo.
echo [Deploy] Terminating existing server process...
wmic process where "name='python.exe' and commandline like '%%run_gui.py%%'" call terminate > nul 2>&1
timeout /t 2 /nobreak > nul

echo.
echo [Deploy] Restarting server on 0.0.0.0:8000...
start "PROJECT Studio" /D "%~dp0" "%~dp0.venv\Scripts\python.exe" "%~dp0run_gui.py" --host 0.0.0.0 --port 8000

echo [Deploy] Deployment completed!
endlocal
