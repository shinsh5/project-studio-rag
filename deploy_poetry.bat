@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"

echo ============================================================
echo [Deploy] Downloading latest code from GitHub...
echo ============================================================
git pull origin main
if errorlevel 1 exit /b 1

set "POETRY_CMD=F:\RAG\tools\poetry\Scripts\poetry.exe"
if not exist "%POETRY_CMD%" (
    echo [Error] Poetry was not found: %POETRY_CMD%
    exit /b 1
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
set "SERVER_LOG_DIR=F:\RAG\logs\project-studio-rag"
if not exist "%SERVER_LOG_DIR%" mkdir "%SERVER_LOG_DIR%"

rem Prevent the GitHub Actions runner from terminating the deployed server
rem when it cleans up child processes after the workflow job finishes.
set "RUNNER_TRACKING_ID="
start "PROJECT Studio" /B /D "%~dp0" "%~dp0.venv\Scripts\python.exe" "%~dp0run_gui.py" --host 0.0.0.0 --port 8000 1>>"%SERVER_LOG_DIR%\server.out.log" 2>>"%SERVER_LOG_DIR%\server.err.log"
if errorlevel 1 exit /b 1

echo [Deploy] Deployment completed!
endlocal
exit /b 0
