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
call "%POETRY_CMD%" sync --no-root
if errorlevel 1 exit /b 1

echo.
echo [Deploy] Terminating existing server process...
set "SERVER_LOG_DIR=F:\RAG\logs\project-studio-rag"
set "SERVER_PID_FILE=%SERVER_LOG_DIR%\server.pid"
if not exist "%SERVER_LOG_DIR%" mkdir "%SERVER_LOG_DIR%"

if exist "%SERVER_PID_FILE%" (
    powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$serverPid = [int](Get-Content -LiteralPath '%SERVER_PID_FILE%'); $process = Get-Process -Id $serverPid -ErrorAction SilentlyContinue; if ($process -and $process.ProcessName -eq 'python') { & taskkill.exe /PID $serverPid /T /F | Out-Null; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }; Remove-Item -LiteralPath '%SERVER_PID_FILE%' -Force -ErrorAction SilentlyContinue"
    if errorlevel 1 exit /b 1
)
timeout /t 2 /nobreak > nul

echo.
echo [Deploy] Restarting server on 0.0.0.0:8000...
rem Prevent the GitHub Actions runner from terminating the deployed server
rem when it cleans up child processes after the workflow job finishes.
set "RUNNER_TRACKING_ID="
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$process = Start-Process -FilePath '%~dp0.venv\Scripts\python.exe' -ArgumentList @('%~dp0run_gui.py', '--host', '0.0.0.0', '--port', '8000') -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput '%SERVER_LOG_DIR%\server.out.log' -RedirectStandardError '%SERVER_LOG_DIR%\server.err.log' -PassThru; [IO.File]::WriteAllText('%SERVER_PID_FILE%', [string]$process.Id)"
if errorlevel 1 exit /b 1

echo [Deploy] Deployment completed!
endlocal
exit /b 0
