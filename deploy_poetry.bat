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

rem A venv Python launcher and Uvicorn reload can leave child processes behind.
rem Port 8000 is dedicated to this service, so terminate any remaining Python
rem listener before starting the new revision.
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$listenerPids = @(Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique); foreach ($listenerPid in $listenerPids) { $process = Get-Process -Id $listenerPid -ErrorAction SilentlyContinue; if (-not $process) { $orphanChildren = @(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $listenerPid }); if (-not $orphanChildren) { Write-Error ('Port 8000 is owned by unavailable PID {0}, and no child process could be resolved.' -f $listenerPid); exit 1 }; foreach ($child in $orphanChildren) { if ($child.Name -notmatch '^pythonw?\.exe$') { Write-Error ('Port 8000 socket from PID {0} is inherited by non-Python process {1} ({2}).' -f $listenerPid, $child.ProcessId, $child.Name); exit 1 }; & taskkill.exe /PID $child.ProcessId /T /F | Out-Null; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }; continue }; if ($process.ProcessName -notmatch '^pythonw?$') { Write-Error ('Port 8000 is occupied by non-Python process {0} ({1}).' -f $listenerPid, $process.ProcessName); exit 1 }; & taskkill.exe /PID $listenerPid /T /F | Out-Null; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }; $deadline = (Get-Date).AddSeconds(15); while ((Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }; if (Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue) { Write-Error 'Port 8000 is still occupied after stopping the previous server.'; exit 1 }"
if errorlevel 1 exit /b 1

echo.
echo [Deploy] Restarting server on 127.0.0.1:8000...
rem Prevent the GitHub Actions runner from terminating the deployed server
rem when it cleans up child processes after the workflow job finishes.
set "RUNNER_TRACKING_ID="
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$process = Start-Process -FilePath '%~dp0.venv\Scripts\python.exe' -ArgumentList @('%~dp0run_gui.py', '--host', '127.0.0.1', '--port', '8000') -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput '%SERVER_LOG_DIR%\server.out.log' -RedirectStandardError '%SERVER_LOG_DIR%\server.err.log' -PassThru; [IO.File]::WriteAllText('%SERVER_PID_FILE%', [string]$process.Id)"
if errorlevel 1 exit /b 1

echo [Deploy] Waiting for the server health check...
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$deadline = (Get-Date).AddSeconds(90); do { try { $openApi = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/openapi.json' -TimeoutSec 5; if ($openApi.paths.PSObject.Properties.Name -contains '/api/evaluate-single-stream') { exit 0 } } catch {}; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); exit 1"
if errorlevel 1 (
    echo [Error] Server did not become healthy on port 8000 within 90 seconds.
    exit /b 1
)

echo [Deploy] Deployment completed!
endlocal
exit /b 0
