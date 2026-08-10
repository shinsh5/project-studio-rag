@echo off
chcp 65001 > nul
echo ============================================================
echo [Deploy] Downloading latest code from GitHub...
echo ============================================================
git pull origin main

echo.
echo [Deploy] Terminating existing server process...
REM Terminate existing run_gui.py process
wmic process where "name='python.exe' and commandline like '%%run_gui.py%%'" call terminate > nul 2>&1

REM Wait 2 seconds
timeout /t 2 /nobreak > nul

echo.
echo [Deploy] Restarting server...
start "" run_gui.bat

echo [Deploy] Deployment completed!
