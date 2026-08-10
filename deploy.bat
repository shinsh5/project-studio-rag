@echo off
chcp 65001 > nul
echo ============================================================
echo [Deploy] GitHub에서 최신 코드를 다운로드합니다...
echo ============================================================
git pull origin main

echo.
echo [Deploy] 기존 서버 프로세스를 찾아 종료합니다...
:: 기존 run_gui.py를 실행 중인 프로세스 종료 (에러 메시지는 숨김)
wmic process where "name='python.exe' and commandline like '%%run_gui.py%%'" call terminate > nul 2>&1

:: 프로세스가 안전하게 종료될 때까지 2초 대기
timeout /t 2 /nobreak > nul

echo.
echo [Deploy] 최신 코드로 서버를 다시 시작합니다...
:: 백그라운드 새 창에서 run_gui.bat 실행
start "" run_gui.bat

echo [Deploy] 배포가 완료되었습니다!
