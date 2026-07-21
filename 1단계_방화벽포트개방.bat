@echo off
chcp 65001 > nul
echo ============================================================
echo [PROJECT Studio] Windows 방화벽 8000번 포트 개방 스크립트
echo ============================================================

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [경고] 관리자 권한이 필요합니다!
    echo 이 파일을 우클릭하고 '관리자 권한으로 실행'을 선택해주세요.
    pause
    exit /b 1
)

echo [진행 중] 8000번 포트(TCP) 인바운드 방화벽 규칙을 등록합니다...
powershell -Command "New-NetFirewallRule -DisplayName 'PROJECT Studio Web GUI (Port 8000)' -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow -Force"

if %errorLevel% equ 0 (
    echo.
    echo [성공] 8000번 포트가 완벽하게 개방되었습니다!
    echo 이제 같은 Wi-Fi에 연결된 기기에서 http://^<내-PC-IP^>:8000 으로 접속 가능합니다.
) else (
    echo [오류] 방화벽 규칙 등록에 실패했습니다.
)
pause
