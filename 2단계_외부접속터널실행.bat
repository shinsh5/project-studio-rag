@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

REM Read .env file
if exist .env (
    for /f "tokens=1,* delims==" %%A in (.env) do (
        if "%%A"=="NGROK_AUTHTOKEN" set NGROK_AUTHTOKEN=%%B
        if "%%A"=="NGROK_DOMAIN" set NGROK_DOMAIN=%%B
    )
)

if "%NGROK_AUTHTOKEN%"=="" (
    echo ============================================================
    echo [오류] .env 파일에 NGROK_AUTHTOKEN이 설정되어 있지 않습니다.
    echo 1. https://dashboard.ngrok.com/ 에 가입 또는 로그인합니다.
    echo 2. 좌측 메뉴 "Getting Started" ^> "Your Authtoken" 에서 토큰을 복사합니다.
    echo 3. .env 파일의 맨 아래에 "NGROK_AUTHTOKEN=복사한토큰" 을 추가하고 저장해주세요.
    echo ============================================================
    pause
    exit /b
)

if "%NGROK_DOMAIN%"=="" (
    echo ============================================================
    echo [오류] .env 파일에 NGROK_DOMAIN이 설정되어 있지 않습니다.
    echo 1. ngrok 대시보드의 좌측 메뉴 "Cloud Edge" ^> "Domains" 로 이동합니다.
    echo 2. 발급받은 무료 고정 도메인을 확인합니다. 예: xxx-yyy-zzz.ngrok-free.app
    echo 3. .env 파일의 맨 아래에 "NGROK_DOMAIN=고정도메인" 을 추가하고 저장해주세요.
    echo ============================================================
    pause
    exit /b
)

echo ============================================================
echo [PROJECT Studio] External Access Tunnel (ngrok)
echo ============================================================
echo Target URL: https://%NGROK_DOMAIN%
echo ============================================================
echo.

echo [Ngrok] 토큰 설정 적용 중...
call ngrok config add-authtoken %NGROK_AUTHTOKEN% > nul

:loop
echo [Connecting...] Starting ngrok on port 8000...
ngrok http --domain=%NGROK_DOMAIN% 8000
echo.
echo [Notice] Tunnel disconnected or timed out. Reconnecting in 3 seconds...
timeout /t 3 > nul
goto loop
