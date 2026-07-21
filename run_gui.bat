@echo off
chcp 65001 > nul
echo ============================================================
echo [PROJECT Studio] GUI 모드 서버 및 브라우저를 시작합니다...
echo ============================================================

set PY_CMD=python
if exist ".\.venv\Scripts\python.exe" (
    set PY_CMD=.\.venv\Scripts\python.exe
) else if exist "..\rag_poc\.venv\Scripts\python.exe" (
    set PY_CMD=..\rag_poc\.venv\Scripts\python.exe
)

%PY_CMD% run_gui.py
if %errorlevel% neq 0 (
    echo [오류] 실행 중 문제가 발생했습니다. 파이썬 가상환경 및 의존성이 설치되었는지 확인하세요.
    pause
)
