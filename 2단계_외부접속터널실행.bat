@echo off
cls
echo ============================================================
echo [PROJECT Studio] External Access Tunnel (loca.lt)
echo ============================================================
echo Target URL: https://project-studio-rag.loca.lt
echo ============================================================
echo.

:loop
echo [Connecting...] Starting localtunnel on port 8000...
call npx -y localtunnel --port 8000 --subdomain project-studio-rag
echo.
echo [Notice] Tunnel disconnected or timed out. Reconnecting in 3 seconds...
timeout /t 3 > nul
goto loop
