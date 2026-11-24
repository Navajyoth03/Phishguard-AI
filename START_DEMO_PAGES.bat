@echo off
title PhishGuard Demo Server
color 0B
echo ========================================================
echo   PHISHGUARD MULTI-MODAL DEMO ENVIRONMENT
echo ========================================================
echo.
echo Starting Demo Pages Web Server at: http://127.0.0.1:8000
echo (Ensure your AI backend server.py is also running on port 5000)
echo.

start http://127.0.0.1:8000/index.html

python demo_server.py
pause
