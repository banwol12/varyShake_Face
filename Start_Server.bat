@echo off
title NCT 127 Web Server - Startup
cd /d "%~dp0\varyshake"

echo ==============================================
echo   NCT 127 Web Server Starting...
echo ==============================================

:: Open the web browser automatically
echo Launching browser: http://localhost:5173
start http://localhost:5173

:: Start the server
echo Starting server...
node server.js

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start the server.
    echo Please check if Node.js is installed and port 5173 is free.
    pause
)
