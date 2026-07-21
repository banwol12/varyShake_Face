@echo off
title NCT 127 Web Server - Shutdown
echo ==============================================
echo   Stopping NCT 127 Web Server (Port 5173)...
echo ==============================================

:: Find process ID listening on port 5173 and kill it
set "PID="
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING') do set PID=%%a

if "%PID%"=="" (
    echo No server found running on port 5173.
) else (
    taskkill /f /pid %PID%
    echo Server on port 5173 successfully stopped.
)

pause
