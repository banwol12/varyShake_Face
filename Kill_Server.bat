@echo off
title NCT 127 Web Server - Shutdown
echo ==============================================
echo   Stopping NCT 127 Web Server (Port 5173 & 5001)...
echo ==============================================

:: Find process ID listening on port 5173 (Node.js) and kill it
set "PID_5173="
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING') do set PID_5173=%%a

if "%PID_5173%"=="" (
    echo No Node.js server found running on port 5173.
) else (
    taskkill /f /pid %PID_5173%
    echo Server on port 5173 successfully stopped.
)

:: Find process ID listening on port 5001 (Python 512D Service) and kill it
set "PID_5001="
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5001 ^| findstr LISTENING') do set PID_5001=%%a

if "%PID_5001%"=="" (
    echo No Python 512D service found running on port 5001.
) else (
    taskkill /f /pid %PID_5001%
    echo Python 512D service on port 5001 successfully stopped.
)

pause
