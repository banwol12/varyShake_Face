@echo off
title NCT 127 Web Server - Shutdown
echo ==============================================
echo   Stopping NCT 127 Web Server (Port 5173)...
echo ==============================================

:: Find process ID listening on port 5173 (Node.js) and kill it
set "PID_NODE="
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING') do set PID_NODE=%%a

if "%PID_NODE%"=="" (
    echo No Node server found running on port 5173.
) else (
    taskkill /f /pid %PID_NODE%
    echo Node server on port 5173 successfully stopped.
)

:: Find process ID listening on port 5001 (Python 512D Service) and kill it
set "PID_PY="
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5001 ^| findstr LISTENING') do set PID_PY=%%a

if not "%PID_PY%"=="" (
    taskkill /f /pid %PID_PY%
    echo Python 512D Service on port 5001 successfully stopped.
)

pause
