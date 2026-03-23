@echo off
echo ============================================
echo   AirWriting Backend Services (Windows)
echo ============================================

:: ── Step 1: Kill any existing Python processes on our ports ──
echo.
echo [1/4] Cleaning up zombie processes (PowerShell)...
powershell -ExecutionPolicy Bypass -File "%~dp0kill_ports.ps1"
echo   Done.

:: ── Step 2: Wait for ports to fully release ──
echo.
echo [2/4] Waiting for ports to release...
timeout /t 2 /nobreak >nul

:: ── Step 3: Launch all services (with venv activation) ──
echo.
echo [3/4] Launching services...

:: Start IMU Engine
start "IMU Engine" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python main.py --load-bias"
echo   Started: IMU Engine (main.py)

:: Wait briefly for IMU engine to bind port 12345
timeout /t 1 /nobreak >nul

:: Start WebSocket Relay
start "WebSocket Relay" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python tools\web_relay.py"
echo   Started: WebSocket Relay (web_relay.py)

:: Wait briefly for relay to bind port 12346+18765
timeout /t 1 /nobreak >nul

:: Start Flask Web App
start "Flask Web App" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python platform_app\app.py"
echo   Started: Flask ML API (app.py)

:: Wait briefly
timeout /t 1 /nobreak >nul

:: Start Action Dispatcher
start "Action Dispatcher" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python tools\action_dispatcher.py"
echo   Started: Action Dispatcher (action_dispatcher.py)

:: ── Step 4: Done ──
echo.
echo ============================================
echo [4/4] All 4 services launched successfully!
echo ============================================
echo.
echo   IMU Engine:       UDP :12345 (from ESP32)
echo   WebSocket Relay:  UDP :12346 / WS :18765
echo   Flask ML API:     HTTP :5000
echo   Action Dispatch:  UDP :12348 / WS :18800 (to Phone)
echo.
echo [5/5] Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul
start http://localhost:5000
echo.
echo Close each CMD window to stop its service.
pause
