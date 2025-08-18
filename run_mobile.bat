@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Smart Waste • Avvio per Mobile (LAN)

cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Manca l'ambiente. Avvia prima run_mobile_setup.bat
  pause & exit /b 1
)
call ".venv\Scripts\activate.bat"

rem Host/porta
set "FLASK_RUN_HOST=0.0.0.0"
set "FLASK_RUN_PORT=5000"

rem IP LAN (PowerShell)
for /f %%I in ('powershell -NoP -C "(Get-NetIPAddress -AddressFamily IPv4 ^| ?{ $_.IPAddress -notlike ''127.*'' -and $_.IPAddress -notlike ''169.254.*'' } ^| Select -First 1 -Expand IPAddress)"') do set "LANIP=%%I"
if not defined LANIP set "LANIP=127.0.0.1"

echo.
echo PC:     http://127.0.0.1:%FLASK_RUN_PORT%/app
echo Mobile: http://%LANIP%:%FLASK_RUN_PORT%/app
echo.
echo Se il firewall chiede permesso, consenti sulle "Reti Private".
echo Lascia questa finestra aperta. Chiudi con CTRL+C.
echo.

rem Avvio in questa stessa finestra e NON chiuderla mai:
cmd /k python -u app.py
