@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)
echo Ambiente attivo.
echo.
call "%~dp0run_mobile.bat"
echo.
echo [DEBUG] Fine script. Premi un tasto per chiudere...
pause >nul
