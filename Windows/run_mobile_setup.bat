@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Smart Waste • Setup dipendenze

cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Creo virtualenv .venv ...
  (py -3 -m venv .venv) || (python -m venv .venv) || (echo ERRORE: Python 3 non trovato.& pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"

echo Aggiorno pip...
python -m pip install --upgrade pip setuptools wheel

if exist requirements.txt (
  echo Installo requirements...
  python -m pip install -r requirements.txt
)

rem Fallback: assicurati dei moduli core se requirements ha fallito
for %%P in (Flask Flask-Login Flask-WTF flask-cors python-dotenv pymongo pillow email-validator) do (
  python - <<#PY 1>nul 2>nul
import importlib,sys
sys.exit(0 if importlib.util.find_spec("%%~P".replace("-","_")) else 1)
#PY
  if errorlevel 1 python -m pip install "%%~P"
)

rem PyTorch CPU (se mancante)
python - <<#PY
import importlib,sys
sys.exit(0 if importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision') else 1)
#PY
if errorlevel 1 (
  echo Installo PyTorch CPU...
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
)

echo.
echo Setup completato.
pause
