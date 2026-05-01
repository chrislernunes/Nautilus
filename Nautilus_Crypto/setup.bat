@echo off
REM setup.bat — Windows one-shot setup for Nautilus BTC
REM Run from the nautilus_btc\ directory in PowerShell or CMD:
REM   .\setup.bat

echo.
echo ============================================================
echo   NAUTILUS CRYPTO  --  Windows Setup
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: python not found. Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

FOR /F "tokens=*" %%i IN ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') DO SET PY_VER=%%i
echo Found Python %PY_VER%

REM Create venv
IF NOT EXIST ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate
call .venv\Scripts\activate.bat
echo Virtual environment activated.

REM Upgrade pip
python -m pip install --upgrade pip --quiet

REM Install all dependencies
echo.
echo Installing Python dependencies...
pip install -r requirements.txt
IF ERRORLEVEL 1 (
    echo ERROR: pip install failed. Check your internet connection.
    pause
    exit /b 1
)
echo Dependencies installed.

REM Create directories
if not exist "logs"   mkdir logs
if not exist "models" mkdir models

echo.
echo ============================================================
echo   Setup complete!
echo.
echo   Activate venv:      .venv\Scripts\activate
echo   Smoke test:         python scripts\smoke_test.py
echo   Run tests:          python -m pytest tests\ -v
echo   Launch dashboard:   python main.py
echo   Backtest only:      python main.py --backtest
echo ============================================================
echo.
pause
