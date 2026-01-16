@echo off
chcp 65001 >nul
echo 🛡️ SilverGuard System Launching...
echo [INFO] Running in deployment mode.
echo ===========================================

cd /d "%~dp0"

echo [1/3] Checking environment...
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (venv) not found.
    echo Please run 'setup_deployment.bat' first.
    pause
    exit
)
call venv\Scripts\activate.bat

echo [2/3] Updating system status...
echo {"last_active": "%date% %time%"} > data\status.json

echo [3/3] Launching Modules...
echo    - Starting Detection Engine (Background)...
start /B python main.py

echo    - Starting Dashboard Interface...
streamlit run dashboard.py --server.headless true

pause
