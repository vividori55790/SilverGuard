@echo off
chcp 65001 > nul
echo 🚀 SilverGuard AI System Starting...
echo ===================================
echo [1] Launching Dashboard UI...
cd /d "%~dp0"
start "SilverGuard Control Panel" streamlit run SilverGuard\dashboard.py

echo [2] Launching AI Engine (Core)...
start "SilverGuard Engine (Do Not Close)" python SilverGuard\main.py

echo ===================================
echo ✅ System Initialized.
echo - Close the console windows to stop the system.
echo - Access Dashboard at http://localhost:8501
pause
