@echo off
chcp 65001 >nul
echo 🛡️ SilverGuard Deployment Setup
echo ===========================================

cd /d "%~dp0"

if exist "venv" (
    echo [INFO] Existing environment found. Using it.
) else (
    echo [INFO] Creating new python virtual environment...
    python -m venv venv
)

echo [INFO] Activating environment...
call venv\Scripts\activate.bat

echo [INFO] Installing dependencies...
echo (This may take a few minutes)
python -m pip install --upgrade pip
pip install -r requirements.txt

# Ensure torch is installed with CUDA if available, but for simplicity relying on requirements.txt
# (User's requirements.txt has generated specific versions, hopefully correct)

echo.
echo ✅ Setup Complete!
echo You can now run 'run_app.bat' to start SilverGuard.
pause
