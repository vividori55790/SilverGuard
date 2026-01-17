@echo off
chcp 65001 > nul
title SilverGuard - AI 낙상 감지 시스템
color 0B

echo ============================================================
echo           🛡️ SilverGuard v2.0 시작
echo ============================================================

:: 가상환경 활성화
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ⚠️ 가상환경이 없습니다. INSTALL.bat을 먼저 실행하세요.
    pause
    exit /b 1
)

:: 엔진 + 대시보드 동시 실행
echo.
echo 🚀 시스템 시작 중...
echo.
echo [엔진] 카메라 모니터링 시작...
echo [대시보드] http://localhost:8501
echo.
echo ⚠️ 종료하려면 이 창에서 Ctrl+C를 누르세요.
echo ============================================================

:: 백그라운드로 대시보드 실행
start "SilverGuard Dashboard" /min cmd /c "call venv\Scripts\activate.bat && streamlit run dashboard.py --server.headless true"

:: 3초 대기 후 브라우저 열기
timeout /t 3 /nobreak > nul
start http://localhost:8501

:: 엔진 실행 (포그라운드)
python main.py

:: 종료 시 대시보드도 종료
taskkill /FI "WINDOWTITLE eq SilverGuard Dashboard*" /F > nul 2>&1
echo.
echo 👋 시스템이 종료되었습니다.
pause
