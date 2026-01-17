@echo off
chcp 65001 > nul
title SilverGuard 설치 마법사
color 0A

echo ============================================================
echo           🛡️ SilverGuard v2.0 - 설치 마법사
echo           AI 낙상 감지 시스템
echo ============================================================
echo.

:: 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️  관리자 권한이 필요합니다.
    echo     이 파일을 우클릭하여 "관리자 권한으로 실행"하세요.
    pause
    exit /b 1
)

:: Python 확인
echo [1/5] Python 확인 중...
python --version > nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Python이 설치되지 않았습니다.
    echo.
    echo 📥 Python 3.11 이상을 설치해주세요:
    echo    https://www.python.org/downloads/
    echo.
    echo    ⚠️ 설치 시 "Add Python to PATH" 반드시 체크!
    pause
    exit /b 1
)
echo ✅ Python 확인 완료

:: 가상환경 생성
echo.
echo [2/5] 가상환경 생성 중...
if not exist "venv" (
    python -m venv venv
    if %errorLevel% neq 0 (
        echo ❌ 가상환경 생성 실패
        pause
        exit /b 1
    )
)
echo ✅ 가상환경 준비 완료

:: 가상환경 활성화
call venv\Scripts\activate.bat

:: 패키지 설치
echo.
echo [3/5] 필수 패키지 설치 중... (시간이 걸릴 수 있습니다)
pip install --upgrade pip > nul 2>&1
pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo ❌ 패키지 설치 실패
    pause
    exit /b 1
)
echo ✅ 패키지 설치 완료

:: YOLO 모델 다운로드
echo.
echo [4/5] AI 모델 확인 중...
if not exist "models\yolo11n-pose.pt" (
    echo    YOLO 모델 다운로드 중...
    python download_model.py
)
if not exist "models\stgcn_fall.pth" (
    echo ⚠️  ST-GCN 모델이 없습니다. 학습이 필요합니다.
)
echo ✅ AI 모델 확인 완료

:: 데이터 폴더 생성
echo.
echo [5/5] 데이터 폴더 생성 중...
if not exist "data" mkdir data
if not exist "data\alert_images" mkdir data\alert_images
if not exist "data\verified_falls" mkdir data\verified_falls
if not exist "data\false_alarms" mkdir data\false_alarms
if not exist "data\videos" mkdir data\videos
echo ✅ 폴더 구조 생성 완료

:: 완료
echo.
echo ============================================================
echo           ✅ 설치 완료!
echo ============================================================
echo.
echo 🚀 실행 방법:
echo    1. Start_SilverGuard.bat 실행 (메인 프로그램)
echo    2. 웹 대시보드: http://localhost:8501
echo.
echo 📋 초기 설정:
echo    1. 대시보드 접속 (비밀번호: silver1234)
echo    2. 텔레그램 봇 토큰/챗ID 입력
echo    3. 보호자 연락처 입력
echo.
echo 📖 자세한 사용법: Readme.md 참고
echo ============================================================
pause
