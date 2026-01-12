# SilverGuard/utils.py
import os
import requests  # pip install requests 필요

# ==================================================
# [수정본] 폴더가 SilverGuard 폴더 안에 있을 때 전용 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # SilverGuard 폴더 위치

# 데이터 폴더 설정 (SilverGuard 안의 data 폴더를 찾습니다)
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 모델 폴더 설정 (SilverGuard 안의 models 폴더를 찾습니다)
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 세부 경로 설정
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
ALERT_DIR = os.path.join(DATA_DIR, 'alert_images')
CSV_PATH = os.path.join(DATA_DIR, 'dataset.csv')

YOLO_MODEL_PATH = os.path.join(MODEL_DIR, 'yolov8n-pose.pt')
ML_MODEL_PATH = os.path.join(MODEL_DIR, 'fall_classifier.pkl')
# ==================================================
# [2] 시스템 설정값
# ==================================================
TEST_VIDEO_NAME = 'fall_test.mp4' 

# 로컬 웹캠 사용 시에는 False로 설정 (전체 화면 사용)
CROP_RIGHT_HALF = False  

FALL_TIME_THRESHOLD = 5.0 

# 움직임 감지 임계값 (픽셀 수)
# 값이 클수록 둔감해지고(작은 움직임 무시), 작을수록 민감해집니다.
MOTION_THRESHOLD = 3000 

# 텔레그램 알림 설정
# BotFather에게 받은 토큰과 Chat ID를 입력하세요.
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# 멀티 카메라 소스 (테스트용)
CAMERA_SOURCES = [0] # 기본 웹캠

# ==================================================
# [3] 유틸리티 함수
# ==================================================
def ensure_dirs():
    """필요한 폴더가 없으면 생성"""
    os.makedirs(ALERT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def send_telegram_alert(image_path, message):
    """
    낙상 감지 시 텔레그램으로 이미지와 메시지를 전송합니다.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        # 설정이 안 되어 있으면 조용히 리턴 (에러 방지)
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(image_path, 'rb') as img_file:
            files = {'photo': img_file}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message}
            response = requests.post(url, files=files, data=data)
            
        if response.status_code == 200:
            print("🔔 텔레그램 알림 전송 성공!")
        else:
            print(f"❌ 텔레그램 전송 실패: {response.text}")
    except Exception as e:
        print(f"❌ 텔레그램 연결 오류: {e}")