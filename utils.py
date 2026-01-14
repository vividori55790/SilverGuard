# SilverGuard/utils.py
import os
import requests
import json
import datetime # 시간 확인을 위해 추가

# ==================================================
# [1] 경로 설정
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
ALERT_DIR = os.path.join(DATA_DIR, 'alert_images')
SETTINGS_PATH = os.path.join(DATA_DIR, 'settings.json')
# 시스템 상태(심장박동)를 저장할 파일
STATUS_PATH = os.path.join(DATA_DIR, 'status.json') 

YOLO_MODEL_PATH = os.path.join(MODEL_DIR, 'yolov8n-pose.pt')

# ==================================================
# [2] 시스템 설정값
# ==================================================
TEST_VIDEO_NAME = 'fall_test.mp4' 
CROP_RIGHT_HALF = False  
FALL_TIME_THRESHOLD = 5.0 
MOTION_THRESHOLD = 3000 

# ==================================================
# [3] 유틸리티 함수
# ==================================================
def ensure_dirs():
    """필요한 폴더가 없으면 생성"""
    os.makedirs(ALERT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def get_telegram_settings():
    """저장된 설정 파일에서 텔레그램 토큰을 가져옵니다."""
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('TELEGRAM_TOKEN'), data.get('TELEGRAM_CHAT_ID')
    except Exception:
        pass
    return None, None

def send_telegram_alert(image_path, message):
    """설정 파일에서 토큰을 읽어와 전송합니다."""
    token, chat_id = get_telegram_settings()
    if not token or not chat_id:
        print("❌ 텔레그램 설정이 없습니다. 대시보드에서 설정해주세요.")
        return False

    try:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        with open(image_path, 'rb') as img_file:
            files = {'photo': img_file}
            data = {'chat_id': chat_id, 'caption': message}
            response = requests.post(url, files=files, data=data, timeout=10)
            
        if response.status_code == 200:
            print("🔔 텔레그램 알림 전송 성공!")
            return True
        else:
            print(f"❌ 전송 실패: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 연결 오류: {e}")
        return False

# [추가됨] 시스템 상태 관리 함수들
def update_heartbeat():
    """main.py가 실행 중임을 알리는 심장박동 시간을 기록합니다."""
    try:
        data = {"last_active": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        with open(STATUS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception:
        pass

def is_system_running():
    """최근 10초 이내에 심장박동이 있었는지 확인합니다."""
    if not os.path.exists(STATUS_PATH):
        return False
    try:
        with open(STATUS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            last_active_str = data.get("last_active")
            if last_active_str:
                last_active = datetime.datetime.strptime(last_active_str, "%Y-%m-%d %H:%M:%S")
                # 현재 시간과 기록된 시간의 차이가 10초 이내면 실행 중으로 판단
                if (datetime.datetime.now() - last_active).total_seconds() < 10:
                    return True
    except Exception:
        pass
    return False