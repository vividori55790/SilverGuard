# SilverGuard/utils.py
import os
import requests
import json
import datetime # 시간 확인을 위해 추가
import pandas as pd

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

YOLO_MODEL_PATH = os.path.join(MODEL_DIR, 'yolo11n-pose.pt')
CSV_PATH = os.path.join(DATA_DIR, 'dataset.csv')

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

def send_telegram_alert(image_path, message, gif_path=None):
    """설정 파일에서 토큰을 읽어와 전송합니다. (gif_path가 있으면 영상도 전송)"""
    token, chat_id = get_telegram_settings()
    if not token or not chat_id:
        print("❌ 텔레그램 설정이 없습니다. 대시보드에서 설정해주세요.")
        return False

    try:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        with open(image_path, 'rb') as img_file:
            # 병원 정보 찾기
            try:
                with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    region1 = settings.get("USER_REGION_1", "")
                    region2 = settings.get("USER_REGION_2", "")
                    
                if region1 and region2:
                    hospitals = find_nearby_hospitals(region1, region2)
                    if hospitals:
                        message += "\n\n🏥 [인근 종합병원 정보]"
                        for h in hospitals:
                            message += f"\n- {h['name']} ({h['phone']})\n  {h['address']}"
            except:
                pass

            files = {'photo': img_file}
            data = {'chat_id': chat_id, 'caption': message}
            response = requests.post(url, files=files, data=data, timeout=10)
        
        # 2. 영상 전송 (있을 경우)
        if gif_path and os.path.exists(gif_path):
            url_video = f"https://api.telegram.org/bot{token}/sendVideo"
            with open(gif_path, 'rb') as video_file:
                files_video = {'video': video_file}
                data_video = {'chat_id': chat_id, 'caption': "🎥 사고 당시 상황 기록 (3초)"}
                res_video = requests.post(url_video, files=files_video, data=data_video, timeout=60)
                
                if res_video.status_code == 200:
                    print(f"🎬 동영상 전송 성공! ({gif_path})")
                else:
                    print(f"❌ 동영상 전송 실패: {res_video.text}")
                
        if response.status_code == 200:
            print("🔔 텔레그램 알림 전송 성공!")
            return True
        else:
            print(f"❌ 전송 실패: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 연결 오류: {e}")
        return False

def format_phone_number(number):
    """
    숫자나 문자열로 된 전화번호를 '0XX-XXX-XXXX' 형식으로 변환합니다.
    기존 데이터(float/int)에서 누락된 지역번호 앞자리 '0'을 복구합니다.
    """
    try:
        # 1. 문자열 변환 및 소수점 제거
        s = str(number).replace('.0', '').replace('-', '')
        if not s or s == 'nan': return "정보 없음"
        
        # 2. 지역번호 0 복구 (서울 2..., 경기 3... 등)
        # 1로 시작하는 대표번호(1588 등)나 010 핸드폰 번호가 아닌 경우,
        # 2~6으로 시작하면 앞에 0을 붙여줍니다.
        if len(s) >= 8 and s[0] in ['2', '3', '4', '5', '6']:
            s = '0' + s
            
        # 3. 하이픈 포맷팅
        if len(s) == 9: # 02-333-4444
            return f"{s[:2]}-{s[2:5]}-{s[5:]}"
        elif len(s) == 10: 
            if s.startswith('02'): # 02-3333-4444
                return f"{s[:2]}-{s[2:6]}-{s[6:]}"
            else: # 031-333-4444
                return f"{s[:3]}-{s[3:6]}-{s[6:]}"
        elif len(s) == 11: # 031-3333-4444
            return f"{s[:3]}-{s[3:7]}-{s[7:]}"
        elif len(s) == 8: # 1588-1234
            return f"{s[:4]}-{s[4:]}"
            
        return s
    except:
        return str(number)

def find_nearby_hospitals(region1, region2):
    """
    CSV에서 사용자의 지역(예: 서울특별시, 중구)에 있는 영업 중인 종합병원을 찾습니다.
    """
    csv_path = os.path.join(DATA_DIR, '건강_병원.csv')
    if not os.path.exists(csv_path):
        return []

    try:
        # 데이터 로드 (인코딩 주의)
        df = pd.read_csv(csv_path, encoding='cp949')
        
        # 1. 영업 중인 병원만 (영업상태명 == '영업/정상')
        df = df[df['영업상태명'] == '영업/정상']
        
        # 2. 종합병원만 (업태구분명 == '종합병원')
        df = df[df['업태구분명'] == '종합병원']
        
        # 3. 주소 필터링 (도로명주소 또는 지번주소에 지역명이 포함되어야 함)
        # NaN 처리 후 문자열 검색
        mask_road = df['도로명주소'].fillna('').str.contains(region1) & df['도로명주소'].fillna('').str.contains(region2)
        mask_jibun = df['지번주소'].fillna('').str.contains(region1) & df['지번주소'].fillna('').str.contains(region2)
        
        results = df[mask_road | mask_jibun]
        
        # 필요한 정보만 추출 (병원명, 전화번호, 주소)
        hospitals = []
        for _, row in results.iterrows():
            addr = row['도로명주소'] if pd.notna(row['도로명주소']) else row['지번주소']
            hospitals.append({
                "name": row['사업장명'],
                "phone": format_phone_number(row['전화번호']),
                "address": addr
            })
            
        return hospitals[:3] # 최대 3개까지만 반환
    except Exception as e:
        print(f"⚠️ 병원 정보 검색 실패: {e}")
        return []

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