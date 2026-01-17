import os
import requests
import json
import datetime # 시간 확인을 위해 추가
import pandas as pd
import ctypes
import time
import shutil # [Fix] Added missing import

# ==================================================
# [1] 경로 설정
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
ALERT_DIR = os.path.join(DATA_DIR, 'alert_images')
VERIFIED_DIR = os.path.join(DATA_DIR, 'verified_falls')      # [NEW] 실제 낙상 데이터
FALSE_ALARM_DIR = os.path.join(DATA_DIR, 'false_alarms')    # [NEW] 오작동 데이터
ARCHIVE_DIR = os.path.join(DATA_DIR, 'archive')             # [NEW] 학습 완료된 데이터 보관
ARCHIVE_VERIFIED = os.path.join(ARCHIVE_DIR, 'verified_falls')
ARCHIVE_FALSE = os.path.join(ARCHIVE_DIR, 'false_alarms')

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
    os.makedirs(VERIFIED_DIR, exist_ok=True)
    os.makedirs(FALSE_ALARM_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_VERIFIED, exist_ok=True)
    os.makedirs(ARCHIVE_FALSE, exist_ok=True)
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

def send_telegram_message(text):
    """이미지 없이 텍스트 메시지만 전송합니다. (시스템 알림용)"""
    token, chat_id = get_telegram_settings()
    if not token or not chat_id: return False
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {'chat_id': chat_id, 'text': text}
        requests.post(url, data=data, timeout=5)
        return True
    except:
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
def update_heartbeat(extra_data=None):
    """main.py가 실행 중임을 알리는 심장박동 시간을 기록합니다. 추가 정보도 함께 저장합니다."""
    try:
        data = {"last_active": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        if extra_data:
            data.update(extra_data)
            
        with open(STATUS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
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




def move_alert_to_classified(jpg_path, target_dir):
    """
    낙상 알림 파일들(jpg, npy, mp4)을 지정된 폴더(Verified/FalseAlarm)로 이동시킵니다.
    """
    try:
        ensure_dirs()
        if not os.path.exists(jpg_path): return False
        
        filename = os.path.basename(jpg_path)
        base_name = os.path.splitext(filename)[0]
        
        # Helper to move with overwrite
        def move_file(src, dst_folder):
            if os.path.exists(src):
                dst = os.path.join(dst_folder, os.path.basename(src))
                if os.path.exists(dst): os.remove(dst)
                shutil.move(src, dst)
                return True
            return False

        # 1. Move JPG
        move_file(jpg_path, target_dir)
        
        # 2. Move NPY
        npy_name = base_name + ".npy"
        move_file(os.path.join(ALERT_DIR, npy_name), target_dir)
        
        # 3. Move Video
        vid_name = base_name.replace("FALL_", "FALL_VIDEO_") + ".mp4"
        move_file(os.path.join(ALERT_DIR, vid_name), target_dir)
        
        print(f"📦 데이터 자동 분류 완료: {base_name} -> {target_dir}")
        return True
    except Exception as e:
        print(f"⚠️ 데이터 이동 실패: {e}")
        return False

def find_and_maximize_window():
    """휴대폰과 연결 앱 윈도우를 찾아 최상단으로 올리고 최대화합니다."""
    targets = ["휴대폰과 연결", "Phone Link", "통화", "Call", "Android"]
    found_hwnd = []
    
    def enum_cb(hwnd, _):
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value
        if any(t in title for t in targets):
            # Check if likely the main window (visible)
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                found_hwnd.append(hwnd)
        return True

    CMPFUNC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
    ctypes.windll.user32.EnumWindows(CMPFUNC(enum_cb), 0)
    
    if found_hwnd:
        hwnd = found_hwnd[0] # First match
        # SW_MAXIMIZE = 3, SW_RESTORE = 9
        ctypes.windll.user32.ShowWindow(hwnd, 3) 
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        return True
    return False

def make_phone_call(phone_number):
    """
    Windows의 'tel:' 프로토콜로 전화 앱을 실행하고, 
    창을 최대화한 뒤 마우스로 통화 버튼을 클릭합니다.
    """
    try:
        # 전화번호 정제
        clean_number = "".join(filter(str.isdigit, str(phone_number)))
        
        # 1. 앱 실행
        # tel: 프로토콜로 앱을 호출하되, 빈 내용으로 호출하여 앱을 포커싱
        os.startfile("tel:")
        print(f"📞 PC에서 전화 앱 실행 중... (대상: {clean_number})")
        
        # 앱 로딩 대기
        time.sleep(3.0)
        
        user32 = ctypes.windll.user32
        
        # 2. 창 찾기 및 최대화
        print("🖥️ 전화 앱 윈도우 최대화 시도...")
        maximized = False
        for _ in range(5): 
            if find_and_maximize_window():
                maximized = True
                print("✅ 전화 앱 윈도우를 찾아 최대화했습니다.")
                break
            time.sleep(1.0)
            
        if not maximized:
            print("⚠️ 윈도우를 찾지 못했습니다. 키보드 입력이 다른 창으로 갈 수 있습니다.")
        
        # 확실히 포커스 잡히도록 대기
        time.sleep(1.0)

        # 3. 키패드 입력 (한 글자씩 타이핑)
        print(f"⌨️ 전화번호 키패드 입력 중: {clean_number}")
        for char in clean_number:
            if '0' <= char <= '9':
                vk = ord(char) # 0-9의 ASCII 코드는 가상 키코드와 일치 (0x30~0x39)
                user32.keybd_event(vk, 0, 0, 0)
                time.sleep(0.05)
                user32.keybd_event(vk, 0, 2, 0)
                time.sleep(0.05)
        
        time.sleep(0.5)

        # 4. 엔터 입력 (발신)
        print("🚀 Enter 키로 통화 시작")
        user32.keybd_event(0x0D, 0, 0, 0) # Enter Down
        time.sleep(0.1)
        user32.keybd_event(0x0D, 0, 2, 0) # Enter Up
        
        return True
    except Exception as e:
        print(f"❌ 전화 발신 실패: {e}")
        return False