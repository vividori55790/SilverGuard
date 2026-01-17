# SilverGuard/dashboard.py
import streamlit as st
import os
import json
import time
import sys
import numpy as np
import cv2
from PIL import Image
import utils

# 현재 디렉토리를 경로에 추가하여 voice_module import 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import voice_module
except ImportError:
    voice_module = None

st.set_page_config(page_title="SilverGuard Dashboard", layout="wide")

# [CSS] 입력창의 'Press Enter to submit' 문구 숨기기
st.markdown("""
    <style>
    [data-testid="InputInstructions"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
st.title("🛡️ SilverGuard: AI 낙상 감지 시스템")
st.markdown("---")

def load_settings():
    if os.path.exists(utils.SETTINGS_PATH):
        try:
            with open(utils.SETTINGS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_settings(token, chat_id, contact, privacy_mode, region1, region2, extra_cam):
    data = {
        "TELEGRAM_TOKEN": token,
        "TELEGRAM_CHAT_ID": chat_id,
        "EMERGENCY_CONTACT": contact,
        "PRIVACY_MODE": privacy_mode,
        "USER_REGION_1": region1,
        "USER_REGION_2": region2,
        "EXTRA_CAM": extra_cam
    }
    with open(utils.SETTINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Load settings
settings = load_settings()
default_token = settings.get("TELEGRAM_TOKEN", "")
default_chat_id = settings.get("TELEGRAM_CHAT_ID", "")
default_contact = settings.get("EMERGENCY_CONTACT", "010-0000-0000") # Reverted to original default for contact
default_privacy = settings.get("PRIVACY_MODE", False)
default_extra_cam = settings.get("EXTRA_CAM", "")
default_region1 = settings.get("USER_REGION_1", "서울특별시") # Reverted to original key
default_region2 = settings.get("USER_REGION_2", "중구") # Reverted to original key
default_conf = settings.get("AI_CONFIDENCE", 0.65)
default_strictness = settings.get("AI_STRICTNESS", "Medium")
# [NEW] Default ID/PW
default_cam_user = settings.get("CAM_USER", "")
default_cam_pass = settings.get("CAM_PASS", "")


# [SECURITY] Authentication State
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def check_login(password, current_password):
    if password == current_password:
        st.session_state['authenticated'] = True
        st.rerun() # Refresh to show content
    else:
        st.error("🚫비밀번호가 올바르지 않습니다.")

# Load dashboard password
dashboard_pw = settings.get("DASHBOARD_PW", "silver1234") # Default Password

# Show Login Page if not authenticated
if not st.session_state['authenticated']:
    st.markdown("### 🔒 SilverGuard 보안 진입")
    input_pw = st.text_input("대시보드 접속 비밀번호를 입력하세요", type="password")
    if st.button("로그인"):
        check_login(input_pw, dashboard_pw)
    
    st.info(f"초기 비밀번호는 'silver1234' 입니다.")
    st.stop() # Stop execution here until logged in

# Main Dashboard Content (Only reachable if authenticated)
tab1, tab2 = st.tabs(["📊 실시간 모니터링", "📁 사고 기록 갤러리"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚙️ 시스템 상태")
        if utils.is_system_running():
            st.success("✅ 시스템 정상 가동 중 (Running)")
            
            # Show Camera Status Details
            try:
                if os.path.exists(utils.STATUS_PATH):
                    with open(utils.STATUS_PATH, 'r', encoding='utf-8') as f:
                        status_data = json.load(f)
                        cam_count = status_data.get("active_cameras", 1)
                        cam_ids = status_data.get("camera_ids", [0])
                        st.write(f"📹 연결된 카메라: {cam_count}대 (ID: {cam_ids})")
            except: pass

            if default_privacy:
                st.info("🔒 버추얼(사생활 보호) 모드 작동 중")
            else:
                st.warning("📷 실시간 카메라 모드 작동 중")
        else:
            st.error("🛑 시스템 중지됨 (Stopped)")
            st.info("터미널에서 'python main.py'를 실행해주세요.")
        
        if st.button("상태 새로고침"):
            st.rerun()
            
        st.subheader("🚨 통합 테스트")
        if st.button("🚀 낙상 시뮬레이션 (즉시 발동)", help="실제 상황과 동일하게 알림, 통화, 영상 저장이 수행됩니다.", type="primary"):
            # Create a signal file that Engine will pick up
            signal_path = os.path.join(utils.DATA_DIR, 'TRIGGER_FALL_SIM.signal')
            with open(signal_path, 'w') as f:
                f.write("TRIGGER")
            st.success("✅ 시뮬레이션 신호를 보냈습니다! 엔진 콘솔을 확인하세요.")

    with col2:
        st.subheader("🛠️ 통합 설정")
        with st.form("settings_form"):
            # 토글 값 설정
            privacy_mode = st.toggle("🛡️ 버추얼 모드 (사생활 보호)", value=default_privacy)
            if privacy_mode:
                st.caption("카메라 화면 대신 AI가 인식한 '스켈레톤(뼈대)'만 화면에 표시합니다.")
            
            st.divider()
            
            st.subheader("📷 카메라 연결 설정")
            extra_cam = st.text_input("카메라 주소 (IP 또는 RTSP)", value=default_extra_cam, placeholder="예: 192.168.0.10, rtsp://...")
            extra_cam_enabled = st.toggle("원격 카메라 사용", value=(len(default_extra_cam) > 0)) # [NEW] Toggle
            st.caption("체크 해제 시 원격 카메라는 연결하지 않습니다.")

            # [NEW] Credential Inputs
            col_cred1, col_cred2 = st.columns(2)
            with col_cred1:
                cam_user = st.text_input("카메라 ID (선택)", value=default_cam_user, placeholder="admin")
            with col_cred2:
                cam_pass = st.text_input("카메라 PW (선택)", value=default_cam_pass, type="password", placeholder="1234")
            
            st.divider()
            contact = st.text_input("보호자 긴급 연락처 (쉼표로 구분)", value=default_contact, placeholder="예: 010-1234-5678")
            
            # [Auto Call Moved Here]
            default_autocall = settings.get("AUTO_CALL_ENABLED", False)
            auto_call = st.toggle("📞 자동 전화 걸기 (PC 연결 앱)", value=default_autocall, help="낙상 감지 시 PC에 연결된 전화 앱으로 즉시 전화를 겁니다.")
            
            st.write("📍 위치 설정 (지역)")
            col_loc1, col_loc2 = st.columns(2)
            with col_loc1:
                region1 = st.text_input("시/도", value=default_region1, placeholder="예: 서울특별시")
            with col_loc2:
                region2 = st.text_input("시/군/구", value=default_region2, placeholder="예: 강남구")

            # Sensitivity Controls
            st.write("🎛️ AI 민감도 설정")
            
            # 1. Confidence Threshold
            ai_conf = st.slider("낙상 확신도 기준 (높을수록 신중함)", 
                                min_value=0.4, max_value=0.9, 
                                value=float(default_conf), step=0.05, 
                                help="AI가 '낙상이다!'라고 확신하는 정도입니다. 너무 낮으면 오작동이 늘고, 너무 높으면 낙상을 놓칠 수 있습니다.")
            
            # 2. Strictness (Heuristics)
            strictness_options = ["Low (민감함)", "Medium (권장)", "High (엄격함)"]
            
            # Map string to index
            strict_idx = 1 # Medium default
            if "Low" in default_strictness: strict_idx = 0
            if "High" in default_strictness: strict_idx = 2
            
            strictness_ui = st.select_slider("오작동 방지 강도 (Strictness)", 
                                            options=strictness_options, 
                                            value=strictness_options[strict_idx],
                                            help="앉기/눕기 등을 낙상으로 오인하지 않도록 하는 안전장치 강도입니다.")
            
            # Parse back to simple string
            ai_strictness = "Medium"
            if "Low" in strictness_ui: ai_strictness = "Low"
            elif "High" in strictness_ui: ai_strictness = "High"

            st.divider()
            telegram_token = st.text_input("텔레그램 봇 토큰", value=default_token, type="password")
            chat_id = st.text_input("텔레그램 챗 ID", value=default_chat_id)
            st.divider()
            st.write("🔐 보안 설정")
            new_dash_pw = st.text_input("대시보드 접속 비밀번호 변경", value=dashboard_pw, type="password")

            if st.form_submit_button("설정 저장"):
                # Save all
                if not extra_cam_enabled:
                     extra_cam = "" # Clear if disabled

                new_settings = {
                    "TELEGRAM_TOKEN": telegram_token,
                    "TELEGRAM_CHAT_ID": chat_id,
                    "EMERGENCY_CONTACT": contact,
                    "PRIVACY_MODE": privacy_mode,
                    "EXTRA_CAM": extra_cam,
                    "CAM_USER": cam_user,    
                    "CAM_PASS": cam_pass,    
                    "USER_REGION_1": region1, 
                    "USER_REGION_2": region2, 
                    "AI_CONFIDENCE": ai_conf,
                    "AI_STRICTNESS": ai_strictness,
                    "AUTO_CALL_ENABLED": auto_call, 
                    "DASHBOARD_PW": new_dash_pw
                }
                with open(utils.SETTINGS_PATH, 'w', encoding='utf-8') as f:
                    json.dump(new_settings, f, ensure_ascii=False, indent=4)
                    
                st.success("✅ 설정이 저장되었습니다! 엔진에 즉시 반영됩니다.")
                time.sleep(1.0)
                st.rerun()
with tab2:
    st.header("🚨 감지된 낙상 사고 기록")
    if st.button("갤러리 새로고침"):
        st.rerun()

    if not os.path.exists(utils.ALERT_DIR):
        st.warning("아직 생성된 알림 폴더가 없습니다.")
    else:
        image_files = sorted([f for f in os.listdir(utils.ALERT_DIR) if f.endswith('.jpg')], reverse=True)
        if not image_files:
            st.info("현재 감지된 사고 기록이 없습니다.")
        else:
            cols = st.columns(3)
            for idx, file_name in enumerate(image_files):
                img_path = os.path.join(utils.ALERT_DIR, file_name)
                try:
                    image = Image.open(img_path)
                    with cols[idx % 3]:
                        st.image(image, caption=f"시간: {file_name[5:-4]}", use_container_width=True)
                        
                        # [Feedback System]
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("⭕ 실제 낙상", key=f"true_{file_name}", help="학습 데이터로 사용하여 감지력을 높입니다."):
                                # Move to VERIFIED
                                utils.ensure_dirs()
                                target_dir = utils.VERIFIED_DIR
                                base_name = os.path.splitext(file_name)[0]
                                
                                # Move JPG
                                os.rename(img_path, os.path.join(target_dir, file_name))
                                # Move NPY
                                npy_name = base_name + ".npy"
                                if os.path.exists(os.path.join(utils.ALERT_DIR, npy_name)):
                                    os.rename(os.path.join(utils.ALERT_DIR, npy_name), os.path.join(target_dir, npy_name))
                                # Move Video
                                vid_name = base_name.replace("FALL_", "FALL_VIDEO_") + ".mp4"
                                if os.path.exists(os.path.join(utils.ALERT_DIR, vid_name)):
                                    os.rename(os.path.join(utils.ALERT_DIR, vid_name), os.path.join(target_dir, vid_name))
                                
                                st.success("✅ 학습 데이터로 분류됨")
                                time.sleep(0.5)
                                st.rerun()
                                
                        with c2:
                            if st.button("❌ 오작동", key=f"false_{file_name}", help="오작동 데이터로 사용하여 실수를 줄입니다."):
                                # Move to FALSE_ALARM
                                utils.ensure_dirs()
                                target_dir = utils.FALSE_ALARM_DIR
                                base_name = os.path.splitext(file_name)[0]
                                
                                # Move JPG
                                os.rename(img_path, os.path.join(target_dir, file_name))
                                # Move NPY
                                npy_name = base_name + ".npy"
                                if os.path.exists(os.path.join(utils.ALERT_DIR, npy_name)):
                                    os.rename(os.path.join(utils.ALERT_DIR, npy_name), os.path.join(target_dir, npy_name))
                                # Move Video
                                vid_name = base_name.replace("FALL_", "FALL_VIDEO_") + ".mp4"
                                if os.path.exists(os.path.join(utils.ALERT_DIR, vid_name)):
                                    os.rename(os.path.join(utils.ALERT_DIR, vid_name), os.path.join(target_dir, vid_name))
                                
                                st.warning("❎ 오작동 사례로 등록됨")
                                time.sleep(0.5)
                                st.rerun()

                        # Just Delete
                        if st.button(f"🗑️ 영구 삭제 (분류 안함)", key=f"del_{file_name}"):
                            os.remove(img_path)
                            video_path = img_path.replace(".jpg", ".mp4").replace("FALL_", "FALL_VIDEO_")
                            if os.path.exists(video_path):
                                os.remove(video_path)
                            npy_path = img_path.replace(".jpg", ".npy")
                            if os.path.exists(npy_path):
                                os.remove(npy_path)
                            st.rerun()
                except: pass

# ==========================================
# [사이드바] 디버깅 도구
# ==========================================
st.sidebar.title("🔧 디버깅 도구")
if st.sidebar.button("🚨 낙상 시뮬레이션 (TEST)"):
    st.sidebar.warning("⚠️ 낙상 감지 시나리오를 시작합니다...")
    
    # 1. 테스트 이미지 생성 (검은 화면에 텍스트)
    if not os.path.exists(utils.ALERT_DIR):
        os.makedirs(utils.ALERT_DIR)
        
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(utils.ALERT_DIR, f"TEST_FALL_{timestamp}.jpg")
    
    # 더미 이미지 생성 (Create blank image)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Write text on image
    cv2.putText(dummy_img, "TEST FALL DETECTION", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(save_path, dummy_img)
    
    st.sidebar.write("📸 테스트 이미지 생성 완료")
    
    # 2. 텔레그램 전송
    st.sidebar.write("📤 텔레그램 알림 전송 중...")
    success = utils.send_telegram_alert(save_path, "🚨 [TEST] 낙상 시뮬레이션 발생!", None)
    if success:
        st.sidebar.success("✅ 텔레그램 전송 성공")
    else:
        st.sidebar.error("❌ 텔레그램 전송 실패")
        
    # 3. 음성 모듈 테스트
    if voice_module:
        st.sidebar.write("🎙️ 음성 확인 모듈 실행 중... (약 10~20초 소요)")
        # Streamlit이 멈추는 것을 방지하기 위해 간단히 안내만 표시하고 실행
        result = voice_module.run_voice_emergency_check(save_path)
        st.sidebar.info(f"🗣️ 음성 모듈 결과: {result}")
    else:
        st.sidebar.error("❌ voice_module을 불러올 수 없습니다.")
        
    st.sidebar.success("✅ 시뮬레이션 종료")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("✉️ 수동 메세지 전송")
with st.sidebar.form("manual_msg_form"):
    custom_msg = st.text_area("보낼 내용", placeholder="예: 시스템 점검 중입니다.")
    if st.form_submit_button("전송하기"):
        if custom_msg.strip():
            # Dummy image for the function requirement (function expects image path)
            # We can modify utils to accept None, OR just send a token image, OR use requests directly here.
            # Easiest: Use requests directly here or create a dummy helper in utils.
            # Let's import requests in dashboard if needed, or use utils.
            # utils.send_telegram_alert requires image path.
            # Let's quickly make a temp image or just use requests here for simplicity.
            
            token = settings.get("TELEGRAM_TOKEN")
            chat_id = settings.get("TELEGRAM_CHAT_ID")
            
            if token and chat_id:
                try:
                    import requests
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    data = {'chat_id': chat_id, 'text': f"💬 [관리자 메시지]\n{custom_msg}"}
                    response = requests.post(url, data=data, timeout=5)
                    if response.status_code == 200:
                        st.success("전송 성공!")
                    else:
                        st.error(f"전송 실패: {response.text}")
                except Exception as e:
                    st.error(f"에러: {e}")
            else:
                 st.error("텔레그램 설정이 비어있습니다.")
        else:
            st.warning("내용을 입력해주세요.")