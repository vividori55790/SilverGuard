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

current_settings = load_settings()
default_contact = current_settings.get("EMERGENCY_CONTACT", "010-0000-0000")
default_token = current_settings.get("TELEGRAM_TOKEN", "")
default_chat_id = current_settings.get("TELEGRAM_CHAT_ID", "")
default_privacy = current_settings.get("PRIVACY_MODE", False)
default_region1 = current_settings.get("USER_REGION_1", "서울특별시")
default_region2 = current_settings.get("USER_REGION_2", "중구")
default_extra_cam = current_settings.get("EXTRA_CAM", "")

tab1, tab2 = st.tabs(["📊 실시간 모니터링", "📁 사고 기록 갤러리"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚙️ 시스템 상태")
        if utils.is_system_running():
            st.success("✅ 시스템 정상 가동 중 (Running)")
            if default_privacy:
                st.info("🔒 버추얼(사생활 보호) 모드 작동 중")
            else:
                st.warning("📷 실시간 카메라 모드 작동 중")
        else:
            st.error("🛑 시스템 중지됨 (Stopped)")
            st.info("터미널에서 'python main.py'를 실행해주세요.")
        
        if st.button("상태 새로고침"):
            st.rerun()
            
        st.subheader("🔍 감지 민감도")
        st.slider("낙상 판단 대기 시간 (초)", 1.0, 10.0, 5.0, disabled=True)

    with col2:
        st.subheader("🛠️ 통합 설정")
        with st.form("settings_form"):
            privacy_mode = st.toggle("🛡️ 버추얼 모드 (사생활 보호)", value=default_privacy)
            if privacy_mode:
                st.caption("카메라 화면 대신 AI가 인식한 '스켈레톤(뼈대)'만 화면에 표시합니다.")
            
            st.divider()
            extra_cam = st.text_input("추가 카메라 (번호 또는 RTSP 주소)", value=default_extra_cam, placeholder="예: 1 또는 rtsp://admin:1234@192.168.0.10/stream")
            st.caption("비워두면 기본 카메라(0번)만 사용합니다. 숫자는 USB캠 번호, 주소는 IP카메라입니다.")

            st.divider()
            contact = st.text_input("보호자 긴급 연락처", value=default_contact)
            
            st.write("📍 위치 설정 (지역)")
            col_loc1, col_loc2 = st.columns(2)
            with col_loc1:
                region1 = st.text_input("시/도", value=default_region1, placeholder="예: 서울특별시")
            with col_loc2:
                region2 = st.text_input("시/군/구", value=default_region2, placeholder="예: 강남구")

            telegram_token = st.text_input("텔레그램 봇 토큰", value=default_token, type="password")
            chat_id = st.text_input("텔레그램 챗 ID", value=default_chat_id)
            
            if st.form_submit_button("설정 저장"):
                save_settings(telegram_token, chat_id, contact, privacy_mode, region1, region2, extra_cam)
                st.success("✅ 설정이 저장되었습니다! (main.py에 즉시 적용됩니다)")

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
                        # 파일명을 키로 사용하여 삭제 버튼 고유성 보장
                        if st.button(f"삭제", key=f"del_{file_name}"):
                            os.remove(img_path)
                            # 관련된 영상 파일도 있으면 삭제
                            video_path = img_path.replace(".jpg", ".mp4").replace("FALL_", "FALL_VIDEO_")
                            if os.path.exists(video_path):
                                os.remove(video_path)
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