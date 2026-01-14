# SilverGuard/dashboard.py
import streamlit as st
import os
import json
import time
from PIL import Image
import utils

st.set_page_config(page_title="SilverGuard Dashboard", layout="wide")
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

def save_settings(token, chat_id, contact, privacy_mode):
    data = {
        "TELEGRAM_TOKEN": token,
        "TELEGRAM_CHAT_ID": chat_id,
        "EMERGENCY_CONTACT": contact,
        "PRIVACY_MODE": privacy_mode 
    }
    with open(utils.SETTINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

current_settings = load_settings()
default_contact = current_settings.get("EMERGENCY_CONTACT", "010-0000-0000")
default_token = current_settings.get("TELEGRAM_TOKEN", "")
default_chat_id = current_settings.get("TELEGRAM_CHAT_ID", "")
default_privacy = current_settings.get("PRIVACY_MODE", False) 

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
        # 여기가 핵심: 버추얼 모드 스위치
        privacy_mode = st.toggle("🛡️ 버추얼 모드 (사생활 보호)", value=default_privacy)
        if privacy_mode:
            st.caption("카메라 화면 대신 AI가 인식한 '스켈레톤(뼈대)'만 화면에 표시합니다.")
        
        st.divider()
        contact = st.text_input("보호자 긴급 연락처", value=default_contact)
        telegram_token = st.text_input("텔레그램 봇 토큰", value=default_token, type="password")
        chat_id = st.text_input("텔레그램 챗 ID", value=default_chat_id)
        
        if st.button("설정 저장"):
            save_settings(telegram_token, chat_id, contact, privacy_mode)
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
                        if st.button(f"삭제", key=f"del_{idx}"):
                            os.remove(img_path)
                            st.rerun()
                except: pass