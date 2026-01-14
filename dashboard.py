# SilverGuard/dashboard.py
import streamlit as st
import os
import json
import time # 자동 새로고침을 위해
from PIL import Image
import utils

st.set_page_config(page_title="SilverGuard Dashboard", layout="wide")
st.title("🛡️ SilverGuard: AI 낙상 감지 시스템")
st.markdown("---")

# 설정 파일 로드/저장 함수
def load_settings():
    if os.path.exists(utils.SETTINGS_PATH):
        with open(utils.SETTINGS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_settings(token, chat_id, contact):
    data = {
        "TELEGRAM_TOKEN": token,
        "TELEGRAM_CHAT_ID": chat_id,
        "EMERGENCY_CONTACT": contact
    }
    with open(utils.SETTINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

current_settings = load_settings()
default_contact = current_settings.get("EMERGENCY_CONTACT", "010-0000-0000")
default_token = current_settings.get("TELEGRAM_TOKEN", "")
default_chat_id = current_settings.get("TELEGRAM_CHAT_ID", "")

tab1, tab2 = st.tabs(["📊 실시간 모니터링", "📁 사고 기록 갤러리"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ 시스템 상태")
        
        # [수정됨] 실제 시스템 상태를 확인합니다.
        if utils.is_system_running():
            st.success("✅ 시스템 정상 가동 중 (Running)")
            st.caption("AI가 영상을 실시간으로 분석하고 있습니다.")
        else:
            st.error("🛑 시스템 중지됨 (Stopped)")
            st.info("터미널에서 'python main.py'를 실행해주세요.")
        
        if st.button("상태 새로고침"):
            st.rerun()
            
        st.subheader("🔍 민감도 설정")
        threshold = st.slider("낙상 판단 대기 시간 (초)", 1.0, 10.0, 5.0)

    with col2:
        st.subheader("📞 알림 설정")
        contact = st.text_input("보호자 긴급 연락처", value=default_contact)
        telegram_token = st.text_input("텔레그램 봇 토큰", value=default_token, type="password")
        chat_id = st.text_input("텔레그램 챗 ID", value=default_chat_id)
        
        if st.button("설정 저장"):
            save_settings(telegram_token, chat_id, contact)
            st.success("✅ 설정이 저장되었습니다!")

with tab2:
    st.header("🚨 감지된 낙상 사고 기록")
    if st.button("갤러리 새로고침"):
        st.rerun()

    if not os.path.exists(utils.ALERT_DIR):
        st.warning("아직 생성된 알림 폴더가 없습니다.")
    else:
        image_files = sorted(
            [f for f in os.listdir(utils.ALERT_DIR) if f.endswith('.jpg')],
            reverse=True
        )
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
                except:
                    pass