# SilverGuard/dashboard.py
import streamlit as st
import os
from PIL import Image
import utils

# 페이지 설정
st.set_page_config(page_title="SilverGuard Dashboard", layout="wide")

st.title("🛡️ SilverGuard: AI 낙상 감지 시스템")
st.markdown("---")

# 탭 구성
tab1, tab2 = st.tabs(["📊 실시간 모니터링", "📁 사고 기록 갤러리"])

# [탭 1] 설정 및 상태
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ 시스템 제어")
        is_running = st.toggle("감시 시스템 활성화", value=True)
        st.metric(label="현재 상태", value="가동 중 (Running)" if is_running else "중지 (Stopped)")
        
        st.subheader("🔍 민감도 설정")
        threshold = st.slider("낙상 판단 대기 시간 (초)", 1.0, 10.0, 5.0)
        st.caption(f"넘어진 상태로 {threshold}초 이상 유지 시 신고합니다.")

    with col2:
        st.subheader("📞 알림 설정")
        contact = st.text_input("보호자 긴급 연락처", "010-0000-0000")
        telegram_token = st.text_input("텔레그램 봇 토큰", type="password")
        chat_id = st.text_input("텔레그램 챗 ID")
        if st.button("설정 저장"):
            st.success("설정이 저장되었습니다.")

# [탭 2] 사고 기록 이미지 확인
with tab2:
    st.header("🚨 감지된 낙상 사고 기록")
    
    # 이미지 폴더 확인
    if not os.path.exists(utils.ALERT_DIR):
        st.warning("아직 생성된 알림 폴더가 없습니다.")
    else:
        # 최신순 정렬
        image_files = sorted(
            [f for f in os.listdir(utils.ALERT_DIR) if f.endswith('.jpg')],
            reverse=True
        )
        
        if not image_files:
            st.info("현재 감지된 사고 기록이 없습니다.")
        else:
            # 3열 그리드로 이미지 표시
            cols = st.columns(3)
            for idx, file_name in enumerate(image_files):
                img_path = os.path.join(utils.ALERT_DIR, file_name)
                image = Image.open(img_path)
                
                with cols[idx % 3]:
                    st.image(image, caption=f"시간: {file_name[5:-4]}", use_container_width=True)
                    if st.button(f"삭제", key=f"del_{idx}"):
                        os.remove(img_path)
                        st.experimental_rerun()