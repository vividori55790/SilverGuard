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

# ==========================================
# [사이드바] 시스템 모니터링 (Moved to Authenticated Section)
# ==========================================
if False: # Previously 'with st.sidebar:'
    st.title("💻 시스템 리소스 모니터")
    
    if os.path.exists(utils.STATUS_PATH):
        try:
             with open(utils.STATUS_PATH, 'r', encoding='utf-8') as f:
                 status = json.load(f)
                 perf = status.get("perf", {})
                 fps = status.get("fps_real", 0)
                 
                 # Display FPS
                 st.metric("실시간 FPS", f"{fps:.1f}", help="현재 시스템이 처리하는 초당 프레임 수입니다.")
                 
                 if perf:
                     st.caption("작업별 리소스 점유율 (1프레임 당)")
                     # Calculate total ms
                     total_ms = sum(perf.values())
                     if total_ms > 0:
                         # AI Inference
                         inf_ms = perf.get("AI Inference", 0)
                         st.progress(min(1.0, inf_ms/total_ms), text=f"🤖 AI 분석 ({inf_ms}ms)")
                         
                         # Visual Overlay
                         ovr_ms = perf.get("Visual Overlay", 0)
                         st.progress(min(1.0, ovr_ms/total_ms), text=f"🎨 화면/아바타 ({ovr_ms}ms)")
                         
                         # Capture
                         cap_ms = perf.get("Capture (IO)", 0)
                         st.progress(min(1.0, cap_ms/total_ms), text=f"📷 카메라 입력 ({cap_ms}ms)")
                         
                         # Idle
                         idle_ms = perf.get("Idle (Free)", 0)
                         st.progress(min(1.0, idle_ms/total_ms), text=f"💤 유휴 자원 ({idle_ms}ms)")
                     else:
                         st.info("데이터 수집 중...")
                 else:
                     st.info("엔진 대기 중...")
        except Exception as e:
            st.error(f"모니터링 오류: {e}")
    else:
        st.warning("엔진이 실행되지 않았습니다.")

    st.divider()
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
            st.write("⚙️ 시스템 성능 최적화")
            default_fps = settings.get("TARGET_FPS", 30)
            target_fps = st.slider("목표 FPS (낮을수록 리소스 절약)", 5, 60, int(default_fps), 1, help="PC 사양이 낮다면 15~20으로 설정하세요. (권장: 30)")

            st.divider()
            telegram_token = st.text_input("텔레그램 봇 토큰", value=default_token, type="password")
            chat_id = st.text_input("텔레그램 챗 ID", value=default_chat_id)
            st.divider()
            st.write("🔐 보안 설정")
            new_dash_pw = st.text_input("대시보드 접속 비밀번호 변경", value=dashboard_pw, type="password")

            if st.form_submit_button("설정 저장"):
                # Save all (정보 손실 방지: 비활성화되어도 내용은 저장)
                new_settings = {
                    "TELEGRAM_TOKEN": telegram_token,
                    "TELEGRAM_CHAT_ID": chat_id,
                    "EMERGENCY_CONTACT": contact,
                    "PRIVACY_MODE": privacy_mode,
                    "EXTRA_CAM": extra_cam,
                    "EXTRA_CAM_ENABLED": extra_cam_enabled, # [NEW] 상태 저장
                    "CAM_USER": cam_user,    
                    "CAM_PASS": cam_pass,    
                    "USER_REGION_1": region1, 
                    "USER_REGION_2": region2, 
                    "AI_CONFIDENCE": ai_conf,
                    "AI_STRICTNESS": ai_strictness,
                    "AUTO_CALL_ENABLED": auto_call, 
                    "TARGET_FPS": target_fps, # [NEW]
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
        # [Fix] Support various image formats and Sort by Modified Time (Newest First)
        valid_exts = ('.jpg', '.jpeg', '.png')
        raw_files = [f for f in os.listdir(utils.ALERT_DIR) if f.lower().endswith(valid_exts)]
        image_files = sorted(raw_files, key=lambda x: os.path.getmtime(os.path.join(utils.ALERT_DIR, x)), reverse=True)
        
        if not image_files:
            st.info("현재 처리할 새로운 알림이 없습니다. (모두 분류됨)")
        else:
            # --- Batch Actions ---
            st.write(f"총 {len(image_files)}개의 알림 대기 중")
            
            # Select All toggle logic implies session state, but simplified:
            # just show form or checkboxes.
            
            with st.form("batch_process_form"):
                # [UX Improvement] Action Buttons at TOP
                st.write("🔽 분류 작업을 선택하세요:")
                c_top1, c_top2, c_top3 = st.columns(3)
                
                # We use variables for button states
                pressed_verify = c_top1.form_submit_button("⭕ 실제 낙상 승인")
                pressed_false = c_top2.form_submit_button("❌ 오작동 신고")
                pressed_delete = c_top3.form_submit_button("🗑️ 선택 삭제")
                
                st.divider()
                
                cols = st.columns(3)
                selected_files = []
                
                for idx, file_name in enumerate(image_files):
                    img_path = os.path.join(utils.ALERT_DIR, file_name)
                    with cols[idx % 3]:
                        # 1. Image
                        try:
                            image = Image.open(img_path)
                            st.image(image, caption=f"시간: {file_name[5:-4]}", width='stretch')
                        except: st.error("이미지 로드 실패")
                        
                        # 2. Checkbox for selection
                        if st.checkbox(f"선택하기", key=f"chk_{file_name}"):
                            selected_files.append(file_name)

                        # 3. Video
                        base_name = os.path.splitext(file_name)[0]
                        video_name = base_name.replace("FALL_", "FALL_VIDEO_") + ".mp4"
                        video_path = os.path.join(utils.ALERT_DIR, video_name)
                        
                        if os.path.exists(video_path):
                            with st.expander("🎬 영상 보기"):
                                st.video(video_path)
                
                st.divider()
                # Bottom Buttons (Optional backup)
                c_btm1, c_btm2, c_btm3 = st.columns(3)
                if c_btm1.form_submit_button("⭕ 실제 낙상 승인 (하단)"): pressed_verify = True
                if c_btm2.form_submit_button("❌ 오작동 신고 (하단)"): pressed_false = True
                if c_btm3.form_submit_button("🗑️ 선택 삭제 (하단)"): pressed_delete = True

                # --- Processing Logic ---
                if pressed_verify:
                     utils.ensure_dirs()
                     count = 0
                     for fname in selected_files:
                         try:
                             base = os.path.splitext(fname)[0]
                             src_img = os.path.join(utils.ALERT_DIR, fname)
                             dst_dir = utils.VERIFIED_DIR
                             if os.path.exists(src_img): os.rename(src_img, os.path.join(dst_dir, fname))
                             
                             npy_name = base + ".npy"
                             src_npy = os.path.join(utils.ALERT_DIR, npy_name)
                             if os.path.exists(src_npy): os.rename(src_npy, os.path.join(dst_dir, npy_name))
                             
                             vid_name = base.replace("FALL_", "FALL_VIDEO_") + ".mp4"
                             src_vid = os.path.join(utils.ALERT_DIR, vid_name)
                             if os.path.exists(src_vid): os.rename(src_vid, os.path.join(dst_dir, vid_name))
                             count += 1
                         except: pass
                     if count > 0:
                         st.success(f"{count}개 항목을 '실제 낙상'으로 분류했습니다.")
                         time.sleep(1)
                         st.rerun()
                     else:
                         st.warning("선택된 항목이 없습니다.")

                if pressed_false:
                     utils.ensure_dirs()
                     count = 0
                     for fname in selected_files:
                         try:
                             base = os.path.splitext(fname)[0]
                             src_img = os.path.join(utils.ALERT_DIR, fname)
                             dst_dir = utils.FALSE_ALARM_DIR
                             if os.path.exists(src_img): os.rename(src_img, os.path.join(dst_dir, fname))
                             
                             npy_name = base + ".npy"
                             src_npy = os.path.join(utils.ALERT_DIR, npy_name)
                             if os.path.exists(src_npy): os.rename(src_npy, os.path.join(dst_dir, npy_name))
                             
                             vid_name = base.replace("FALL_", "FALL_VIDEO_") + ".mp4"
                             src_vid = os.path.join(utils.ALERT_DIR, vid_name)
                             if os.path.exists(src_vid): os.rename(src_vid, os.path.join(dst_dir, vid_name))
                             count += 1
                         except: pass
                     if count > 0:
                         st.warning(f"{count}개 항목을 '오작동'으로 분류했습니다.")
                         time.sleep(1)
                         st.rerun()
                     else:
                         st.warning("선택된 항목이 없습니다.")
                
                if pressed_delete:
                     count = 0
                     for fname in selected_files:
                         try:
                             base = os.path.splitext(fname)[0]
                             src_img = os.path.join(utils.ALERT_DIR, fname)
                             if os.path.exists(src_img): os.remove(src_img)
                             
                             npy_name = base + ".npy"
                             src_npy = os.path.join(utils.ALERT_DIR, npy_name)
                             if os.path.exists(src_npy): os.remove(src_npy)
                             
                             vid_name = base.replace("FALL_", "FALL_VIDEO_") + ".mp4"
                             src_vid = os.path.join(utils.ALERT_DIR, vid_name)
                             if os.path.exists(src_vid): os.remove(src_vid)
                             count += 1
                         except: pass
                     if count > 0:
                         st.error(f"{count}개 항목을 삭제했습니다.")
                         time.sleep(1)
                         st.rerun()
                     else:
                         st.warning("선택된 항목이 없습니다.")

# ==========================================
# [사이드바] 관리자 도구 (학습 및 통계)
# ==========================================
import subprocess
import sys

# Log file for background training
TRAIN_LOG_FILE = os.path.join(utils.BASE_DIR, 'train_log.txt')

def is_process_running(pid):
    """Check if process is running given PID."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        # Fallback for Windows if psutil not installed
        try:
            # os.kill(pid, 0) works on Unix, on Windows it might throw PermissionError or similar
            os.kill(pid, 0)
            return True
        except OSError:
            return False

with st.sidebar:
    st.header("🛠️ 관리자 도구")
    
    st.subheader("📊 데이터 현황")
    verified_count = len([f for f in os.listdir(utils.VERIFIED_DIR) if f.endswith('.npy')]) if os.path.exists(utils.VERIFIED_DIR) else 0
    false_count = len([f for f in os.listdir(utils.FALSE_ALARM_DIR) if f.endswith('.npy')]) if os.path.exists(utils.FALSE_ALARM_DIR) else 0
    
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.metric("⭕ 실제 낙상", f"{verified_count}건")
    col_stat2.metric("❌ 오작동", f"{false_count}건")
    
    st.divider()
    
    st.subheader("🧠 AI 모델 재학습")
    st.info("이제 재학습은 데이터가 쌓이면 엔진이 '자동'으로 수행합니다.\n(하루 1회 / 데이터 30건 이상 시)")
    
    st.divider()
    
    # [Restored Resource Monitor]
    st.subheader("💻 시스템 리소스 모니터")
    if os.path.exists(utils.STATUS_PATH):
        try:
             with open(utils.STATUS_PATH, 'r', encoding='utf-8') as f:
                 status = json.load(f)
                 perf = status.get("perf", {})
                 fps = status.get("fps_real", 0)
                 
                 # Display FPS
                 st.metric("실시간 FPS", f"{fps:.1f}", help="현재 시스템이 처리하는 초당 프레임 수입니다.")
                 
                 if perf:
                     st.caption("작업별 리소스 점유율 (1프레임 당)")
                     # Calculate total ms
                     total_ms = sum(perf.values())
                     if total_ms > 0:
                         # AI Inference
                         inf_ms = perf.get("AI Inference", 0)
                         st.progress(min(1.0, inf_ms/total_ms), text=f"🤖 AI 분석 ({inf_ms}ms)")
                         
                         # Visual Overlay
                         ovr_ms = perf.get("Visual Overlay", 0)
                         st.progress(min(1.0, ovr_ms/total_ms), text=f"🎨 화면/아바타 ({ovr_ms}ms)")
                         
                         # Capture
                         cap_ms = perf.get("Capture (IO)", 0)
                         st.progress(min(1.0, cap_ms/total_ms), text=f"📷 카메라 입력 ({cap_ms}ms)")
                         
                         # Idle
                         idle_ms = perf.get("Idle (Free)", 0)
                         st.progress(min(1.0, idle_ms/total_ms), text=f"💤 유휴 자원 ({idle_ms}ms)")
                     else:
                         st.info("데이터 수집 중...")
                 else:
                     st.info("엔진 대기 중...")
        except Exception as e:
            st.error(f"모니터링 오류: {e}")
    else:
        st.warning("엔진이 실행되지 않았습니다.")

    st.divider()
    st.subheader("🔐 시스템 제어")
    st.info("설정이 변경되거나 모델이 학습되면 엔진을 재시작해야 적용됩니다.")

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