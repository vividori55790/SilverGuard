"""
SilverGuard 낙상 감지 → 알림 워크플로우 핵심 요약

이 파일은 낙상 감지부터 텔레그램 알림, 전화 발신까지의 
전체 프로세스가 어떻게 연결되어 있는지 한눈에 보여줍니다.
"""

# ============================================================
# [단계 1] 낙상 감지 (core/engine.py)
# ============================================================

"""
def _handle_detection(cam_data, pred_cls, conf, frame, ...):
    if pred_cls == 1 and conf > 0.7:  # Fall detected
        if not cam_data['fall_state']:
            cam_data['fall_state'] = True
            
            # 1. 학습 데이터 저장 (.npy)
            raw_seq = np.array(cam_data['detector'].frame_buffer)
            np.save(npy_path, raw_seq)
            
            # 2. 버퍼 스냅샷 생성 (멀티스레드 안전성)
            buffer_snapshot = list(cam_data['buffer'])
            
            # 3. 알림 프로세스 시작 (별도 스레드)
            self._process_alert_async(frame, buffer_snapshot, timestamp)
"""

# ============================================================
# [단계 2] 알림 프로세스 (core/engine.py)
# ============================================================

"""
def _trigger_alert(frame, buffer, timestamp_override=None):
    # A. 이미지 저장
    save_path = os.path.join(utils.ALERT_DIR, f"FALL_{timestamp}.jpg")
    cv2.imwrite(save_path, frame)
    
    # B. 영상 저장 (버퍼에 20프레임 이상 있을 때)
    if len(buffer) > 20:
        video_path = os.path.join(utils.ALERT_DIR, f"FALL_VIDEO_{timestamp}.mp4")
        # mp4v 또는 XVID 코덱으로 저장
        
    # C. 음성 확인 (voice_module.run_voice_emergency_check)
    stop_continuous_listening()  # 백그라운드 음성 감지 일시 중지
    time.sleep(1.0)
    
    voice_res = run_voice_emergency_check(save_path)
    # 가능한 결과: "SAFE", "EMERGENCY", "CHECK_NEEDED", 
    #              "CRITICAL_SOUND", "NO_RESPONSE_EMERGENCY"
    
    start_continuous_listening(self._on_voice_trigger)  # 재개
    
    # D. 알림 전송 (SAFE가 아닐 때만)
    if voice_res != "SAFE":
        if is_internet_available():
            if time.time() - self.last_alert_time > self.alert_cooldown:
                # 텔레그램 전송
                utils.send_telegram_alert(save_path, alert_msg, video_path)
                
                # 전화 발신 (설정 시)
                if self.auto_call_enabled and self.emergency_contact:
                    phone_nums = str(self.emergency_contact).split(',')
                    utils.make_phone_call(phone_nums[0].strip())
                
                self.last_alert_time = time.time()
        else:
            # 오프라인 모드: 대기열 저장
            activate_offline_safety_mode(save_path, offline_msg, video_path)
    else:
        print("✅ 사용자 확인 결과 '안전'하므로 알림을 전송하지 않습니다.")
"""

# ============================================================
# [단계 3] 음성 확인 (voice_module.py)
# ============================================================

"""
def run_voice_emergency_check(image_path):
    # 1차 시도
    speak("낙상이 감지되었습니다. 괜찮으십니까? 대답이 없으시면 구조 요청을 보냅니다.")
    status, detail = listen_and_analyze(timeout=10)
    
    # 2차 시도 (무응답 시)
    if status == "SILENCE":
        set_max_volume()  # 볼륨 100%
        time.sleep(1)
        speak("잘 안 들리실 수 있어 다시 크게 여쭤보겠습니다...")
        status, detail = listen_and_analyze(timeout=10)
    
    # 결과 판정
    result = "CHECK_NEEDED"
    
    if status == "VOICE":
        if any(word in detail for word in ["괜찮아", "어", "나 안 다쳤어", ...]):
            speak("확인되었습니다. 시스템을 정상 상태로 유지합니다.")
            utils.move_alert_to_classified(image_path, utils.FALSE_ALARM_DIR)
            result = "SAFE"  # ← 알림 전송 안 함!
            
        elif any(word in detail for word in ["아니", "아파", "도와줘", ...]):
            speak("위급 상황임을 확인했습니다...")
            utils.send_telegram_alert(...)  # 즉시 전송
            utils.move_alert_to_classified(image_path, utils.VERIFIED_DIR)
            result = "EMERGENCY"
            
    elif status == "SOUND_DETECTED":
        result = "CRITICAL_SOUND"
        
    else:  # SILENCE
        speak("응답이 전혀 없어 비상 상황으로 간주하고 구조 요청을 전송합니다.")
        result = "NO_RESPONSE_EMERGENCY"
    
    return result
"""

# ============================================================
# [단계 4] 텔레그램 전송 (utils.py)
# ============================================================

"""
def send_telegram_alert(image_path, message, gif_path=None):
    token, chat_id = get_telegram_settings()
    
    # 1. 병원 정보 추가
    if region1 and region2:
        hospitals = find_nearby_hospitals(region1, region2)
        if hospitals:
            message += "\\n\\n🏥 [인근 종합병원 정보]"
            for h in hospitals:
                message += f"\\n- {h['name']} ({h['phone']})"
    
    # 2. 사진 전송
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(image_path, 'rb') as img_file:
        files = {'photo': img_file}
        data = {'chat_id': chat_id, 'caption': message}
        response = requests.post(url, files=files, data=data, timeout=10)
    
    # 3. 영상 전송 (있을 경우)
    if gif_path and os.path.exists(gif_path):
        url_video = f"https://api.telegram.org/bot{token}/sendVideo"
        with open(gif_path, 'rb') as video_file:
            files_video = {'video': video_file}
            data_video = {'chat_id': chat_id, 'caption': "🎥 사고 당시 상황 기록"}
            requests.post(url_video, files=files_video, data=data_video, timeout=60)
"""

# ============================================================
# [단계 5] 전화 발신 (utils.py)
# ============================================================

"""
def make_phone_call(phone_number):
    clean_number = "".join(filter(str.isdigit, str(phone_number)))
    
    # 1. tel: 프로토콜로 앱 실행
    os.startfile("tel:")
    time.sleep(3.0)
    
    # 2. 윈도우 찾기 및 최대화
    for _ in range(5):
        if find_and_maximize_window():  # "휴대폰과 연결" 찾기
            break
        time.sleep(1.0)
    
    time.sleep(1.0)
    
    # 3. 키패드 입력
    for char in clean_number:
        if '0' <= char <= '9':
            vk = ord(char)
            user32.keybd_event(vk, 0, 0, 0)  # Press
            time.sleep(0.05)
            user32.keybd_event(vk, 0, 2, 0)  # Release
            time.sleep(0.05)
    
    time.sleep(0.5)
    
    # 4. Enter로 발신
    user32.keybd_event(0x0D, 0, 0, 0)
    time.sleep(0.1)
    user32.keybd_event(0x0D, 0, 2, 0)
"""

# ============================================================
# [핵심 파일 구조]
# ============================================================

"""
SilverGuard/
│
├── core/
│   ├── engine.py          # 메인 루프, 낙상 감지 처리, 알림 트리거
│   └── detectors.py       # ST-GCN 모델, 낙상 판정 로직
│
├── voice_module.py        # TTS, STT, 음성 확인 워크플로우
├── utils.py               # 텔레그램, 전화, 병원 검색, 파일 관리
├── dashboard.py           # Streamlit 웹 대시보드
│
├── data/
│   ├── alert_images/      # FALL_*.jpg, FALL_VIDEO_*.mp4, FALL_*.npy
│   ├── verified_falls/    # 실제 낙상 (학습 데이터)
│   └── false_alarms/      # 오작동 (학습 데이터)
│
└── models/
    └── stgcn_fall.pth     # ST-GCN 낙상 감지 모델
"""

# ============================================================
# [설정 파일 예시] data/settings.json
# ============================================================

"""
{
    "TELEGRAM_TOKEN": "당신의_봇_토큰",
    "TELEGRAM_CHAT_ID": "당신의_챗_ID",
    "EMERGENCY_CONTACT": "010-1234-5678",
    "AUTO_CALL_ENABLED": true,
    "PRIVACY_MODE": false,
    "USER_REGION_1": "서울특별시",
    "USER_REGION_2": "중구",
    "AI_CONFIDENCE": 0.65,
    "AI_STRICTNESS": "Medium",
    "EXTRA_CAM": "http://192.168.0.10:8080/video",
    "EXTRA_CAM_ENABLED": true,
    "CAM_USER": "admin",
    "CAM_PASS": "1234",
    "DASHBOARD_PW": "silver1234"
}
"""

# ============================================================
# [테스트 방법]
# ============================================================

"""
1. 통합 테스트 스크립트 실행:
   python test_alert_workflow.py
   
2. 대시보드에서 시뮬레이션:
   streamlit run dashboard.py
   → "낙상 시뮬레이션 (즉시 발동)" 버튼 클릭
   
3. 실제 낙상 감지 테스트:
   python main.py
   → 카메라 앞에서 쓰러지는 동작 수행
   → "괜찮아" 또는 무응답으로 응답 테스트
"""

# ============================================================
# [문제 해결]
# ============================================================

"""
❌ 음성이 안 나와요
→ subprocess import 확인 (voice_module.py 첫 줄)
→ Windows 오디오 드라이버 정상 확인
→ 볼륨이 음소거되지 않았는지 확인

❌ 텔레그램이 안 와요
→ settings.json에 토큰/챗ID 입력 확인
→ 봇과 1:1 대화를 시작했는지 확인
→ 인터넷 연결 확인
→ 음성 확인에서 "괜찮아"를 말하지 않았는지 확인

❌ 전화가 안 걸려요
→ "휴대폰과 연결" 앱 설치 확인
→ 스마트폰 페어링 확인
→ AUTO_CALL_ENABLED: true 확인
→ EMERGENCY_CONTACT 입력 확인

❌ 모니터 창이 안 떠요
→ 카메라 연결 확인
→ voice_module.py의 백그라운드 마이크 초기화 블로킹 제거 확인
→ cv2.waitKey() 호출 확인
"""

# ============================================================
# [버전 정보]
# ============================================================

"""
마지막 업데이트: 2026-01-17 21:58 KST
주요 수정사항:
  - PowerShell 기반 TTS로 전환 (COM 충돌 해결)
  - subprocess import 추가 (음성 출력 필수)
  - 백그라운드 마이크 초기화 블로킹 제거 (모니터 윈도우 표시 개선)
  - 전체 워크플로우 검증 완료

상태: ✅ 모든 기능 정상 작동 확인
"""
