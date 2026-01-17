import pyttsx3
import speech_recognition as sr
import numpy as np
import time
import utils
import ctypes
import pythoncom
import requests
import json
import os
from urllib.parse import quote
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from threading import Thread

# 윈도우 키 입력을 위한 설정 (볼륨 강제 조절용)
VK_VOLUME_UP = 0xAF

def set_max_volume():
    """윈도우 시스템 볼륨을 100%로 강제 설정"""
    print("🔊 볼륨을 높이는 중...")
    
    # 방법 1: pycaw를 이용한 정석적인 볼륨 조절
    try:
        pythoncom.CoInitialize()
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(1.0, None)
        print("✅ 정석적인 방법으로 볼륨을 100%로 설정했습니다.")
        return
    except Exception as e:
        print(f"⚠️ 정석 방법 실패 ({e}), 강제 모드로 전환합니다.")

    # 방법 2: 윈도우 볼륨 업 키(Hardware Key)를 강제로 50번 누르기
    # 이 방법은 어떤 환경에서도 무조건 볼륨이 올라갑니다.
    try:
        for _ in range(50):
            ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, 0, 0) # 누름
            ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, 2, 0) # 뗌
            time.sleep(0.01)
        print("✅ 강제 키 입력을 통해 볼륨을 높였습니다.")
    except Exception as e:
        print(f"❌ 모든 볼륨 조절 실패: {e}")

def broadcast_tts_to_ipcamera(text):
    """
    연결된 IP 카메라(Android IP Webcam 앱 등)에 TTS 명령을 전송하여
    스마트폰 스피커에서도 동시에 소리가 나게 합니다.
    """
    try:
        if not os.path.exists(utils.SETTINGS_PATH): return
        
        with open(utils.SETTINGS_PATH, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            
        raw_val = str(settings.get("EXTRA_CAM", ""))
        user = str(settings.get("CAM_USER", "")).strip()
        pwd = str(settings.get("CAM_PASS", "")).strip()
        
        if not raw_val: return
        
        targets = [p.strip() for p in raw_val.split(',')]
        
        def send_tts_worker(url_base, user, pwd, msg):
            # IP Webcam Android App API: http://IP:8080/tts?text=...
            # Construct Base URL (Remove /video or similar paths)
            try:
                # Handle full URL input
                if "://" in url_base:
                    # http://192.168.1.5:8080/video -> http://192.168.1.5:8080
                    import urllib.parse
                    parsed = urllib.parse.urlparse(url_base)
                    scheme = parsed.scheme
                    netloc = parsed.netloc # includes port
                    base = f"{scheme}://{netloc}"
                else:
                    # Just IP? Assume HTTP 8080
                    base = f"http://{url_base}"
                    if ":" not in url_base: base += ":8080"
                
                # Setup Auth
                auth = None
                if user and pwd:
                    auth = (user, pwd)
                elif "@" in base:
                    # Extract auth from URL if present
                    # requests handles basic auth if passed manually, but requests.get(url_with_auth) works too usually.
                    pass 
                
                tts_url = f"{base}/tts?text={quote(msg)}"
                # print(f"📡 Sending TTS to Camera: {base}")
                
                requests.get(tts_url, auth=auth, timeout=3)
            except: 
                pass

        for t in targets:
            # Sync call might block voice, so use simple thread
            Thread(target=send_tts_worker, args=(t, user, pwd, text)).start()
            
    except Exception as e:
        print(f"⚠️ IP Camera TTS Fail: {e}")

def speak(text):
    """AI가 음성으로 메시지 출력 (PC + 스마트폰 동시 송출)"""
    print(f"📢 AI: {text}")
    
    # 1. Broadcast to Phones (Background)
    broadcast_tts_to_ipcamera(text)
    
    # 2. Local PC TTS
    try:
        # 매번 엔진을 초기화하여 충돌 방지
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text)
        engine.runAndWait()
        del engine 
    except Exception as e:
        print(f"⚠️ 음성 출력 오류: {e}")

def get_audio_rms(audio_data):
    """소리 크기 측정"""
    data = np.frombuffer(audio_data, dtype=np.int16)
    if len(data) == 0: return 0
    return np.sqrt(np.mean(data.astype(np.float64)**2))

def listen_and_analyze(timeout=10):
    """음성 인식 및 주변 소리 분석"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"🎤 {timeout}초 동안 대답을 기다리는 중...")
        try:
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=5)
            try:
                text = r.recognize_google(audio, language='ko-KR')
                return "VOICE", text
            except sr.UnknownValueError:
                rms = get_audio_rms(audio.get_raw_data())
                if rms > 1000:
                    return "SOUND_DETECTED", rms
                return "SILENCE", 0
        except Exception:
            return "SILENCE", 0

def run_voice_emergency_check(image_path):
    """낙상 발생 시 실행되는 메인 로직"""
    
    # 1차 시도 (일반 볼륨)
    speak("낙상이 감지되었습니다. 괜찮으십니까? 대답이 없으시면 구조 요청을 보냅니다.")
    status, detail = listen_and_analyze(timeout=10)
    
    # 2차 시도 (무응답 시 볼륨업 후 재질문)
    if status == "SILENCE":
        print("❓ 응답 없음: 볼륨을 최대치로 높입니다.")
        set_max_volume()
        time.sleep(1) 
        speak("잘 안 들리실 수 있어 다시 크게 여쭤보겠습니다. 괜찮으신가요? 응답이 없으시면 비상 상황으로 간주합니다.")
        status, detail = listen_and_analyze(timeout=10)

    # 최종 결과 판단
    if status == "VOICE":
        print(f"👤 인식된 대답: {detail}")
        if any(word in detail for word in ["괜찮아", "어", "나 안 다쳤어", "아무렇지 않아", "문제 없어"]):
            speak("확인되었습니다. 시스템을 정상 상태로 유지합니다.")
            return "SAFE"
        elif any(word in detail for word in ["아니", "아파", "도와줘", "살려줘", "병원", "119", "구조"]):
            speak("위급 상황임을 확인했습니다. 보호자에게 즉시 알림을 보냅니다.")
            # [긴급] 즉시 텔레그램 전송 (메인 로직과 별개로 여기서 바로 전송)
            utils.send_telegram_alert(image_path, f"🚨 [긴급 구조 요청] 사용자가 육성으로 구조를 요청했습니다!\n🗣️ 인식된 말: \"{detail}\"")
            return "EMERGENCY"
        else:
            speak("상황 확인이 정확하지 않아 일단 보호자에게 알림을 보냅니다.")
            return "CHECK_NEEDED"
            
    elif status == "SOUND_DETECTED":
        speak("이상 소음이 감지되어 즉시 보호자에게 알립니다.")
        return "CRITICAL_SOUND"
        
    else:
        print("🚨 최종 무응답: 비상 상황 확정")
        speak("응답이 전혀 없어 비상 상황으로 간주하고 구조 요청을 전송합니다.")
        return "NO_RESPONSE_EMERGENCY"

# ... (Existing code) ...

# Global stopper for background listening
_background_stopper = None
_background_recognizer = None

class BackgroundVoiceMonitor:
    def __init__(self, callback_func):
        self.callback = callback_func
        self.r = sr.Recognizer()
        self.mic = sr.Microphone()
        self.is_running = False
        self.stopper = None
        
        # Adjust ambient noise initially
        try:
            with self.mic as source:
                self.r.adjust_for_ambient_noise(source, duration=1)
        except: pass

    def start(self):
        if self.is_running: return
        print("👂 [Voice] 백그라운드 구조 요청 감지 시작...")
        self.is_running = True
        # listen_in_background creates a daemon thread
        self.stopper = self.r.listen_in_background(self.mic, self._audio_handler, phrase_time_limit=3)

    def stop(self):
        if self.stopper:
            self.stopper(wait_for_stop=False)
            self.stopper = None
        self.is_running = False
        print("🔇 [Voice] 백그라운드 감지 중지")

    def _audio_handler(self, recognizer, audio):
        if not self.is_running: return
        try:
            # Recognize text (online)
            # Short timeout to avoid blocking thread too long
            text = recognizer.recognize_google(audio, language='ko-KR')
            print(f"👂 [Voice Heard]: {text}")
            
            # Keywords
            triggers = ["도와줘", "살려줘", "119", "신고해", "구조", "아파", "긴급"]
            if any(k in text for k in triggers):
                print(f"🚨 [Voice Alert] 구조 요청 키워드 감지!: {text}")
                if self.callback:
                    self.callback(text)
                    
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            pass
        except Exception as e:
            print(f"⚠️ Voice Error: {e}")

# Global instance
_monitor = None

def start_continuous_listening(callback):
    global _monitor
    if _monitor is None:
        _monitor = BackgroundVoiceMonitor(callback)
    _monitor.start()

def stop_continuous_listening():
    global _monitor
    if _monitor:
        _monitor.stop()

if __name__ == "__main__":
    print("📢 voice_module 통합 테스트")
    # run_voice_emergency_check("test.jpg")
    
    def test_cb(text):
        print(f"Callback Triggered: {text}")
        
    start_continuous_listening(test_cb)
    while True:
        time.sleep(1)