import os
import sys
import time
import datetime
import json
import cv2
import numpy as np
import threading
from collections import deque

# Parent import support
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils
from voice_module import run_voice_emergency_check
from offline_mode import is_internet_available, activate_offline_safety_mode, sync_unsent_data
import skeleton_avatar
from core.detectors import FallDetector
from voice_module import start_continuous_listening, stop_continuous_listening

class SilverGuardEngine:
    def __init__(self):
        print("🚀 SilverGuard 엔진 초기화 중...")
        utils.ensure_dirs()
        
        # Load Settings for Camera
        extra_cam_source = None
        if os.path.exists(utils.SETTINGS_PATH):
            try:
                with open(utils.SETTINGS_PATH, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    val = settings.get("EXTRA_CAM", "")
                    if val and str(val).strip() != "":
                        # 숫자면 int형, 아니면 str형
                        extra_cam_source = int(val) if str(val).isdigit() else val
            except: pass

        # Multi-Camera Setup
        self.cams = []
        
        # Cam 0 (Default)
        cap0 = cv2.VideoCapture(0)
        using_test = False
        if not cap0.isOpened():
            print("⚠️ 기본 카메라(0)를 찾을 수 없어 테스트 영상을 불러옵니다.")
            using_test = True
            cap0 = cv2.VideoCapture(os.path.join(utils.VIDEO_DIR, utils.TEST_VIDEO_NAME))
            
        self.cams.append({
            'id': 0,
            'source': 0 if not using_test else os.path.join(utils.VIDEO_DIR, utils.TEST_VIDEO_NAME),
            'cap': cap0,
            'detector': FallDetector(),
            'buffer': deque(maxlen=150), # 버퍼 150프레임 (약 5초)
            'test_mode': using_test,
            'status': "Initializing",
            'color': (200, 200, 200),
            'fall_state': False,
            'last_retry_time': 0
        })
        
        # Store current extra cam config for dynamic updates
        self.current_extra_cam_source = extra_cam_source
        
        # Cam 1 (Extra)
        if extra_cam_source is not None:
            self._add_extra_camera(extra_cam_source)

        # Global Runtime State
        self.frame_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = 60
        self.is_privacy_mode = False
        
        # [Auto Call Config]
        self.auto_call_enabled = False
        self.emergency_contact = ""

        # Start background voice listener
        start_continuous_listening(self._on_voice_trigger)

    def run(self):
        print(f"🟢 모니터링 시작! (카메라 {len(self.cams)}대 가동 중)")
        sync_unsent_data()
        
        # [Performance] Stats & Config
        self.perf_stats = {"capture": 0.0, "inference": 0.0, "overlay": 0.0, "settings": 0.0, "idle": 0.0}
        self.target_fps = 30 # Can be updated from settings
        
        try:
            while True:
                loop_start = time.perf_counter() # Start Timer
                
                self.frame_count += 1
                
                # [Perf] Measure Settings Check
                t_set_start = time.perf_counter()
                
                # Check Settings & Model periodically (약 1초마다)
                if self.frame_count % 30 == 0:
                    self.is_privacy_mode = skeleton_avatar.check_privacy_mode()
                    # ... (rest of logic continues)
                    
                    # [NEW] Auto-Reload AI Model if updated
                    try:
                        model_path = os.path.join(utils.MODEL_DIR, 'stgcn_fall.pth')
                        if os.path.exists(model_path):
                            mtime = os.path.getmtime(model_path)
                            # Initialize last_model_time if not set
                            if not hasattr(self, 'last_model_time'): self.last_model_time = mtime
                            
                            if mtime > self.last_model_time:
                                print(f"🧠 AI 모델 업데이트 감지! ({mtime}) -> 엔진에 즉시 반영합니다.")
                                self.last_model_time = mtime
                                # Reload detectors
                                from core.detection import FallDetector # Ensure import
                                for cam in self.cams:
                                    # Create new detector to load new weights
                                    # Preserve sensitivity settings
                                    old_conf = cam['detector'].confidence_threshold
                                    old_strict = cam['detector'].strictness
                                    cam['detector'] = FallDetector(is_file=False) # Source doesn't matter for init logic mostly
                                    cam['detector'].set_sensitivity(old_conf, old_strict)
                                print("✅ 모든 카메라의 AI 모델 리로드 완료.")
                    except Exception as e:
                        print(f"⚠️ 모델 리로드 실패: {e}")

                    try:
                        with open(utils.SETTINGS_PATH, 'r', encoding='utf-8') as f:
                            settings = json.load(f)
                            # Update Auto Call Settings
                            self.auto_call_enabled = settings.get("AUTO_CALL_ENABLED", False)
                            self.emergency_contact = settings.get("EMERGENCY_CONTACT", "")
                            
                            for cam in self.cams:
                                cam['detector'].set_sensitivity(
                                    settings.get("AI_CONFIDENCE", 0.65), settings.get("AI_STRICTNESS", "Medium")
                                )
                            # Camera Config Logic
                            raw_val = str(settings.get("EXTRA_CAM", ""))
                            extra_cam_enabled = settings.get("EXTRA_CAM_ENABLED", True) # [NEW] Check enabled
                            
                            cam_user = str(settings.get("CAM_USER", "")).strip()
                            cam_pass = str(settings.get("CAM_PASS", "")).strip()
                            current_creds = (cam_user, cam_pass)

                            new_sources = []
                            # Only add sources if enabled and string exists
                            if extra_cam_enabled and raw_val.strip():
                                parts = [p.strip() for p in raw_val.split(',')]
                                for p in parts:
                                    if p: new_sources.append(int(p) if p.isdigit() else p)
                            
                            # Compare with LAST LOADED config (Raw User Input)
                            if not hasattr(self, 'last_raw_sources'):
                                self.last_raw_sources = []
                                self.last_creds = ("", "") 
                            
                            # Check if Sources OR Credentials changed directly
                            if new_sources != self.last_raw_sources or current_creds != self.last_creds:
                                print(f"🔄 카메라 설정/계정 변경 감지: {self.last_raw_sources} -> {new_sources}")
                                self.last_raw_sources = new_sources 
                                self.last_creds = current_creds
                                self._update_cameras(new_sources, cam_user, cam_pass)
                    except Exception as e: 
                        # print(f"Settings Error: {e}") 
                        pass
                    
                    active_sources = [c.get('source', 'Unknown') for c in self.cams]
                    cam_status_info = {"active_cameras": len(self.cams), "camera_ids": active_sources}
                    utils.update_heartbeat(cam_status_info)

                # Sync offline data
                if self.frame_count % 100 == 0:
                    sync_unsent_data()

                # [SIMULATION TRIGGER CHECK]
                sim_signal_path = os.path.join(utils.DATA_DIR, 'TRIGGER_FALL_SIM.signal')
                if os.path.exists(sim_signal_path):
                    try:
                        os.remove(sim_signal_path)
                        print("🚀 [TEST] 낙상 시뮬레이션 버튼에 의해 강제 트리거됨!")
                        
                        # Use first camera buffer if available
                        if len(self.cams) > 0:
                            target_cam = self.cams[0]
                            # Use last frame or black
                            ret, frame = target_cam['cap'].read()
                            if not ret: 
                                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                                cv2.putText(frame, "SIMULATION TEST", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            else:
                                frame = cv2.resize(frame, (640, 480))
                                
                            self._trigger_alert(frame, target_cam['buffer'])
                    except Exception as e:
                        print(f"Simulation Error: {e}")

                frames_to_show = []

                for cam_data in self.cams:
                    ret, frame = cam_data['cap'].read()
                    
                    # [Robustness] Handle read failure
                    if not ret:
                        if cam_data['test_mode']:
                            cam_data['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cam_data['cap'].read()
                        else:
                            # Reconnect Logic
                            current_time = time.time()
                            if current_time - cam_data.get('last_retry_time', 0) > 3.0:
                                cam_data['last_retry_time'] = current_time
                                print(f"⚠️ 카메라 {cam_data['id']} 신호 없음... 재연결 시도 중...")
                                try:
                                    # Reuse source parsing logic if valid
                                    src = cam_data['source']
                                    if isinstance(src, str) and src.startswith("rtsp"):
                                         os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                                    
                                    cam_data['cap'].release()
                                    cam_data['cap'] = cv2.VideoCapture(src)
                                    
                                    if cam_data['cap'].isOpened():
                                        print(f"✅ 카메라 {cam_data['id']} 재연결 성공!")
                                        cam_data['warmup'] = 60 # Reset warmup
                                        ret, frame = cam_data['cap'].read()
                                    
                                    if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                                        del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
                                except Exception as e:
                                    print(f"❌ 재연결 중 에러: {e}")

                        if not ret:
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(frame, "No Signal / Reconnecting...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Resize
                    try:
                        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                            frame = cv2.resize(frame, (640, 480))
                        else:
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    except:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)

                    if utils.CROP_RIGHT_HALF:
                        frame = frame[:, frame.shape[1]//2:]

                    # [Robustness] Warmup Period (Ignore first N frames after connection)
                    if cam_data.get('warmup', 0) > 0:
                        cam_data['warmup'] -= 1
                        cv2.putText(frame, f"Initializing... {cam_data['warmup']}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Update buffer but SKIP inference
                        display = self._draw_overlay(frame, cam_data)
                        cam_data['buffer'].append(display.copy())
                        frames_to_show.append(display)
                        continue

                    # Inference Skipping
                    run_inference = (self.frame_count % 2 == 0)
                    
                    t_inf_s = time.perf_counter()
                    if run_inference:
                        try:
                            pred_cls, conf, _, _, _, _, reason = cam_data['detector'].process(frame, timestamp=time.time())
                            kpts = cam_data['detector'].last_kpts_xy
                            confs = cam_data['detector'].last_confs
                            bbox = cam_data['detector'].last_bbox
                            self._handle_detection(cam_data, pred_cls, conf, frame, kpts, confs, bbox, reason)
                        except Exception as e:
                            print(f"⚠️ Inference Error: {e}")
                    
                    # Update Inference Stats (EMA)
                    self.perf_stats['inference'] = self.perf_stats['inference'] * 0.9 + (time.perf_counter() - t_inf_s) * 0.1

                    t_ovr_s = time.perf_counter()
                    display = self._draw_overlay(frame, cam_data)
                    # Update Overlay Stats (EMA)
                    self.perf_stats['overlay'] = self.perf_stats['overlay'] * 0.9 + (time.perf_counter() - t_ovr_s) * 0.1

                    cam_data['buffer'].append(display.copy())
                    frames_to_show.append(display)

                if len(frames_to_show) > 1:
                    final_display = cv2.hconcat(frames_to_show)
                elif len(frames_to_show) == 1:
                    final_display = frames_to_show[0]
                else:
                    final_display = np.zeros((480, 640, 3), dtype=np.uint8)

                cv2.imshow("SilverGuard Monitor", final_display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # [FPS Control & Stats Reporting]
                loop_dt = time.perf_counter() - loop_start
                target_dt = 1.0 / self.target_fps
                sleep_t = target_dt - loop_dt
                
                if sleep_t > 0:
                    time.sleep(sleep_t)
                    self.perf_stats['idle'] = self.perf_stats['idle'] * 0.9 + sleep_t * 0.1
                    loop_dt += sleep_t # Total time includes sleep
                else:
                     self.perf_stats['idle'] = 0.0
                
                # Capture time estimation (Total - (Inf + Overlay + Settings + Idle))
                # Note: 'Settings' is small/occasional. Capture includes Overhead.
                non_capture_time = self.perf_stats['inference'] + self.perf_stats['overlay'] + self.perf_stats['idle']
                est_capture = max(0.0, loop_dt - non_capture_time)
                self.perf_stats['capture'] = self.perf_stats['capture'] * 0.9 + est_capture * 0.1

                # Update Heartbeat with Rich Stats (Every ~15 frames)
                if self.frame_count % 15 == 0:
                     active_sources = [c.get('source', 'Unknown') for c in self.cams]
                     status = {
                         "active_cameras": len(self.cams), 
                         "camera_ids": active_sources,
                         "fps_real": 1.0/max(0.001, loop_dt),
                         "perf": {
                             "Capture (IO)": round(self.perf_stats['capture'] * 1000, 1),
                             "AI Inference": round(self.perf_stats['inference'] * 1000, 1),
                             "Visual Overlay": round(self.perf_stats['overlay'] * 1000, 1),
                             "Idle (Free)": round(self.perf_stats['idle'] * 1000, 1)
                         }
                     }
                     utils.update_heartbeat(status)
                     
        except Exception as e:
            print(f"\n❌ [CRITICAL ERROR] 시스템 중단: {e}")
            import traceback
            traceback.print_exc()
        finally:
            stop_continuous_listening()
            for c in self.cams:
                if c['cap']: c['cap'].release()
            cv2.destroyAllWindows()
            print("👋 시스템을 종료합니다.")

    def _on_voice_trigger(self, text):
        """Called by background voice monitor thread"""
        print(f"🎤 [Voice Callback] Triggered: {text}")
        
        # Trigger an alert using the primary camera's current buffer
        if len(self.cams) > 0:
             # Just use the first camera for context
             cam = self.cams[0]
             
             # Pause listener to avoid conflict
             stop_continuous_listening()
             
             try:
                 # Manually trigger alert
                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                 readable_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                 
                 # Create a dummy frame or use last frame
                 if cam['buffer']:
                     frame = cam['buffer'][-1]
                 else:
                     frame = np.zeros((480, 640, 3), dtype=np.uint8)
                     cv2.putText(frame, "VOICE ALERT", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                 save_path = os.path.join(utils.ALERT_DIR, f"VOICE_ALERT_{timestamp}.jpg")
                 cv2.imwrite(save_path, frame)
                 
                 msg = f"🗣️ [음성 구조 요청 감지]\n시간: {readable_time}\n내용: \"{text}\""
                 
                 if is_internet_available():
                     utils.send_telegram_alert(save_path, msg, None)
                     
                     # [Auto Call on Voice Command]
                     if self.auto_call_enabled and self.emergency_contact:
                        phone_nums = str(self.emergency_contact).split(',')
                        if phone_nums:
                            print(f"🗣️ 음성 구조 요청으로 전화 발신 시도...")
                            utils.make_phone_call(phone_nums[0].strip())
                 
                 # Restart listener after a short delay
                 time.sleep(2)
                 start_continuous_listening(self._on_voice_trigger)
                 
             except Exception as e:
                 print(f"Voice Alert Error: {e}")
                 # Ensure restart
                 start_continuous_listening(self._on_voice_trigger)

    def _handle_detection(self, cam_data, pred_cls, conf, frame, kpts, confs, bbox, reason=""):
        # State Management
        cam_data['status'], cam_data['color'] = "Monitoring...", (0, 255, 0)
        
        # pred_cls == 1 이면 'Fall' 이라고 가정
        if pred_cls == 1 and conf > 0.7:
            # 상태 메시지에 감지 원인(reason) 포함
            status_text = f"FALL! ({conf*100:.0f}%)"
            if reason:
                 status_text += f" [{reason}]"
            
            cam_data['status'], cam_data['color'] = status_text, (0, 0, 255)
            
            if not cam_data['fall_state']:
                cam_data['fall_state'] = True
                print(f"🚨 카메라 {cam_data['id']}에서 낙상 감지됨!")
                
                # 알림 전송 (데이터 저장 포함)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 1. Save Learning Data (Numpy)
                # cam_data['detector'].frame_buffer is deque of (17, 3)
                try:
                    raw_seq = np.array(cam_data['detector'].frame_buffer)
                    npy_path = os.path.join(utils.ALERT_DIR, f"FALL_{timestamp}.npy")
                    np.save(npy_path, raw_seq)
                except Exception as e:
                    print(f"⚠️ 학습 데이터 저장 실패: {e}")

                # [Fix] Create Snapshot of buffer for Thread Safety
                # 메인 스레드가 계속 업데이트하는 버퍼를 스레드에서 읽으면 충돌/에러 발생 가능
                buffer_snapshot = list(cam_data['buffer'])

                if self.is_privacy_mode:
                    safe_img = skeleton_avatar.render_privacy_frame(frame.shape, kpts, confs)
                    self._process_alert_async(safe_img, buffer_snapshot, timestamp)
                else:
                    self._process_alert_async(frame, buffer_snapshot, timestamp)
        else:
            cam_data['fall_state'] = False

    def _process_alert_async(self, frame, buffer, timestamp):
        """Run alert logic (save, voice, telegram) in a separate thread to prevent UI freeze"""
        if not hasattr(self, 'is_processing_alert'): 
            self.is_processing_alert = False
            
        if self.is_processing_alert:
            return
            
        self.is_processing_alert = True
        
        def worker():
            try:
                self._trigger_alert(frame, buffer, timestamp_override=timestamp)
            except Exception as e:
                print(f"⚠️ Alert Worker Failed: {e}")
            finally:
                self.is_processing_alert = False
                
        threading.Thread(target=worker, daemon=True).start()

    def _trigger_alert(self, frame, buffer, timestamp_override=None):
        if timestamp_override:
            timestamp = timestamp_override
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # [추가] 메시지용 가독성 좋은 시간 포맷
        readable_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        save_path = os.path.join(utils.ALERT_DIR, f"FALL_{timestamp}.jpg")
        
        # Save Frame
        try:
             if not cv2.imwrite(save_path, frame):
                 print(f"❌ 이미지 저장 실패: {save_path}")
                 return
             print(f"📸 낙상 이미지 저장됨: {save_path}")
        except Exception as e:
             print(f"❌ 이미지 저장 중 에러: {e}")
             return

        pass 
        
        # [Robustness] Save Video Logic with Better Codec Fallback
        video_path = None
        if len(buffer) > 20: 
            video_save_path = os.path.join(utils.ALERT_DIR, f"FALL_VIDEO_{timestamp}.mp4")
            try:
                # Check dimensions
                if len(buffer) == 0: raise Exception("Buffer empty")
                h, w, _ = buffer[0].shape
                if h <= 0 or w <= 0: raise Exception("Invalid frame dimensions")

                # Try codecs in order: mp4v (Most compatible), XVID (Robust)
                # Removed 'avc1' to avoid libopenh264 errors on some Windows setups
                codecs_to_try = ['mp4v', 'XVID']
                out = None
                
                for codec in codecs_to_try:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        temp_out = cv2.VideoWriter(video_save_path, fourcc, 30.0, (w, h))
                        
                        if temp_out.isOpened():
                            # print(f"🎥 코덱 '{codec}'으로 영상 저장 시도...")
                            for f in buffer:
                                temp_out.write(f)
                            temp_out.release()
                            out = temp_out
                            break # Success
                    except: continue

                if os.path.exists(video_save_path) and os.path.getsize(video_save_path) > 1000:
                    video_path = video_save_path
                    print(f"🎥 낙상 영상 저장 완료")
                else:
                    print("⚠️ 영상 파일 생성 실패 (코덱 호환성 이슈 가능성)")
            except Exception as e:
                print(f"⚠️ 영상 저장 실패: {e}")

        # Voice Check
        # Pause background listener while active check runs
        stop_continuous_listening()
        time.sleep(1.0) # [Robustness] Wait for mic to be fully released
        
        try:
            voice_res = run_voice_emergency_check(save_path)
        finally:
            # Resume background listener
            start_continuous_listening(self._on_voice_trigger)
        
        # [추가] 알림 메시지에 시간 포함
        alert_msg = f"🚨 낙상 발생! (시간: {readable_time})\n결과: {voice_res}"

        # Dispatch Alert
        # Dispatch Alert only if NOT Safe
        if voice_res != "SAFE":
            if is_internet_available():
                if time.time() - self.last_alert_time > self.alert_cooldown:
                    # 온라인이면 바로 전송
                    utils.send_telegram_alert(save_path, alert_msg, video_path)
                    
                    # [Auto Call - Windows Phone Link]
                    if self.auto_call_enabled and self.emergency_contact:
                        phone_nums = str(self.emergency_contact).split(',')
                        if phone_nums:
                            utils.make_phone_call(phone_nums[0].strip())
                    
                    self.last_alert_time = time.time()
            else:
                if time.time() - self.last_alert_time > self.alert_cooldown:
                    # 오프라인이면 대기열에 저장
                    offline_msg = f"낙상 감지! (시간: {readable_time})\n결과: {voice_res}"
                    activate_offline_safety_mode(save_path, offline_msg, video_path)
                    self.last_alert_time = time.time()
        else:
            print("✅ 사용자 확인 결과 '안전'하므로 알림을 전송하지 않습니다.")

    def _draw_overlay(self, frame, cam_data):
        detector = cam_data['detector']
        
        if self.is_privacy_mode:
            kpts = detector.last_kpts_xy
            confs = detector.last_confs
            # 아바타(종이인형) 모드로 그리기
            display_frame = skeleton_avatar.draw_virtual_avatar(frame, kpts, confs)
        else:
            display_frame = frame.copy()
            # Draw Skeleton & Box logic
            if detector.detected:
                bbox = detector.last_bbox
                kpts = detector.last_kpts_xy
                confs = detector.last_confs
                
                # Draw points
                if kpts is not None:
                    for idx, (x, y) in enumerate(kpts):
                        if confs[idx] > 0.5:
                            cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                
                # Draw Box
                if bbox is not None:
                    cv2.rectangle(display_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), cam_data['color'], 2)
                    cv2.putText(display_frame, cam_data['status'], (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cam_data['color'], 2)
        
        # Cam ID Label
        cv2.putText(display_frame, f"CAM {cam_data['id']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # [New] Date/Time Overlay
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, time_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return display_frame

    def _update_cameras(self, source_list, user="", password=""):
        # 1. Remove all extra cameras (keep id=0)
        self.cams = [c for c in self.cams if c['id'] == 0]
        
        # 2. Add new cameras
        for idx, src in enumerate(source_list):
            self._add_extra_camera(src, cam_id=idx+1, user=user, password=password)

    def _add_extra_camera(self, source, cam_id=1, user="", password=""):
        # Scan and connect in a separate thread to prevent freezing
        t = threading.Thread(target=self._scan_and_connect_worker, args=(source, cam_id, user, password))
        t.daemon = True
        t.start()
        
    def _scan_and_connect_worker(self, source, cam_id, user, password):
        # [Security] Log source without credentials if possible, or just masked
        masked_user = f"{user[:2]}***" if user and len(user) > 2 else "***"
        print(f"📷 [Async] 추가 카메라 연결 시도 (ID={cam_id}, User={masked_user}): {source}")
        
        # Helper to inject auth
        def inject_auth(url, u, p):
            if not u: return url
            if "@" in url: return url # Already has auth
            if "://" in url:
                scheme, rest = url.split("://", 1)
                return f"{scheme}://{u}:{p}@{rest}"
            return url 
            
        # 1. Typo Fix
        final_source = source
        if isinstance(source, str) and source.startswith('rstp'):
            final_source = source.replace('rstp', 'rtsp', 1)
            print(f"🔧 주소 오타 자동 수정: {source} -> {final_source}")
        
        candidates = []
        
        if isinstance(final_source, str):
            # Extract IP part if possible to generate fallbacks
            base_addr = None
            
            # A. If it's pure IP:Port (No Protocol)
            if "://" not in final_source and ("." in final_source):
                base_addr = final_source.strip()
                candidates.append(final_source) 
                
            # B. If it's a full URL (Has Protocol)
            elif "://" in final_source:
                try:
                    prefix, rest = final_source.split("://", 1)
                    if "/" in rest:
                        base_addr = rest.split("/", 1)[0]
                    else:
                        base_addr = rest
                    
                    # USER INPUT PRIORITY
                    # Inject Auth if missing and user provided it
                    if user and "@" not in final_source:
                        authed_source = f"{prefix}://{user}:{password}@{rest}"
                        candidates.append(authed_source)
                        candidates.append(final_source) # Try without auth too
                    else:
                        candidates.append(final_source)
                except:
                    base_addr = None
                    candidates.append(final_source)
            else:
                candidates.append(final_source)

            # Generate smart fallbacks if we found a base address
            if base_addr and "@" in base_addr: # Strip auth from base addr for fallback generation
                base_addr = base_addr.split("@")[1]

            if base_addr:
                print(f"🔍 감지된 주소({base_addr})를 기반으로 추가 후보를 생성합니다...")
                
                # Prepare pure IP (Remove port if present)
                ip_only = base_addr
                if ":" in base_addr and not base_addr.endswith("]"):
                     parts = base_addr.split(":")
                     if parts[-1].isdigit():
                         ip_only = ":".join(parts[:-1])

                # Base templates configuration: (Pattern, UsePureIP)
                # UsePureIP=True means we force using the IP without port because the template adds a port.
                templates_config = [
                    # [User Feedback] High priority for IP Webcam standard (http://ip:8080/video)
                    ("http://{}:8080/video", True),
                    ("http://{}/video", False),
                    
                    ("rtsp://{}/h264_pcm.sdp", False),
                    ("rtsp://{}/h264_ulaw.sdp", False),
                    ("http://{}/videofeed", False),
                    ("rtsp://{}/live/ch0", False),
                    ("rtsp://{}/stream1", False),
                    ("rtsp://{}/main", False),
                    ("rtsp://{}/live/main", False),
                    ("http://{}/", False),
                    ("http://{}:8080/video", True), # Fallback duplicate just in case
                ]
                
                smart_fallbacks = []
                for t_pat, use_pure_ip in templates_config:
                    # Choose correct argument
                    arg = ip_only if use_pure_ip else base_addr
                    
                    raw_url = t_pat.format(arg)
                    if user:
                        smart_fallbacks.append(inject_auth(raw_url, user, password))
                    smart_fallbacks.append(raw_url) # Add non-auth version too
                
                for fb in smart_fallbacks:
                    # Append strict fallbacks ONLY if they differ from existing candidates
                    if fb not in candidates and str(fb) != str(final_source):
                        candidates.append(fb)
        else:
            candidates = [source]
            
        # [Security] Don't log with full passwords
        print(f"📋 연결 후보 수: {len(candidates)}개 (인증/비인증 조합)")

        cap = None
        resolved_src = None
        
        # Set shorter timeout for scanning
        os.environ["OPENCV_FFMPEG_OPEN_TIMEOUT_MS"] = "3000" 
        
        for cand in candidates:
            cand_str = str(cand)
            is_rtsp = cand_str.startswith('rtsp')
            is_http = cand_str.startswith('http')
            
            if is_rtsp:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            elif is_http:
                if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                    del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
            
            # Masking for log
            log_cand = cand_str
            if user and password and password in cand_str:
                log_cand = cand_str.replace(password, "****")
            
            print(f"   👉 [Async] 연결 시도: {log_cand} ...")
            try:
                temp_cap = cv2.VideoCapture(cand)
                
                # Auto-Fix HTTP Specifics (Only if user input failed)
                if not temp_cap.isOpened() and is_http and cand_str == final_source and not cand_str.endswith('/video'):
                     alt = cand_str.rstrip('/') + '/video'
                     print(f"      (HTTP 자동 보정 시도: {alt})")
                     # Try the alt immediately
                     if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                         del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
                     temp_cap_alt = cv2.VideoCapture(alt)
                     if temp_cap_alt.isOpened():
                         temp_cap = temp_cap_alt
                         cand = alt
                         resolved_src = cand
                         print(f"      ✅ 자동 보정 주소 연결 성공!")

                if temp_cap.isOpened():
                    ret, _ = temp_cap.read()
                    if ret:
                        print(f"   ✅ [Async] 연결 성공!")
                        cap = temp_cap
                        resolved_src = cand
                        break
                    else:
                        temp_cap.release()
                        print(f"   ❌ 연결 실패 (신호 없음)")
                else:
                     pass
            except Exception as e:
                print(f"   ⚠️ [Async] 연결 에러: {e}")
                 
        # Cleanup Env
        if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
            del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        if "OPENCV_FFMPEG_OPEN_TIMEOUT_MS" in os.environ:
            del os.environ["OPENCV_FFMPEG_OPEN_TIMEOUT_MS"]

        if cap and cap.isOpened():
            self.cams.append({
                'id': cam_id,
                'source': resolved_src,
                'cap': cap,
                'detector': FallDetector(),
                'buffer': deque(maxlen=150), 
                'test_mode': False,
                'status': "Stabilizing...",
                'color': (200, 200, 200),
                'fall_state': False,
                'last_retry_time': 0,
                'warmup': 60
            })
        else:
            print(f"❌ [Async] 추가 카메라 연결 실패 (ID={cam_id}).")

    # Legacy method replaced by _update_cameras
    def _reload_extra_camera(self, new_source):
        self._update_cameras([new_source] if new_source else [])
