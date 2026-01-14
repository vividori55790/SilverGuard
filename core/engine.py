import os
import sys
import time
import datetime
import json
import cv2
import numpy as np
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
        # List of {'cap': VideoCapture, 'detector': FallDetector, 'buffer': deque, 'status': ..., 'id': int}
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
            'cap': cap0,
            'detector': FallDetector(),
            'buffer': deque(maxlen=45), # 3 sec buffer
            'test_mode': using_test,
            'status': "Initializing",
            'color': (200, 200, 200),
            'fall_state': False
        })
        
        # Cam 1 (Extra)
        if extra_cam_source is not None:
            print(f"📷 추가 카메라 연결 시도: {extra_cam_source}")
            cap1 = cv2.VideoCapture(extra_cam_source)
            
            # [Auto-Fix] 만약 연결 실패했고 URL 형태라면 흔한 엔드포인트(/video)를 붙여서 재시도
            if not cap1.isOpened() and isinstance(extra_cam_source, str) and extra_cam_source.startswith('http'):
                alt_url = extra_cam_source.rstrip('/') + '/video'
                print(f"⚠️ 1차 연결 실패. 엔드포인트 자동 추가 후 재시도: {alt_url}")
                cap1 = cv2.VideoCapture(alt_url)

            if cap1.isOpened():
                print("✅ 추가 카메라 연결 성공!")
                self.cams.append({
                    'id': 1,
                    'cap': cap1,
                    'detector': FallDetector(),
                    'buffer': deque(maxlen=45),
                    'test_mode': False,
                    'status': "Initializing",
                    'color': (200, 200, 200),
                    'fall_state': False
                })
            else:
                print("❌ 추가 카메라 연결 실패 (URL을 확인해주세요. 예: http://.../video)")

        # Global Runtime State
        self.frame_count = 0
        self.skip_frames = 2
        self.last_alert_time = 0
        self.alert_cooldown = 60
        self.is_privacy_mode = False
        
    def run(self):
        print(f"🟢 모니터링 시작! (카메라 {len(self.cams)}대 가동 중)")
        sync_unsent_data()
        
        try:
            while True:
                self.frame_count += 1
                
                # Check Settings periodically
                if self.frame_count % 30 == 0:
                    utils.update_heartbeat()
                    self.is_privacy_mode = skeleton_avatar.check_privacy_mode()
                if self.frame_count % 100 == 0:
                    sync_unsent_data()

                frames_to_show = []

                for cam_data in self.cams:
                    ret, frame = cam_data['cap'].read()
                    if not ret:
                        if cam_data['test_mode']:
                            cam_data['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cam_data['cap'].read()
                        
                        if not ret: # 여전히 못 읽으면 검은 화면
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(frame, "No Signal", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Resize to standard size for consistency
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Preprocessing
                    if utils.CROP_RIGHT_HALF:
                        frame = frame[:, frame.shape[1]//2:]

                    # Video Buffering
                    cam_data['buffer'].append(frame.copy())
                    
                    # Inference
                    if self.frame_count % self.skip_frames == 0:
                        pred_cls, conf, bbox, kpts_xy, confs, is_detected = cam_data['detector'].process_frame(frame, self.skip_frames)
                        
                        if is_detected:
                            self._handle_detection(cam_data, pred_cls, conf, frame, kpts_xy, confs, bbox)
                        else:
                            # [BUG FIX] 사람이 사라지면 낙상 상태 초기화
                            # 단, 즉시 해제보다는 카운터를 두는 게 좋지만 여기선 단순화
                            cam_data['fall_state'] = False 
                            cam_data['status'] = "Monitoring..."
                            cam_data['color'] = (0, 255, 0)
                    
                    # Visualization Prepared
                    # We use cached detection data from detector object or cam_data state
                    # But detector stores last results.
                    frames_to_show.append(self._draw_overlay(frame, cam_data))

                # Display Merged
                if len(frames_to_show) > 1:
                    final_display = cv2.hconcat(frames_to_show)
                elif len(frames_to_show) == 1:
                    final_display = frames_to_show[0]
                else:
                    break

                cv2.imshow("SilverGuard Monitor", final_display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            for c in self.cams:
                c['cap'].release()
            cv2.destroyAllWindows()
            print("👋 시스템을 종료합니다.")

    def _handle_detection(self, cam_data, pred_cls, conf, frame, kpts, confs, bbox):
        # State Management
        cam_data['status'], cam_data['color'] = "Monitoring...", (0, 255, 0)
        
        if pred_cls == 1 and conf > 0.7:
            cam_data['status'], cam_data['color'] = f"FALL! ({conf*100:.0f}%)", (0, 0, 255)
            
            if not cam_data['fall_state']:
                cam_data['fall_state'] = True
                print(f"🚨 카메라 {cam_data['id']}에서 낙상 감지됨!")
                self._trigger_alert(frame, cam_data['buffer'])
        else:
            cam_data['fall_state'] = False

    def _trigger_alert(self, frame, buffer):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(utils.ALERT_DIR, f"FALL_{timestamp}.jpg")
        
        # Save Trigger Frame
        if not cv2.imwrite(save_path, frame):
            print(f"❌ 이미지 저장 실패: {save_path}")
            return
            
        print(f"📸 낙상 이미지 저장됨: {save_path}")
        
        # Save 3-sec Video
        video_path = None
        if len(buffer) > 10:
            video_save_path = os.path.join(utils.ALERT_DIR, f"FALL_VIDEO_{timestamp}.mp4")
            try:
                h, w, _ = buffer[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_save_path, fourcc, 15.0, (w, h))
                for f_img in buffer:
                    out.write(f_img)
                out.release()
                
                # 파일 생성 확인
                if os.path.exists(video_save_path) and os.path.getsize(video_save_path) > 0:
                    video_path = video_save_path
                    print(f"🎥 낙상 영상 저장 완료: {video_path} (크기: {os.path.getsize(video_path)} bytes)")
                else:
                    print("⚠️ 영상 파일이 생성되지 않았거나 비어있습니다.")

            except Exception as e:
                print(f"⚠️ 영상 저장 실패: {e}")

        # Voice Check
        # run_voice_emergency_check only takes one argument (image_path) in voice_module.py
        # Check signature: run_voice_emergency_check(image_path)
        voice_res = run_voice_emergency_check(save_path)
        
        # Dispatch Alert
        if is_internet_available():
            if time.time() - self.last_alert_time > self.alert_cooldown:
                utils.send_telegram_alert(save_path, f"🚨 낙상 발생! (결과: {voice_res})", video_path)
                self.last_alert_time = time.time()
        else:
            if time.time() - self.last_alert_time > self.alert_cooldown:
                activate_offline_safety_mode(save_path, voice_res)
                self.last_alert_time = time.time()

    def _draw_overlay(self, frame, cam_data):
        detector = cam_data['detector']
        
        if self.is_privacy_mode:
            # use detector's last known state from previous frame process
            kpts = detector.last_kpts_xy
            confs = detector.last_confs
            display_frame = skeleton_avatar.draw_virtual_avatar(frame, kpts, confs)
        else:
            display_frame = frame.copy()
            # Draw Skeleton & Box logic
            if detector.detected:
                bbox = detector.last_bbox
                kpts = detector.last_kpts_xy
                confs = detector.last_confs
                
                # Draw points
                for idx, (x, y) in enumerate(kpts):
                    if confs[idx] > 0.5:
                        cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                
                # Draw Box
                if bbox is not None:
                     cv2.rectangle(display_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), cam_data['color'], 2)
                     cv2.putText(display_frame, cam_data['status'], (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cam_data['color'], 2)
        
        # Cam ID Label
        cv2.putText(display_frame, f"CAM {cam_data['id']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return display_frame
