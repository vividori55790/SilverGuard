# SilverGuard/main.py
import cv2
import torch
import joblib
import numpy as np
import time
import os
import datetime
import math
from collections import deque
from ultralytics import YOLO

# 모듈 가져오기
import utils
from models import FallLSTM
from voice_module import run_voice_emergency_check
from offline_mode import is_internet_available, activate_offline_safety_mode, sync_unsent_data
import skeleton_avatar  # [NEW] 새로 만든 아바타 모듈

# ==========================================
# [설정] 모델 파라미터
# ==========================================
SEQUENCE_LENGTH = 30
INPUT_SIZE = 54
HIDDEN_SIZE = 64
NUM_LAYERS = 2
SKIP_FRAMES = 2       
YOLO_IMG_SIZE = 640   

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def main():
    print("🚀 SilverGuard 시스템을 시작합니다...")
    utils.ensure_dirs()

    print("🔍 미전송 알림 확인 중...")
    sync_unsent_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📦 모델 로드 중... (장치: {device})")
    
    yolo_model = YOLO(utils.YOLO_MODEL_PATH)
    
    lstm_path = os.path.join(utils.MODEL_DIR, 'fall_lstm.pth')
    lstm_model = FallLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()
    
    scaler = joblib.load(os.path.join(utils.MODEL_DIR, 'scaler.pkl'))
    print("✅ 모든 모델 로드 완료!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ 카메라를 찾을 수 없어 테스트 영상을 불러옵니다.")
        test_video_path = os.path.join(utils.VIDEO_DIR, utils.TEST_VIDEO_NAME)
        cap = cv2.VideoCapture(test_video_path)

    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prev_head_y, prev_angle = None, None
    is_fall_state = False
    last_alert_time = 0
    ALERT_COOLDOWN = 60 
    frame_count = 0
    
    # 변수 초기화
    last_bbox = None
    last_status = "Initializing"
    last_color = (200, 200, 200)
    last_kpts_xy = []
    last_confs = []
    
    # 현재 모드 상태 (기본값: False)
    is_privacy_mode = False

    print("🟢 모니터링 시작! (종료하려면 화면에서 'q'를 누르세요)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if utils.CROP_RIGHT_HALF:
                frame = frame[:, frame.shape[1]//2:]

            # [1] 설정 확인 (30프레임마다)
            if frame_count % 30 == 0:
                utils.update_heartbeat()
                # 모듈을 통해 프라이버시 모드인지 확인
                is_privacy_mode = skeleton_avatar.check_privacy_mode()

            # [2] Frame Skipping (연산 스킵)
            if frame_count % SKIP_FRAMES != 0:
                # 화면 그리기 로직
                if is_privacy_mode:
                    # [모듈 사용] 버추얼 아바타 화면 생성 (검은 배경 + 뼈대)
                    display_frame = skeleton_avatar.draw_virtual_avatar(frame, last_kpts_xy, last_confs)
                else:
                    # 일반 카메라 화면
                    display_frame = frame.copy()
                    if last_bbox is not None:
                        # 기존 스켈레톤 연결 정보는 skeleton_avatar 파일에 있으니 거기꺼 사용해도 됨
                        # 여기선 간단히 박스와 상태만 그림
                        cv2.rectangle(display_frame, (int(last_bbox[0]), int(last_bbox[1])), (int(last_bbox[2]), int(last_bbox[3])), last_color, 2)
                        # 점 찍기
                        for idx, (x, y) in enumerate(last_kpts_xy):
                            if idx < len(last_confs) and last_confs[idx] > 0.5:
                                cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                
                # 상태 메시지 추가
                if last_bbox is not None:
                    cv2.putText(display_frame, last_status, (int(last_bbox[0]), int(last_bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)

                if frame_count % 100 == 0:
                    sync_unsent_data()

                cv2.imshow("SilverGuard Monitor", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # [3] YOLO & LSTM 추론 (실제 계산)
            results = yolo_model(frame, verbose=False, imgsz=YOLO_IMG_SIZE)
            detected = False
            
            for r in results:
                if r.keypoints is None or len(r.keypoints) == 0: continue
                kpts_xy = r.keypoints.xy[0].cpu().numpy()
                kpts = r.keypoints.xyn[0].cpu().numpy()
                confs = r.keypoints.conf[0].cpu().numpy()
                bbox = r.boxes.xyxy[0].cpu().numpy()
                
                if len(kpts) == 17:
                    detected = True
                    last_bbox, last_kpts_xy, last_confs = bbox, kpts_xy, confs

                    head_y = kpts[0][1]
                    shoulder_mid, hip_mid = (kpts[5] + kpts[6]) / 2, (kpts[11] + kpts[12]) / 2
                    current_angle = calculate_angle(shoulder_mid, hip_mid)
                    head_v = (head_y - prev_head_y) * 30 / SKIP_FRAMES if prev_head_y else 0
                    angle_v = (current_angle - prev_angle) * 30 / SKIP_FRAMES if prev_angle else 0
                    prev_head_y, prev_angle = head_y, current_angle

                    row = []
                    for i in range(17): row.extend([kpts[i][0], kpts[i][1], confs[i]])
                    row.extend([head_v, angle_v, current_angle])
                    frame_buffer.append(row)
                    
                    last_status, last_color = "Monitoring...", (0, 255, 0)
                    
                    if len(frame_buffer) == SEQUENCE_LENGTH:
                        input_seq = scaler.transform(np.array(frame_buffer).reshape(-1, INPUT_SIZE))
                        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            prob = torch.softmax(lstm_model(input_tensor), dim=1)
                            pred_cls, conf = torch.argmax(prob, dim=1).item(), prob[0][torch.argmax(prob, dim=1).item()].item()
                        
                        if pred_cls == 1 and conf > 0.7:
                            last_status, last_color = f"FALL! ({conf*100:.0f}%)", (0, 0, 255)
                            if not is_fall_state:
                                is_fall_state = True
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                save_path = os.path.join(utils.ALERT_DIR, f"FALL_{timestamp}.jpg")
                                
                                # 사고 저장: 버추얼 모드면 스켈레톤 화면을, 아니면 원본을 저장
                                # (원한다면 여기를 if is_privacy_mode: ... 로 분기 가능)
                                cv2.imwrite(save_path, frame) 
                                
                                voice_res = run_voice_emergency_check(save_path)
                                
                                if is_internet_available():
                                    if time.time() - last_alert_time > ALERT_COOLDOWN:
                                        utils.send_telegram_alert(save_path, f"🚨 낙상 발생! (결과: {voice_res})")
                                        last_alert_time = time.time()
                                else:
                                    if time.time() - last_alert_time > ALERT_COOLDOWN:
                                        activate_offline_safety_mode(save_path, voice_res)
                                        last_alert_time = time.time()
                        else:
                            is_fall_state = False
            
            # [추론 프레임 그리기]
            if is_privacy_mode:
                display_frame = skeleton_avatar.draw_virtual_avatar(frame, last_kpts_xy, last_confs)
            else:
                display_frame = frame.copy()
                if detected:
                    for idx, (x, y) in enumerate(last_kpts_xy):
                        if last_confs[idx] > 0.5: cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                    cv2.rectangle(display_frame, (int(last_bbox[0]), int(last_bbox[1])), (int(last_bbox[2]), int(last_bbox[3])), last_color, 2)
            
            # 상태 표시 공통 적용
            if last_bbox is not None:
                cv2.putText(display_frame, last_status, (int(last_bbox[0]), int(last_bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)

            if frame_count % 100 == 0:
                sync_unsent_data()
                
            cv2.imshow("SilverGuard Monitor", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 시스템을 종료합니다.")

if __name__ == '__main__':
    main()