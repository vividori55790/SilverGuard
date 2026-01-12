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
import utils
from models import FallLSTM
from voice_module import run_voice_emergency_check

# ==========================================
# [설정] 모델 파라미터
# ==========================================
SEQUENCE_LENGTH = 30
INPUT_SIZE = 54
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# [최적화 설정]
SKIP_FRAMES = 3       # 3프레임마다 1번만 추론
YOLO_IMG_SIZE = 640   # 입력 해상도

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def main():
    print("🚀 SilverGuard: 로컬 시각화 및 음성 모듈 통합 모드 시작")
    utils.ensure_dirs()

    # 1. 하드웨어 및 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - 실행 장치: {device}")

    # (A) YOLO 모델 로드
    yolo_model = YOLO(utils.YOLO_MODEL_PATH)
    
    # (B) LSTM 모델 로드
    lstm_path = os.path.join(utils.MODEL_DIR, 'fall_lstm.pth')
    lstm_model = FallLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()

    # (C) 스케일러 로드
    scaler = joblib.load(os.path.join(utils.MODEL_DIR, 'scaler.pkl'))

    # 2. 영상 소스
    cap = cv2.VideoCapture(0) # 웹캠 우선
    if not cap.isOpened():
        test_video_path = os.path.join(utils.VIDEO_DIR, utils.TEST_VIDEO_NAME)
        cap = cv2.VideoCapture(test_video_path)

    # 모션 감지기 및 변수 초기화
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prev_head_y, prev_angle = None, None
    is_fall_state = False
    last_alert_time = 0
    ALERT_COOLDOWN = 60 
    frame_count = 0
    last_bbox, last_status, last_color = None, "Initializing", (200, 200, 200)

    # 스켈레톤 연결 정보
    skeleton_connections = [
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
        (5, 6), (11, 12), (5, 11), (6, 12)
    ]

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if utils.CROP_RIGHT_HALF:
                frame = frame[:, frame.shape[1]//2:]

            # [1] Motion Trigger
            fgMask = backSub.apply(frame)
            if cv2.countNonZero(fgMask) < utils.MOTION_THRESHOLD and not is_fall_state:
                cv2.imshow("SilverGuard Local Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue 

            # [2] Frame Skipping
            if frame_count % SKIP_FRAMES != 0:
                if last_bbox is not None:
                    cv2.rectangle(frame, (int(last_bbox[0]), int(last_bbox[1])), (int(last_bbox[2]), int(last_bbox[3])), last_color, 2)
                    cv2.putText(frame, last_status, (int(last_bbox[0]), int(last_bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)
                cv2.imshow("SilverGuard Local Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # [3] YOLO & LSTM 추론
            results = yolo_model(frame, verbose=False, imgsz=YOLO_IMG_SIZE)
            detected = False
            
            for r in results:
                if r.keypoints is None or len(r.keypoints) == 0: continue
                kpts_xy = r.keypoints.xy[0].cpu().numpy()
                kpts = r.keypoints.xyn[0].cpu().numpy()
                confs = r.keypoints.conf[0].cpu().numpy()
                bbox = r.boxes.xyxy[0].cpu().numpy()
                
                if len(kpts) == 17:
                    detected, last_bbox = True, bbox
                    # 스켈레톤 시각화
                    for idx, (x, y) in enumerate(kpts_xy):
                        if confs[idx] > 0.5: cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                    for p1, p2 in skeleton_connections:
                        if confs[p1] > 0.5 and confs[p2] > 0.5:
                            cv2.line(frame, (int(kpts_xy[p1][0]), int(kpts_xy[p1][1])), (int(kpts_xy[p2][0]), int(kpts_xy[p2][1])), (0, 255, 0), 2)

                    # Feature Engineering
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
                                # 사고 캡처 저장 (항상 수행하여 voice_module에 전달)
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                save_path = os.path.join(utils.ALERT_DIR, f"FALL_LOCAL_{timestamp}.jpg")
                                cv2.imwrite(save_path, frame)
                                
                                # 텔레그램은 쿨타임 때만 전송
                                if time.time() - last_alert_time > ALERT_COOLDOWN:
                                    utils.send_telegram_alert(save_path, f"🚨 낙상 발생! ({conf*100:.1f}%)")
                                    last_alert_time = time.time()

                                # 음성 확인 실행
                                voice_res = run_voice_emergency_check(save_path)
                                print(f"🎙️ 음성 결과: {voice_res}")
                        else:
                            is_fall_state = False
                    
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), last_color, 2)
                    cv2.putText(frame, last_status, (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)
            
            if not detected:
                if len(frame_buffer) > 0: frame_buffer.clear()
                prev_head_y, last_bbox = None, None
                
            cv2.imshow("SilverGuard Local Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()