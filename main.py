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

# ==========================================
# [설정] 모델 파라미터
# ==========================================
SEQUENCE_LENGTH = 30
INPUT_SIZE = 54
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# [최적화 설정]
SKIP_FRAMES = 3       # 3프레임마다 1번만 추론 (나머지는 이전 결과 사용)
YOLO_IMG_SIZE = 640   # 입력 해상도 (기본 640 -> 320으로 축소하여 속도 향상)

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def main():
    print("🚀 SilverGuard: 로컬 시각화 모드 시작")
    print(f"   - 해상도: {YOLO_IMG_SIZE}px, 프레임 스킵: {SKIP_FRAMES}")
    print(f"   - 모델 경로: {utils.MODEL_DIR}")
    
    utils.ensure_dirs()

    # 1. 하드웨어 및 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - 실행 장치: {device}")

    # (A) YOLO 모델 로드
    if not os.path.exists(utils.YOLO_MODEL_PATH):
        print(f"❌ YOLO 모델을 찾을 수 없습니다: {utils.YOLO_MODEL_PATH}")
        print("   -> 'docker cp' 명령어로 모델을 가져왔는지 확인하세요.")
        return
    
    try:
        yolo_model = YOLO(utils.YOLO_MODEL_PATH)
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        return
    
    # (B) LSTM 모델 로드
    lstm_path = os.path.join(utils.MODEL_DIR, 'fall_lstm.pth')
    if not os.path.exists(lstm_path):
        print(f"❌ 학습된 LSTM 모델이 없습니다: {lstm_path}")
        return

    lstm_model = FallLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()

    # (C) 스케일러 로드
    scaler_path = os.path.join(utils.MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        print(f"❌ 스케일러 파일이 없습니다: {scaler_path}")
        return
    scaler = joblib.load(scaler_path)

    # 2. 영상 소스 (로컬 웹캠 우선)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ 웹캠(0)을 찾을 수 없습니다. 테스트 영상을 시도합니다.")
        test_video_path = os.path.join(utils.VIDEO_DIR, utils.TEST_VIDEO_NAME)
        if os.path.exists(test_video_path):
             cap = cv2.VideoCapture(test_video_path)
        else:
            print("❌ 실행 가능한 영상 소스가 없습니다.")
            return
            
    print(f"✅ 카메라 연결 성공")

    # 모션 감지기
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prev_head_y = None
    prev_angle = None
    is_fall_state = False
    
    # 텔레그램 쿨다운
    last_alert_time = 0
    ALERT_COOLDOWN = 60 

    frame_count = 0  # 프레임 스킵용 카운터
    
    # 이전 프레임의 결과 저장용 (스킵된 프레임에서 재사용)
    last_bbox = None
    last_status = "Initializing"
    last_color = (200, 200, 200)

    # 스켈레톤 연결 정보 (COCO Keypoints 기준)
    skeleton_connections = [
        (5, 7), (7, 9), (6, 8), (8, 10),      # 팔 (어깨-팔꿈치-손목)
        (11, 13), (13, 15), (12, 14), (14, 16), # 다리 (골반-무릎-발목)
        (5, 6), (11, 12),                     # 어깨선, 골반선
        (5, 11), (6, 12)                      # 몸통 (어깨-골반)
    ]

    print("🎥 모니터링 창이 열립니다. 종료하려면 'q'를 누르세요.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("영상 종료 (Loop)")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_buffer.clear()
                continue
            
            frame_count += 1

            if utils.CROP_RIGHT_HALF:
                h, w, _ = frame.shape
                frame = frame[:, w//2:]

            # ---------------------------------------------------------
            # [1] Motion Trigger
            # ---------------------------------------------------------
            fgMask = backSub.apply(frame)
            motion_pixels = cv2.countNonZero(fgMask)
            
            # 움직임이 적고 낙상 상태가 아니면 추론 건너뛰기
            if motion_pixels < utils.MOTION_THRESHOLD and not is_fall_state:
                cv2.putText(frame, "Sleep Mode (No Motion)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                # 시각화 (로컬)
                cv2.imshow("SilverGuard Local Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue 

            # ---------------------------------------------------------
            # [2] Frame Skipping
            # ---------------------------------------------------------
            if frame_count % SKIP_FRAMES != 0:
                if last_bbox is not None:
                    # 이전 박스와 상태 그대로 그리기
                    cv2.rectangle(frame, (int(last_bbox[0]), int(last_bbox[1])), (int(last_bbox[2]), int(last_bbox[3])), last_color, 2)
                    cv2.putText(frame, last_status, (int(last_bbox[0]), int(last_bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)
                
                cv2.imshow("SilverGuard Local Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # ---------------------------------------------------------
            # [3] YOLO 추론
            # ---------------------------------------------------------
            results = yolo_model(frame, verbose=False, imgsz=YOLO_IMG_SIZE)
            detected = False
            
            for r in results:
                if r.keypoints is None or len(r.keypoints) == 0: continue

                # (1) 데이터 추출
                kpts_xy = r.keypoints.xy[0].cpu().numpy() # 화면 그리기용 (픽셀 좌표)
                kpts = r.keypoints.xyn[0].cpu().numpy()   # LSTM 입력용 (정규화 좌표)
                confs = r.keypoints.conf[0].cpu().numpy()
                bbox = r.boxes.xyxy[0].cpu().numpy()
                
                if len(kpts) == 17:
                    detected = True
                    last_bbox = bbox

                    # (2) 스켈레톤 시각화 (추가된 부분)
                    # 점 그리기
                    for idx, (x, y) in enumerate(kpts_xy):
                        if confs[idx] > 0.5: # 신뢰도가 0.5 이상일 때만
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1) # 노란색 점

                    # 선 그리기
                    for p1, p2 in skeleton_connections:
                        if confs[p1] > 0.5 and confs[p2] > 0.5:
                            pt1 = (int(kpts_xy[p1][0]), int(kpts_xy[p1][1]))
                            pt2 = (int(kpts_xy[p2][0]), int(kpts_xy[p2][1]))
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 2) # 초록색 선

                    # (3) Feature Engineering
                    head_y = kpts[0][1]
                    shoulder_mid = (kpts[5] + kpts[6]) / 2
                    hip_mid = (kpts[11] + kpts[12]) / 2
                    current_angle = calculate_angle(shoulder_mid, hip_mid)
                    
                    if prev_head_y is not None:
                        head_velocity = (head_y - prev_head_y) * 30 / SKIP_FRAMES
                        angle_velocity = (current_angle - prev_angle) * 30 / SKIP_FRAMES
                    else:
                        head_velocity = 0
                        angle_velocity = 0
                    
                    prev_head_y = head_y
                    prev_angle = current_angle
                    
                    row = []
                    for i in range(17): row.extend([kpts[i][0], kpts[i][1], confs[i]])
                    row.extend([head_velocity, angle_velocity, current_angle])
                    
                    frame_buffer.append(row)
                    
                    # (4) LSTM 추론
                    last_status = "Monitoring..."
                    last_color = (0, 255, 0)
                    
                    if len(frame_buffer) == SEQUENCE_LENGTH:
                        input_seq = np.array(frame_buffer).reshape(-1, INPUT_SIZE)
                        input_seq_scaled = scaler.transform(input_seq)
                        input_tensor = torch.tensor(input_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output = lstm_model(input_tensor)
                            prob = torch.softmax(output, dim=1)
                            pred_cls = torch.argmax(prob, dim=1).item()
                            confidence = prob[0][pred_cls].item()
                        
                        if pred_cls == 1 and confidence > 0.7:
                            last_status = f"FALL DETECTED ({confidence*100:.0f}%)"
                            last_color = (0, 0, 255)
                            
                            if not is_fall_state:
                                is_fall_state = True
                                print(f"⚠️ 낙상 감지됨! ({confidence*100:.1f}%)")
                                
                                # 알림 전송
                                current_time = time.time()
                                if current_time - last_alert_time > ALERT_COOLDOWN:
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    save_path = os.path.join(utils.ALERT_DIR, f"FALL_LOCAL_{timestamp}.jpg")
                                    cv2.imwrite(save_path, frame)
                                    print("   -> 텔레그램 전송 시도...")
                                    utils.send_telegram_alert(save_path, f"🚨 [로컬 감지] 낙상 발생!\n확률: {confidence*100:.1f}%")
                                    last_alert_time = current_time
                        else:
                            is_fall_state = False
                    
                    # 박스 그리기 (스켈레톤 위에 그리기)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), last_color, 2)
                    cv2.putText(frame, last_status, (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)
            
            if not detected:
                if len(frame_buffer) > 0: frame_buffer.clear()
                prev_head_y = None
                last_bbox = None
                
            # [시각화 활성화]
            cv2.imshow("SilverGuard Local Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("사용자 중단")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("시스템 종료")

if __name__ == '__main__':
    main()