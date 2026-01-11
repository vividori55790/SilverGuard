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
from models import FallLSTM  # models.py에서 불러옴

# ==========================================
# [설정]
# ==========================================
SEQUENCE_LENGTH = 30  # 학습 때 설정한 윈도우 크기 (30프레임)
INPUT_SIZE = 54       # (17개 키포인트 * 3) + (속도 등 추가 피처 3개)
HIDDEN_SIZE = 64
NUM_LAYERS = 2

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def main():
    print("🚀 SilverGuard: LSTM 기반 실시간 낙상 감지 시작")
    utils.ensure_dirs()

    # 1. 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - 가속 장치: {device}")

    # 2. 모델 로드
    print("   - 모델 로딩 중...")
    yolo_model = YOLO(utils.YOLO_MODEL_PATH)
    
    # LSTM 모델 구조 초기화 및 가중치 로드
    lstm_model = FallLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    lstm_path = os.path.join(utils.MODEL_DIR, 'fall_lstm.pth')
    
    if not os.path.exists(lstm_path):
        print(f"❌ 오류: 학습된 LSTM 모델이 없습니다 ({lstm_path}). train_lstm.py를 먼저 실행하세요.")
        return
        
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval() # 평가 모드 전환

    # 스케일러 로드 (학습 데이터와 똑같은 기준으로 정규화해야 함)
    scaler_path = os.path.join(utils.MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        print("❌ 오류: 스케일러 파일이 없습니다. train_lstm.py 실행 시 생성됩니다.")
        return
    scaler = joblib.load(scaler_path)

    # 3. 영상 소스 설정
    test_video_path = os.path.join(utils.VIDEO_DIR, utils.TEST_VIDEO_NAME)
    # 파일이 있으면 파일 사용, 없으면 0번 카메라(웹캠)
    video_source = test_video_path if os.path.exists(test_video_path) else 0
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("❌ 영상을 열 수 없습니다.")
        return

    # 4. 실시간 데이터 버퍼 (최근 30프레임 저장용)
    # 사람 ID별로 버퍼를 관리해야 여러 명일 때 안 섞이지만, 
    # 여기서는 간단히 '화면 내 가장 크게 잡힌 1명'만 추적한다고 가정합니다.
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # 이전 프레임 정보 (속도 계산용)
    prev_head_y = None
    prev_angle = None
    
    # 낙상 상태 관리
    is_fall_state = False
    fall_start_time = None
    
    print("✅ 감시 시작 (Ctrl+C로 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("영상 종료/루프")
            # 무한 루프 원하면 아래 주석 해제
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # frame_buffer.clear()
            # prev_head_y = None
            # continue
            break

        # (선택) 테스트 영상일 경우 오른쪽 절반만 자르기 (UR Fall 데이터셋 특성)
        # 웹캠 사용 시에는 이 부분을 주석 처리하세요.
        if utils.CROP_RIGHT_HALF:
             h, w, _ = frame.shape
             frame = frame[:, w//2:]

        # YOLO 추론
        results = yolo_model(frame, verbose=False)
        
        # 사람이 감지되었는지 확인
        detected = False
        
        for r in results:
            if r.keypoints is None or len(r.keypoints) == 0: continue

            # 가장 신뢰도가 높거나 크게 잡힌 사람 1명만 선택 (단순화)
            # 실제 배포판에서는 ID Tracking(ByteTrack 등)이 필요할 수 있음
            kpts = r.keypoints.xyn[0].cpu().numpy() # (17, 2)
            confs = r.keypoints.conf[0].cpu().numpy() # (17,)
            bbox = r.boxes.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
            
            if len(kpts) == 17:
                detected = True
                
                # --- Feature Engineering (preprocess_urfall_velocity.py와 동일 로직) ---
                head_y = kpts[0][1]
                shoulder_mid = (kpts[5] + kpts[6]) / 2
                hip_mid = (kpts[11] + kpts[12]) / 2
                current_angle = calculate_angle(shoulder_mid, hip_mid)
                
                if prev_head_y is not None:
                    head_velocity = (head_y - prev_head_y) * 30
                    angle_velocity = (current_angle - prev_angle) * 30
                else:
                    head_velocity = 0
                    angle_velocity = 0
                
                # 상태 업데이트
                prev_head_y = head_y
                prev_angle = current_angle
                
                # 입력 데이터 벡터 생성 (54차원)
                row = []
                # 1. Keypoints (x, y, conf) - 51개
                for i in range(17):
                    row.extend([kpts[i][0], kpts[i][1], confs[i]])
                # 2. Derived Features - 3개
                row.extend([head_velocity, angle_velocity, current_angle])
                
                # 버퍼에 추가
                frame_buffer.append(row)
                
                # --- LSTM 추론 (데이터가 30프레임 찼을 때만 수행) ---
                status_text = "Analyzing..."
                color = (0, 255, 0) # Green
                
                if len(frame_buffer) == SEQUENCE_LENGTH:
                    # (1, 30, 54) 형태로 변환
                    input_seq = np.array(frame_buffer) # (30, 54)
                    
                    # 스케일링 적용 (학습 때 2차원으로 펴서 했으므로 똑같이)
                    input_seq_2d = input_seq.reshape(-1, INPUT_SIZE)
                    input_seq_scaled = scaler.transform(input_seq_2d)
                    input_tensor = torch.tensor(input_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # 예측
                    with torch.no_grad():
                        output = lstm_model(input_tensor)
                        prob = torch.softmax(output, dim=1)
                        pred_cls = torch.argmax(prob, dim=1).item()
                        confidence = prob[0][pred_cls].item()
                    
                    # 결과 처리
                    if pred_cls == 1 and confidence > 0.7: # 낙상(1)
                        status_text = f"FALL DETECTED ({confidence*100:.1f}%)"
                        color = (0, 0, 255) # Red
                        
                        if not is_fall_state:
                            is_fall_state = True
                            fall_start_time = time.time()
                            print(f"⚠️ 낙상 감지! - {status_text}")
                            
                            # 즉시 캡처 저장
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = os.path.join(utils.ALERT_DIR, f"FALL_LSTM_{timestamp}.jpg")
                            cv2.imwrite(save_path, frame)
                            
                    else:
                        status_text = "Normal"
                        is_fall_state = False
                
                # 시각화
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, status_text, (int(bbox[0]), int(bbox[1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if not detected:
            # 사람이 없으면 버퍼 초기화 (다른 사람이 들어오면 섞이므로)
            if len(frame_buffer) > 0: frame_buffer.clear()
            prev_head_y = None
            
    cap.release()
    print("시스템 종료.")

if __name__ == '__main__':
    main()