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
# [설정] 모델 파라미터 (학습 때와 동일해야 함)
# ==========================================
SEQUENCE_LENGTH = 30  # 윈도우 크기 (프레임 수)
INPUT_SIZE = 54       # 입력 피처 개수 (17*3 + 3)
HIDDEN_SIZE = 64
NUM_LAYERS = 2

def calculate_angle(p1, p2):
    """ 두 점 사이 각도 계산 """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def main():
    print("🚀 SilverGuard: LSTM 기반 실시간 낙상 감지 시스템 시작")
    utils.ensure_dirs()

    # 1. 하드웨어 가속 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - 실행 장치: {device}")

    # 2. 모델 로드
    # (A) YOLO 모델
    print("   - YOLOv8-Pose 모델 로딩 중...")
    try:
        yolo_model = YOLO(utils.YOLO_MODEL_PATH)
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        return
    
    # (B) LSTM 모델
    print("   - LSTM 모델 로딩 중...")
    lstm_path = os.path.join(utils.MODEL_DIR, 'fall_lstm.pth')
    if not os.path.exists(lstm_path):
        print(f"❌ 오류: 학습된 LSTM 모델이 없습니다. ({lstm_path})")
        print("   -> train_lstm.py를 먼저 실행하여 모델을 생성하세요.")
        return

    lstm_model = FallLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval() # 평가 모드 설정 (Dropout 비활성화 등)

    # (C) 스케일러 로드
    scaler_path = os.path.join(utils.MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        print("❌ 오류: 스케일러 파일이 없습니다. train_lstm.py 실행 시 생성됩니다.")
        return
    scaler = joblib.load(scaler_path)

    # 3. 영상 소스 설정
    test_video_path = os.path.join(utils.VIDEO_DIR, utils.TEST_VIDEO_NAME)
    # 테스트 영상 파일이 존재하면 파일 사용, 아니면 웹캠(0) 사용
    if os.path.exists(test_video_path):
        video_source = test_video_path
        print(f"   - 입력 소스: 테스트 영상 파일 ({utils.TEST_VIDEO_NAME})")
    else:
        video_source = 0
        print(f"   - 입력 소스: 실시간 웹캠 (Camera 0)")
        
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("❌ 영상을 열 수 없습니다.")
        return

    # [디버깅 추가] 결과 영상 저장을 위한 설정
    # Docker 환경에서는 cv2.imshow가 안되므로 파일로 저장하여 확인합니다.
    output_path = os.path.join(utils.DATA_DIR, 'debug_output.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None # 첫 프레임 읽은 후 초기화

    # 4. 실시간 데이터 처리 변수 초기화
    # 최근 30프레임의 데이터를 저장할 큐 (FIFO 구조, 꽉 차면 오래된 것 자동 삭제)
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # 속도 계산을 위한 이전 프레임 값
    prev_head_y = None
    prev_angle = None
    
    # 낙상 상태 플래그
    is_fall_state = False
    
    print(f"✅ 감시를 시작합니다. (결과 저장 중: {output_path})")
    print("   중단하려면 터미널에서 Ctrl+C를 누르세요.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("영상 종료. (루프 종료)")
                break

            # [UR Fall 테스트용] 오른쪽 절반 자르기
            # 실제 현장 배포 시에는 utils.py에서 CROP_RIGHT_HALF를 False로 변경해야 함
            if utils.CROP_RIGHT_HALF:
                h, w, _ = frame.shape
                frame = frame[:, w//2:]

            # VideoWriter 초기화 (프레임 크기에 맞춰 설정)
            if out is None:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

            # YOLO 추론
            results = yolo_model(frame, verbose=False)
            
            # 사람이 감지되었는지 확인
            detected = False
            
            for r in results:
                if r.keypoints is None or len(r.keypoints) == 0: continue

                # 가장 크게 잡힌 사람 1명만 추적 (ID Tracking 생략)
                kpts = r.keypoints.xyn[0].cpu().numpy() # (17, 2)
                confs = r.keypoints.conf[0].cpu().numpy() # (17,)
                bbox = r.boxes.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                
                if len(kpts) == 17:
                    detected = True
                    
                    # --- Feature Engineering (전처리 코드와 동일 로직) ---
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
                    # (1) Keypoints (x, y, conf)
                    for i in range(17):
                        row.extend([kpts[i][0], kpts[i][1], confs[i]])
                    # (2) Derived Features
                    row.extend([head_velocity, angle_velocity, current_angle])
                    
                    # 버퍼에 추가 (가득 차면 가장 오래된 것이 자동 삭제됨)
                    frame_buffer.append(row)
                    
                    # --- LSTM 추론 (데이터가 30프레임 모였을 때만 수행) ---
                    status_text = "Analyzing..."
                    color = (0, 255, 0) # Green (Normal)
                    
                    if len(frame_buffer) == SEQUENCE_LENGTH:
                        # (1, 30, 54) 형태로 변환
                        input_seq = np.array(frame_buffer) # (30, 54)
                        
                        # 스케일링 적용 (학습 때와 동일하게 2차원으로 펴서 변환)
                        input_seq_2d = input_seq.reshape(-1, INPUT_SIZE)
                        input_seq_scaled = scaler.transform(input_seq_2d)
                        
                        # 텐서 변환 및 차원 추가 (Batch Size = 1)
                        input_tensor = torch.tensor(input_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        # 예측 수행
                        with torch.no_grad():
                            output = lstm_model(input_tensor)
                            prob = torch.softmax(output, dim=1)
                            
                            # Class 0: Normal, Class 1: Fall
                            pred_cls = torch.argmax(prob, dim=1).item()
                            confidence = prob[0][pred_cls].item()
                        
                        # 결과 처리 및 시각화
                        if pred_cls == 1 and confidence > 0.7: # 낙상 확률 70% 이상
                            status_text = f"FALL DETECTED ({confidence*100:.1f}%)"
                            color = (0, 0, 255) # Red (Fall)
                            
                            if not is_fall_state:
                                is_fall_state = True
                                print(f"⚠️ [{datetime.datetime.now().strftime('%H:%M:%S')}] 낙상 감지됨! ({confidence*100:.1f}%)")
                                
                                # 증거 이미지 저장
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                save_path = os.path.join(utils.ALERT_DIR, f"FALL_LSTM_{timestamp}.jpg")
                                cv2.imwrite(save_path, frame)
                                print(f"   -> 이미지 저장: {save_path}")
                                
                        else:
                            status_text = "Normal"
                            is_fall_state = False
                    
                    # 화면에 박스와 텍스트 그리기
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.putText(frame, status_text, (int(bbox[0]), int(bbox[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # 사람이 화면에서 사라지면 버퍼 초기화 (데이터 섞임 방지)
            if not detected:
                if len(frame_buffer) > 0: frame_buffer.clear()
                prev_head_y = None
                
            # [디버깅] 영상 파일에 프레임 저장
            if out is not None:
                out.write(frame)

    except KeyboardInterrupt:
        print("사용자 중단 (Ctrl+C)")
    
    finally:
        cap.release()
        if out is not None:
            out.release()
        print(f"시스템 종료. 결과 영상이 저장되었습니다: {output_path}")

if __name__ == '__main__':
    main()