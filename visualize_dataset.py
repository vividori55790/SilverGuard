import cv2
import torch
import joblib
import numpy as np
import os
import math
from collections import deque
from ultralytics import YOLO
import utils
from models import FallLSTM

# ==========================================
# [설정] 검증할 데이터셋 경로
# ==========================================
# 대괄호([])가 포함된 경로도 문제없이 읽을 수 있도록 수정했습니다.
TARGET_FOLDER = r"C:\Users\vivid\Documents\Git Project\PythonUtil\data\urfall\fall"

# [설정] 모델 파라미터
SEQUENCE_LENGTH = 30
INPUT_SIZE = 54
HIDDEN_SIZE = 64
NUM_LAYERS = 2
SKIP_FRAMES = 3        # 학습 데이터 검증이므로 스킵 없이(1) 설정하여 30fps로 확인
YOLO_IMG_SIZE = 320

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def run_visualization():
    print(f"🚀 학습 데이터 시각화 모드 시작")
    print(f"📂 대상 폴더: {TARGET_FOLDER}")
    
    # -----------------------------------------------------------
    # [수정] glob 대신 os.listdir을 사용하여 특수문자([]) 문제 해결
    # -----------------------------------------------------------
    video_files = []
    
    if not os.path.exists(TARGET_FOLDER):
        print(f"❌ 폴더를 찾을 수 없습니다: {TARGET_FOLDER}")
        print("   -> 경로에 오타가 있는지 확인해주세요.")
        return

    # 폴더 내의 파일을 직접 하나씩 확인하여 리스트에 담습니다.
    for f in os.listdir(TARGET_FOLDER):
        # 대소문자 구분 없이 mp4, avi 파일 찾기
        if f.lower().endswith(('.mp4', '.avi')):
            video_files.append(os.path.join(TARGET_FOLDER, f))
    
    video_files.sort()
    
    if not video_files:
        print("❌ 해당 폴더에 동영상 파일(.mp4, .avi)이 없습니다.")
        return

    print(f"   - 총 {len(video_files)}개의 영상을 발견했습니다.")

    # 2. 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - 실행 장치: {device}")

    # YOLO 모델
    if not os.path.exists(utils.YOLO_MODEL_PATH):
        print(f"❌ YOLO 모델 없음: {utils.YOLO_MODEL_PATH}")
        return
    yolo_model = YOLO(utils.YOLO_MODEL_PATH)
    
    # LSTM 모델
    lstm_path = os.path.join(utils.MODEL_DIR, 'fall_lstm.pth')
    if not os.path.exists(lstm_path):
        print(f"❌ LSTM 모델 없음: {lstm_path}")
        return
        
    lstm_model = FallLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()
    
    # Scaler
    scaler_path = os.path.join(utils.MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler 없음: {scaler_path}")
        return
    scaler = joblib.load(scaler_path)

    # 3. 영상별 반복 처리
    for video_idx, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        print(f"▶️ 재생 중 [{video_idx+1}/{len(video_files)}]: {filename}")
        
        cap = cv2.VideoCapture(video_path)
        
        # 상태 변수 초기화 (새로운 영상 시작 시 리셋)
        frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        prev_head_y = None
        prev_angle = None
        frame_count = 0
        
        # 시각화용 변수
        last_status = "Initializing"
        last_color = (200, 200, 200)

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 영상 끝나면 다음 영상으로

            frame_count += 1
            
            # (옵션) 학습 데이터가 오른쪽 절반만 사용했다면 여기서도 자르기
            # utils.CROP_RIGHT_HALF가 True인 경우에만 작동
            if utils.CROP_RIGHT_HALF:
                 h, w, _ = frame.shape
                 frame = frame[:, w//2:]

            if frame_count % SKIP_FRAMES != 0:
                continue

            # --- YOLO 추론 ---
            results = yolo_model(frame, verbose=False, imgsz=YOLO_IMG_SIZE)
            
            # --- 데이터 추출 및 LSTM ---
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                kpts = results[0].keypoints.xyn[0].cpu().numpy()
                kpts_xy = results[0].keypoints.xy[0].cpu().numpy()
                confs = results[0].keypoints.conf[0].cpu().numpy()
                bbox = results[0].boxes.xyxy[0].cpu().numpy()

                # 스켈레톤 그리기 (시각화)
                for idx, (x, y) in enumerate(kpts_xy):
                    if confs[idx] > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

                if len(kpts) == 17:
                    # ---------------------------------------------------------
                    # [Feature Engineering] - 오작동 방지 로직 적용됨
                    # ---------------------------------------------------------
                    # 1. 엉덩이(11,12) 신뢰도가 낮으면 각도를 0으로 처리 (누운 것으로 오인 방지)
                    if confs[11] < 0.5 or confs[12] < 0.5:
                        current_angle = 0 
                    else:
                        shoulder_mid = (kpts[5] + kpts[6]) / 2
                        hip_mid = (kpts[11] + kpts[12]) / 2
                        current_angle = calculate_angle(shoulder_mid, hip_mid)

                    head_y = kpts[0][1]
                    
                    # 2. 속도 계산
                    if prev_head_y is not None:
                        head_velocity = (head_y - prev_head_y) * 30 / SKIP_FRAMES
                        angle_velocity = (current_angle - prev_angle) * 30 / SKIP_FRAMES
                    else:
                        head_velocity = 0
                        angle_velocity = 0
                    
                    prev_head_y = head_y
                    prev_angle = current_angle
                    
                    # 데이터 구성
                    row = []
                    for i in range(17): row.extend([kpts[i][0], kpts[i][1], confs[i]])
                    row.extend([head_velocity, angle_velocity, current_angle])
                    frame_buffer.append(row)

                    # 3. LSTM 추론
                    confidence = 0
                    pred_cls = 0
                    
                    if len(frame_buffer) == SEQUENCE_LENGTH:
                        input_seq = np.array(frame_buffer).reshape(-1, INPUT_SIZE)
                        input_seq_scaled = scaler.transform(input_seq)
                        input_tensor = torch.tensor(input_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output = lstm_model(input_tensor)
                            prob = torch.softmax(output, dim=1)
                            pred_cls = torch.argmax(prob, dim=1).item()
                            confidence = prob[0][pred_cls].item()

                        # 결과 표시
                        if pred_cls == 1 and confidence > 0.5:
                            last_status = f"FALL DETECTED ({confidence*100:.0f}%)"
                            last_color = (0, 0, 255) # 빨간색
                        else:
                            last_status = f"Normal ({confidence*100:.0f}%)" if pred_cls == 0 else "Warn"
                            last_color = (0, 255, 0) # 초록색

                        # 박스 및 텍스트 그리기
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), last_color, 2)
                        
                        # 디버깅 정보 출력
                        info = f"Angle: {current_angle:.1f} Vel: {head_velocity:.1f}"
                        cv2.putText(frame, last_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)
                        cv2.putText(frame, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"File: {filename}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # [수정] 화면이 너무 커서 잘리는 문제 해결
            # 창 이름을 먼저 만들고, 크기 조절이 가능한 모드(WINDOW_NORMAL)로 설정
            cv2.namedWindow("Dataset Visualization", cv2.WINDOW_NORMAL)
            
            # (선택 사항) 처음에 뜰 때 적당한 크기(예: 960x540)로 강제 조절
            # 원하시면 이 줄의 주석을 풀고 숫자를 바꾸세요.
            # cv2.resizeWindow("Dataset Visualization", 960, 540)

            # 화면 출력
            cv2.imshow("Dataset Visualization", frame)
            
            # [키 입력 처리]
            # q: 종료, Space: 일시정지
            key = cv2.waitKey(33) # 약 30FPS 속도
            if key & 0xFF == ord('q'):
                print("중단됨")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == 32: # Spacebar를 누르면 일시정지
                print("⏸ 일시정지 (아무 키나 누르면 재개)")
                cv2.waitKey(0) 

        cap.release()
    
    cv2.destroyAllWindows()
    print("✅ 모든 영상 재생 완료")

if __name__ == '__main__':
    run_visualization()