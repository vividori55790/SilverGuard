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
from offline_mode import is_internet_available, activate_offline_safety_mode, sync_unsent_data

# ==========================================
# [설정] 모델 파라미터
# ==========================================
SEQUENCE_LENGTH = 30
INPUT_SIZE = 54
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# [최적화 설정]
# 테스트를 위해 프레임 스킵을 줄여서 점이 더 자주 보이게 합니다.
SKIP_FRAMES = 2       
YOLO_IMG_SIZE = 640   

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def main():
    print("🚀 SilverGuard 시스템을 시작합니다...")
    utils.ensure_dirs()

    # [1] 시작 시 미전송 데이터 확인
    print("🔍 미전송 알림 확인 중...")
    sync_unsent_data()

    # [2] 모델 로딩
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📦 모델 로드 중... (장치: {device})")
    
    yolo_model = YOLO(utils.YOLO_MODEL_PATH)
    
    lstm_path = os.path.join(utils.MODEL_DIR, 'fall_lstm.pth')
    lstm_model = FallLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()
    
    scaler = joblib.load(os.path.join(utils.MODEL_DIR, 'scaler.pkl'))
    print("✅ 모든 모델 로드 완료! (움직임 감지 기능을 껐습니다)")

    # [3] 카메라 연결
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ 카메라를 찾을 수 없어 테스트 영상을 불러옵니다.")
        test_video_path = os.path.join(utils.VIDEO_DIR, utils.TEST_VIDEO_NAME)
        cap = cv2.VideoCapture(test_video_path)

    # 변수 초기화
    # backSub = cv2.createBackgroundSubtractorMOG2(...) # 움직임 감지 끔
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prev_head_y, prev_angle = None, None
    is_fall_state = False
    last_alert_time = 0
    ALERT_COOLDOWN = 60 
    frame_count = 0
    
    # 마지막으로 감지된 정보를 저장하는 변수들
    last_bbox = None
    last_status = "Initializing"
    last_color = (200, 200, 200)
    last_kpts_xy = []   # 마지막 관절 위치 저장
    last_confs = []     # 마지막 정확도 저장

    # 스켈레톤 연결 정보
    skeleton_connections = [
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
        (5, 6), (11, 12), (5, 11), (6, 12)
    ]

    print("🟢 모니터링 시작! (종료하려면 화면에서 'q'를 누르세요)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            
            frame_count += 1
            if utils.CROP_RIGHT_HALF:
                frame = frame[:, frame.shape[1]//2:]

            # [Motion Trigger 제거] -> 항상 AI가 작동하도록 수정함
            
            # [2] Frame Skipping (연산량 조절)
            # 추론을 건너뛰는 프레임에서도 '마지막으로 찾은 사람'을 그려줍니다.
            if frame_count % SKIP_FRAMES != 0:
                if last_bbox is not None:
                    # 박스 그리기
                    cv2.rectangle(frame, (int(last_bbox[0]), int(last_bbox[1])), (int(last_bbox[2]), int(last_bbox[3])), last_color, 2)
                    cv2.putText(frame, last_status, (int(last_bbox[0]), int(last_bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)
                    
                    # (중요) 스켈레톤 점도 유지해서 그려주기
                    for idx, (x, y) in enumerate(last_kpts_xy):
                        if idx < len(last_confs) and last_confs[idx] > 0.5:
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                
                # 미전송 데이터 체크 (매번 하면 느리니 여기서 체크)
                if frame_count % 100 == 0:
                    sync_unsent_data()

                cv2.imshow("SilverGuard Local Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # [3] YOLO & LSTM 추론 (실제 계산하는 프레임)
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
                    # 정보를 변수에 저장 (다음 프레임에서도 그리려고)
                    last_bbox = bbox
                    last_kpts_xy = kpts_xy
                    last_confs = confs

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
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                save_path = os.path.join(utils.ALERT_DIR, f"FALL_LOCAL_{timestamp}.jpg")
                                cv2.imwrite(save_path, frame)
                                
                                voice_res = run_voice_emergency_check(save_path)
                                
                                if is_internet_available():
                                    if time.time() - last_alert_time > ALERT_COOLDOWN:
                                        utils.send_telegram_alert(save_path, f"🚨 낙상 발생! (결과: {voice_res})")
                                        last_alert_time = time.time()
                                else:
                                    if time.time() - last_alert_time > ALERT_COOLDOWN:
                                        print("🌐 인터넷 연결 없음! 오프라인 모드 작동.")
                                        activate_offline_safety_mode(save_path, voice_res)
                                        last_alert_time = time.time()
                        else:
                            is_fall_state = False
            
            # (추론 프레임에서도 그리기)
            if detected:
                # 스켈레톤 시각화
                for idx, (x, y) in enumerate(last_kpts_xy):
                    if last_confs[idx] > 0.5: cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                for p1, p2 in skeleton_connections:
                    if last_confs[p1] > 0.5 and last_confs[p2] > 0.5:
                        cv2.line(frame, (int(last_kpts_xy[p1][0]), int(last_kpts_xy[p1][1])), 
                                        (int(last_kpts_xy[p2][0]), int(last_kpts_xy[p2][1])), (0, 255, 0), 2)
                
                cv2.rectangle(frame, (int(last_bbox[0]), int(last_bbox[1])), (int(last_bbox[2]), int(last_bbox[3])), last_color, 2)
                cv2.putText(frame, last_status, (int(last_bbox[0]), int(last_bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)
            else:
                if len(frame_buffer) > 0: frame_buffer.clear()
                prev_head_y, last_bbox = None, None
	# [추가됨] 약 1초(30프레임)마다 시스템이 살아있다는 신호를 보냅니다.
            if frame_count % 30 == 0:
                utils.update_heartbeat()            

            if frame_count % 100 == 0:
                sync_unsent_data()
                
            cv2.imshow("SilverGuard Local Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 시스템을 종료합니다.")

if __name__ == '__main__':
    main()