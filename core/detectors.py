import torch
import numpy as np
import joblib
import os
import math
from collections import deque
from ultralytics import YOLO
import utils
from models import FallLSTM

class FallDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   - FallDetector Device: {self.device}")

        # 1. Load YOLO Model
        # 모델이 없으면 다운로드, 있으면 로드
        if not os.path.exists(utils.YOLO_MODEL_PATH):
            print(f"⚠️ YOLO model not found at {utils.YOLO_MODEL_PATH}, downloading/loading default.")
            self.yolo_model = YOLO('yolo11n-pose.pt') 
        else:
            self.yolo_model = YOLO(utils.YOLO_MODEL_PATH)

        # 2. Load LSTM Model
        # 54 features (17*3 + 3) -> 64 hidden -> 2 classes
        self.lstm_model = FallLSTM(input_size=54, hidden_size=64, num_layers=2).to(self.device)
        lstm_path = os.path.join(utils.MODEL_DIR, 'fall_lstm.pth')
        
        self.lstm_loaded = False
        if os.path.exists(lstm_path):
            try:
                self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
                self.lstm_model.eval()
                self.lstm_loaded = True
                print("✅ LSTM model loaded successfully.")
            except Exception as e:
                print(f"❌ Failed to load LSTM model: {e}")
        else:
            print(f"⚠️ LSTM model file not found at {lstm_path}. Running in YOLO-only mode (Skeleton only).")

        # 3. Load Scaler
        # 학습 데이터 정규화에 사용된 스케일러 로드
        scaler_path = os.path.join(utils.MODEL_DIR, 'scaler.pkl')
        self.scaler = None
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print("✅ Scaler loaded successfully.")
            except Exception as e:
                print(f"❌ Failed to load Scaler: {e}")
        else:
            print("⚠️ Scaler not found. LSTM inference might be incorrect.")

        # Runtime variables
        self.buffer = deque(maxlen=30) # Sequence length 30
        self.prev_head_y = None
        self.prev_angle = None
        
        # Public state for Engine (엔진에서 참조할 변수들)
        self.detected = False
        self.last_kpts_xy = None
        self.last_confs = None
        self.last_bbox = None

    def calculate_angle(self, p1, p2):
        """두 점 사이의 각도 계산 (수직=0도)"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return abs(math.degrees(math.atan2(dx, dy)))

    def process(self, frame):
        """
        프레임 하나를 받아 낙상 여부를 판단합니다.
        Returns:
            pred_cls (int): 0 (Normal) or 1 (Fall)
            conf (float): Confidence score (0.0 ~ 1.0)
        """
        # YOLO Inference (속도를 위해 imgsz=320 사용, 필요시 640으로 변경)
        results = self.yolo_model(frame, verbose=False, imgsz=320)
        
        # 상태 초기화
        self.detected = False
        self.last_kpts_xy = None
        self.last_confs = None
        self.last_bbox = None
        
        if not results:
            return 0, 0.0

        r = results[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return 0, 0.0

        # 가장 신뢰도 높은(첫번째) 사람 데이터 추출
        # xyn: normalized coordinates (0~1), xy: pixel coordinates
        kpts_norm = r.keypoints.xyn[0].cpu().numpy() # (17, 2)
        kpts_pixel = r.keypoints.xy[0].cpu().numpy() # (17, 2)
        confs = r.keypoints.conf[0].cpu().numpy()    # (17,)
        bbox = r.boxes.xyxy[0].cpu().numpy()         # (4,)

        # 엔진이 화면에 그릴 수 있도록 상태 저장
        self.detected = True
        self.last_kpts_xy = kpts_pixel
        self.last_confs = confs
        self.last_bbox = bbox

        # LSTM 모델이나 스케일러가 없으면 감지 불가 -> Normal 리턴
        if not self.lstm_loaded or self.scaler is None:
            return 0, 0.0

        # Feature Extraction (LSTM 입력 데이터 생성)
        if len(kpts_norm) != 17:
            return 0, 0.0

        # 1. Angle Calculation
        # Hip indices: 11 (left), 12 (right). Shoulder: 5, 6.
        if confs[11] < 0.5 or confs[12] < 0.5:
            current_angle = 0 
        else:
            shoulder_mid = (kpts_norm[5] + kpts_norm[6]) / 2
            hip_mid = (kpts_norm[11] + kpts_norm[12]) / 2
            current_angle = self.calculate_angle(shoulder_mid, hip_mid)

        # 2. Velocity Calculation
        head_y = kpts_norm[0][1] # Nose Y
        
        head_velocity = 0
        angle_velocity = 0
        
        if self.prev_head_y is not None:
             # 학습 때 사용한 스케일링에 맞춰 속도 계산 (30fps 기준 보정)
             head_velocity = (head_y - self.prev_head_y) * 30 
             angle_velocity = (current_angle - self.prev_angle) * 30

        self.prev_head_y = head_y
        self.prev_angle = current_angle

        # 3. Construct Feature Row (54 features)
        row = []
        for i in range(17):
            # x, y, conf 순서로 추가
            row.extend([kpts_norm[i][0], kpts_norm[i][1], confs[i]])
        # 추가 피처 3개
        row.extend([head_velocity, angle_velocity, current_angle])
        
        self.buffer.append(row)

        # 버퍼가 덜 찼으면 아직 판단 불가
        if len(self.buffer) < 30:
            return 0, 0.0

        # LSTM Inference
        input_seq = np.array(self.buffer).reshape(-1, 54) # (30, 54)
        
        try:
            # Scaler 적용
            input_seq_scaled = self.scaler.transform(input_seq)
        except Exception as e:
            # 스케일러 에러 시 안전하게 패스
            return 0, 0.0

        # To Tensor: (Batch=1, Seq=30, Feature=54)
        input_tensor = torch.tensor(input_seq_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.lstm_model(input_tensor) # Output: (1, 2)
            prob = torch.softmax(output, dim=1)
            pred_cls = torch.argmax(prob, dim=1).item()
            conf = prob[0][pred_cls].item()

        return pred_cls, conf