import os
import sys
import math
import torch
import joblib
import numpy as np
import cv2
from collections import deque
from ultralytics import YOLO

# Parent directory import support
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils
from models import FallLSTM

class FallDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📦 모델 로드 중... (장치: {self.device})")

        # Constants
        self.SEQUENCE_LENGTH = 30
        self.INPUT_SIZE = 54
        self.HIDDEN_SIZE = 64
        self.NUM_LAYERS = 2
        self.YOLO_IMG_SIZE = 640
        
        # Load models
        self.yolo_model = YOLO(utils.YOLO_MODEL_PATH)
        
        lstm_path = os.path.join(utils.MODEL_DIR, 'fall_lstm.pth')
        self.lstm_model = FallLSTM(self.INPUT_SIZE, self.HIDDEN_SIZE, self.NUM_LAYERS).to(self.device)
        if os.path.exists(lstm_path):
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
        else:
            print(f"⚠️ LSTM Model not found at {lstm_path}")
            
        self.lstm_model.eval()
        
        scaler_path = os.path.join(utils.MODEL_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None
            print(f"⚠️ Scaler not found at {scaler_path}")

        # State buffers
        self.frame_buffer = deque(maxlen=self.SEQUENCE_LENGTH)
        self.prev_head_y = None
        self.prev_angle = None
        
        # Latest detection state (for visualization)
        self.last_bbox = None
        self.last_kpts_xy = []
        self.last_confs = []
        self.detected = False

    def calculate_angle(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return abs(math.degrees(math.atan2(dx, dy)))

    def process_frame(self, frame, skip_frames=2):
        """
        Input: Frame (image)
        Output: (prob_class, confidence, bbox, kpts, confs, is_detected)
        """
        results = self.yolo_model(frame, verbose=False, imgsz=self.YOLO_IMG_SIZE)
        self.detected = False
        
        pred_cls = 0
        conf = 0.0

        for r in results:
            if r.keypoints is None or len(r.keypoints) == 0: continue
            
            kpts_xy = r.keypoints.xy[0].cpu().numpy()
            kpts = r.keypoints.xyn[0].cpu().numpy()
            confs = r.keypoints.conf[0].cpu().numpy()
            bbox = r.boxes.xyxy[0].cpu().numpy()
            
            if len(kpts) == 17:
                self.detected = True
                self.last_bbox = bbox
                self.last_kpts_xy = kpts_xy
                self.last_confs = confs

                # Feature Extraction
                head_y = kpts[0][1]
                shoulder_mid = (kpts[5] + kpts[6]) / 2
                hip_mid = (kpts[11] + kpts[12]) / 2
                
                current_angle = self.calculate_angle(shoulder_mid, hip_mid)
                
                # Velocity calc
                head_v = (head_y - self.prev_head_y) * 30 / skip_frames if self.prev_head_y else 0
                angle_v = (current_angle - self.prev_angle) * 30 / skip_frames if self.prev_angle else 0
                
                self.prev_head_y = head_y
                self.prev_angle = current_angle

                # Buffer append
                row = []
                for i in range(17): 
                    row.extend([kpts[i][0], kpts[i][1], confs[i]])
                row.extend([head_v, angle_v, current_angle])
                self.frame_buffer.append(row)
                
                # LSTM Inference
                if len(self.frame_buffer) == self.SEQUENCE_LENGTH and self.scaler:
                    input_seq = self.scaler.transform(np.array(self.frame_buffer).reshape(-1, self.INPUT_SIZE))
                    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        prob = torch.softmax(self.lstm_model(input_tensor), dim=1)
                        pred_cls = torch.argmax(prob, dim=1).item()
                        conf = prob[0][pred_cls].item()
                        
                return pred_cls, conf, bbox, kpts_xy, confs, True
                
        # If no detection
        return 0, 0.0, None, [], [], False

    def reset_buffer(self):
        self.frame_buffer.clear()
        self.prev_head_y = None
        self.prev_angle = None
