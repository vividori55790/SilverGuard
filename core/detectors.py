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
from stgcn import STGCN

class FallDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📦 모델 로드 중... (장치: {self.device})")

        # Constants
        self.SEQUENCE_LENGTH = 30
        self.YOLO_IMG_SIZE = 640
        
        # Load models
        self.yolo_model = YOLO(utils.YOLO_MODEL_PATH)
        
        stgcn_path = os.path.join(utils.MODEL_DIR, 'stgcn_fall.pth')
        self.stgcn_model = STGCN(in_channels=3, num_class=2).to(self.device)
        
        if os.path.exists(stgcn_path):
            self.stgcn_model.load_state_dict(torch.load(stgcn_path, map_location=self.device))
            print(f"✅ ST-GCN 모델 로드 완료: {stgcn_path}")
        else:
            print(f"⚠️ ST-GCN Model not found at {stgcn_path}")
            
        self.stgcn_model.eval()
        
        # LSTM/Scaler legacy removed
        self.scaler = None

        # State buffers
        self.frame_buffer = deque(maxlen=self.SEQUENCE_LENGTH)
        self.prev_head_y = None
        self.prev_angle = None
        
        # Enhanced Detection Buffers
        self.prob_buffer = deque(maxlen=5) # Smooth probabilities over 5 detections
        self.consecutive_fall_frames = 0
        self.FALL_CONFIDENCE_THRESHOLD = 0.6 # Base threshold
        self.High_CONFIDENCE_THRESHOLD = 0.85 # Threshold to bypass heuristics

        
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
                return self.process_keypoints(kpts, confs, bbox, kpts_xy, skip_frames)
                
        # If no detection
        return 0, 0.0, None, [], [], False

    def process_keypoints(self, kpts, confs, bbox, kpts_xy, skip_frames):
        """
        Process pre-extracted keypoints (17, 2) and confidences (17,)
        """
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

        # Buffer append: ST-GCN expects (x, y, c) for 17 keypoints
        frame_data = [] # (17, 3)
        for i in range(17):
            # x, y normalized are in kpts[i]
            frame_data.append([kpts[i][0], kpts[i][1], confs[i]])
        
        self.frame_buffer.append(frame_data)
        
        pred_cls = 0
        conf = 0.0
        
        # ST-GCN Inference
        if len(self.frame_buffer) == self.SEQUENCE_LENGTH:
            # buffer: List of (17, 3) -> (30, 17, 3)
            # Model expects: (N, C, T, V, M) -> (1, 3, 30, 17, 1)
            
            data_numpy = np.array(self.frame_buffer) # (30, 17, 3)
            data_numpy = data_numpy.transpose(2, 0, 1) # (3, 30, 17) -> (C, T, V)
            
            input_tensor = torch.tensor(data_numpy, dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(-1).to(self.device) # (1, 3, 30, 17, 1)
            
            with torch.no_grad():
                logits = self.stgcn_model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                
                raw_fall_prob = probs[0][1].item()
                self.prob_buffer.append(raw_fall_prob)
                
                # Temporal Smoothing
                smoothed_prob = np.mean(self.prob_buffer)
                
                # Heuristic Checks
                is_pose_suspicious = current_angle > 40 
                is_high_velocity = abs(head_v) > 0.05 
                
                # Decision Logic
                is_fall_detected = False
                
                if smoothed_prob > self.High_CONFIDENCE_THRESHOLD:
                    is_fall_detected = True
                elif smoothed_prob > self.FALL_CONFIDENCE_THRESHOLD:
                    if is_pose_suspicious or is_high_velocity:
                        is_fall_detected = True
                
                if is_fall_detected:
                    pred_cls = 1
                    conf = smoothed_prob
                    self.consecutive_fall_frames += 1
                else:
                    pred_cls = 0
                    conf = 1.0 - smoothed_prob
                    self.consecutive_fall_frames = max(0, self.consecutive_fall_frames - 1)
        
        return pred_cls, conf, bbox, kpts_xy, confs, True
                
        # If no detection
        return 0, 0.0, None, [], [], False

    def reset_buffer(self):
        self.frame_buffer.clear()
        self.prev_head_y = None
        self.prev_angle = None
