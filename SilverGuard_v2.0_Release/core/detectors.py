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
        self.last_timestamp = None # For robust time-based calculation
        
        # Enhanced Detection Buffers
        self.prob_buffer = deque(maxlen=5) # Smooth probabilities over 5 detections
        self.consecutive_fall_frames = 0
        
        # Dynamic Settings (Defaults)
        self.FALL_CONFIDENCE_THRESHOLD = 0.65 
        self.strictness_angle = 50 
        self.strictness_velocity = 0.08
        self.strictness_level = "Medium"

        
        # Latest detection state (for visualization)
        self.last_bbox = None
        self.last_kpts_xy = []
        self.last_confs = []
        self.detected = False

    def set_sensitivity(self, conf_threshold, strictness_level="Medium"):
        """
        Update detection sensitivity dynamically.
        """
        self.FALL_CONFIDENCE_THRESHOLD = float(conf_threshold)
        self.strictness_level = strictness_level
        
        # Adjust heuristics based on strictness
        if strictness_level == "Low":
            # Loose: Easier to trigger fall (Good for catching everything)
            self.strictness_angle = 35 
            self.strictness_velocity = 0.05
        elif strictness_level == "High":
            # Strict: Harder to trigger (Good for avoiding false alarms)
            self.strictness_angle = 60
            self.strictness_velocity = 0.12
        else:
            # Medium (Default)
            self.strictness_angle = 50
            self.strictness_velocity = 0.08

    def calculate_angle(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return abs(math.degrees(math.atan2(dx, dy)))

    def process(self, frame, timestamp=None):
        return self.process_frame(frame, timestamp=timestamp)

    def process_frame(self, frame, timestamp=None, skip_frames=2):
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
                return self.process_keypoints(kpts, confs, bbox, kpts_xy, timestamp, skip_frames)
                
        # If no detection
        return 0, 0.0, None, [], [], False, ""

    def process_keypoints(self, kpts, confs, bbox, kpts_xy, timestamp=None, skip_frames=2):
        """
        Process pre-extracted keypoints (17, 2) and confidences (17,)
        timestamp: float (seconds), usually time.time(). If None, assumes constant FPS.
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
        
        # --- Robust Velocity Calculation ---
        # Calculate dt (time difference)
        dt = 0.0
        if timestamp is not None and self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
        else:
            # Fallback for first frame or if no timestamp provided
            # Assume 30 FPS * skip_frames
            dt = (1.0 / 30.0) * skip_frames
        
        self.last_timestamp = timestamp if timestamp is not None else None

        # Safety: Handle dropped frames / Signal Lag
        # If dt is too large (e.g. > 0.5s), it means we lost signal or video jumped.
        # Calculating velocity across this gap creates massive spikes (Teleportation).
        # We must ignore velocity for this frame.
        is_signal_loss = dt > 0.5
        
        head_v = 0.0
        angle_v = 0.0
        
        if self.prev_head_y is not None and not is_signal_loss and dt > 0:
            # Velocity = Change / Time (Units/sec)
            # Original logic was: (diff) * 30/skip ~ (diff) / (skip/30) ~ diff / dt
            # So we can simply use diff / dt to stay consistent with physics.
            head_v = (head_y - self.prev_head_y) / dt
            # However, previous threshold (0.08) was loosely based on per-frame changes scaled.
            # Let's adjust scale. 
            # If prev logic was `diff * (30/2)` -> `diff * 15`.
            # New logic is `diff / dt`. At 30fps/skip2, `dt = 0.066`. `1/0.066 = 15`.
            # So `diff / dt` is roughly equivalent to old `diff * 15`. 
            # The units are compatible.
            
            angle_v = (current_angle - self.prev_angle) / dt
        
        if is_signal_loss:
            # If signal loss, we reset the buffer because the sequence is broken.
            # ST-GCN needs continuous motion.
            self.frame_buffer.clear()
            self.prob_buffer.clear()
            # We don't return immediately, we treat this as a "Re-init" frame (velocity 0)
        
        self.prev_head_y = head_y
        self.prev_angle = current_angle

        # Buffer append: ST-GCN expects (x, y, c) for 17 keypoints
        frame_data = [] # (17, 3)
        for i in range(17):
            # x, y normalized are in kpts[i]
            frame_data.append([kpts[i][0], kpts[i][1], confs[i]])
        
        self.frame_buffer.append(frame_data)
        
        # --- Robustness 1: Check Critical Keypoint Visibility ---
        # If shoulders or hips are not visible, Angle calculation is garbage.
        # Indices: 5,6 (Shoulders), 11,12 (Hips)
        # We require at least one shoulder and one hip to be confident, 
        # or just average confidence of torso to be decent.
        torso_confs = [confs[5], confs[6], confs[11], confs[12]]
        avg_torso_conf = sum(torso_confs) / 4.0
        is_keypoints_reliable = avg_torso_conf > 0.4 
        
        pred_cls = 0
        conf = 0.0
        reason = ""
        
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
                
                # Heuristic Checks (The "Body Twisting" Logic)
                is_body_horizontal = False
                if is_keypoints_reliable:
                     # Only check angle if we actually see the torso
                     is_body_horizontal = current_angle > self.strictness_angle
                
                # 2. Velocity Check (Rapid descent)
                is_rapid_descent = abs(head_v) > self.strictness_velocity
                
                # 3. Bounding Box Aspect Ratio (Height / Width)
                # Standing/Sitting: H > W (Ratio > 1.0)
                # Lying down: W > H (Ratio < 1.0, or close to 1)
                if bbox is not None:
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    aspect_ratio = h / (w + 1e-6)
                    is_bbox_flat = aspect_ratio < 1.2 # Less than 1.2 means not very tall
                else:
                    is_bbox_flat = True # Default to pass if no bbox (unlikely)

                # Decision Tree
                is_fall_detected = False
                reason = ""
                
                # Logic:
                # To be a fall, it usually needs to be somewhat horizontal OR moving very fast down.
                # If the probability is SUPER high (>0.9), we might trust it more, 
                # but "Sitting" often yields high prob if trained on UR Fall (which has sitting).
                # So we MUST enforce some geometric rule for ADL safety.
                
                if smoothed_prob > self.FALL_CONFIDENCE_THRESHOLD:
                    # Condition A: Body is clearly horizontal (Best indicator)
                    if is_body_horizontal:
                        is_fall_detected = True
                        reason = f"Horizontal(Ang={int(current_angle)})"
                        
                    # Condition B: High Probability + (Rapid Descent or Flat BBox)
                    # Even if angle is vertical (e.g. crumbling down), velocity should be high
                    elif smoothed_prob > 0.85:
                        if is_rapid_descent:
                             is_fall_detected = True
                             reason = f"HighProb+Velocity"
                        elif is_bbox_flat:
                             is_fall_detected = True
                             reason = f"HighProb+FlatBox"
                             
                    # Condition C: Sitting Protection
                    # If Angle is Vertical (< Threshold) AND Velocity is Slow, REJECT even if prob is high.
                    # (This implicitly happens because we didn't set True above)
                
                if is_fall_detected:
                    self.consecutive_fall_frames += 1
                else:
                    self.consecutive_fall_frames = max(0, self.consecutive_fall_frames - 1)
                    
                # --- Robustness 2: Debounce Logic ---
                # Require N consecutive frames to confirm fall.
                # This prevents single-frame glitches from triggering the alarm.
                MIN_CONSECUTIVE_FRAMES = 3
                
                if self.consecutive_fall_frames >= MIN_CONSECUTIVE_FRAMES:
                     pred_cls = 1
                     conf = smoothed_prob
                     # Keep reason from the detection logic
                else:
                     pred_cls = 0
                     conf = 1.0 - smoothed_prob
                     reason = "" # Clear reason if not confirmed
        
        return pred_cls, conf, bbox, kpts_xy, confs, True, reason
                
        # If no detection
        return 0, 0.0, None, [], [], False, ""

    def reset_buffer(self):
        self.frame_buffer.clear()
        self.prev_head_y = None
        self.prev_angle = None
