import os
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import random

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import utils
from core.detectors import FallDetector 

def run_evaluation_quick():
    print("🚀 Starting Quick Quantitative Evaluation (Subset)...")
    
    # 1. Load CSV
    csv_path = os.path.join(utils.DATA_DIR, 'labeling_work.csv')
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Stratified Sampling
    df_fall = df[df['is_fall'] == 1]
    df_norm = df[df['is_fall'] == 0]
    
    # Sample 20 from each if possible
    n_sample = 20
    if len(df_fall) > n_sample: df_fall = df_fall.sample(n=n_sample, random_state=42)
    if len(df_norm) > n_sample: df_norm = df_norm.sample(n=n_sample, random_state=42)
    
    df_subset = pd.concat([df_fall, df_norm])
    df_subset = df_subset.sample(frac=1).reset_index(drop=True) # Shuffle
    
    print(f"📊 Processing Subset: {len(df_subset)} videos ({len(df_fall)} Fall, {len(df_norm)} Normal)")
    
    # 2. Initialize Detector
    detector = FallDetector()
    
    y_true = []
    y_pred = []
    
    missing_files = 0
    
    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Processing"):
        filename = row['filename']
        is_fall_label = int(row['is_fall'])
        
        # Construct path
        video_path = None
        candidates = [
            os.path.join(utils.BASE_DIR, '..', 'data', 'urfall', 'fall', filename),
            os.path.join(utils.BASE_DIR, '..', 'data', 'urfall', 'adl', filename),
            row['filepath'] if isinstance(row['filepath'], str) else ""
        ]
        
        for p in candidates:
            if p and os.path.exists(p):
                video_path = p
                break
        
        if not video_path:
            missing_files += 1
            # If detecting file is an issue, we can't eval
            y_true.append(is_fall_label)
            y_pred.append(0) # Assume negative if not found? Or skip? Better skip.
            # actually let's skip
            y_true.pop()
            continue
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        # Set Start/End
        start_sec = 0
        end_sec = 1000 
        
        if is_fall_label == 1:
            s = row['start_sec']
            e = row['end_sec']
            start_sec = max(0, s - 2)
            end_sec = e + 2
        else:
            # ADL: check first 8 seconds
            end_sec = 8
        
        # Seek
        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        detector.reset_buffer()
        
        video_detected_fall = False
        frame_count = 0
        skip_frames = 2 # Process every 2nd frame
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_pos_sec > end_sec:
                break
                
            frame_count += 1
            
            if frame_count % skip_frames == 0:
                frame = cv2.resize(frame, (640, 480))
                pred_cls, conf, bbox, kpts, confs, is_detected = detector.process_frame(frame, skip_frames)
                
                if is_detected and pred_cls == 1:
                    video_detected_fall = True
                    break 
        
        cap.release()
        
        y_true.append(is_fall_label)
        y_pred.append(1 if video_detected_fall else 0)

    # 3. Calculate Metrics
    if len(y_true) == 0:
        print("❌ No videos processed.")
        return

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Fall'], zero_division=0)
    
    print("\n" + "="*40)
    print("📊 Quick Evaluation Report (Subset)")
    print("="*40)
    print(f"Total Videos Processed: {len(y_true)}")
    print(f"Missing Videos: {missing_files}")
    print(f"Accuracy: {acc:.4f}")
    print("-" * 20)
    print("Confusion Matrix:")
    print(cm)
    print("-" * 20)
    print("Classification Report:")
    print(report)
    
    output_path = os.path.join(current_dir, 'stgcn_evaluation_report_quick.txt')
    with open(output_path, 'w') as f:
        f.write(f"ST-GCN Evaluation Results (Quick Subset)\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

if __name__ == '__main__':
    run_evaluation_quick()
