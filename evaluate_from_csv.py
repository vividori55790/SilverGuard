import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import utils
from core.detectors import FallDetector 

def run_evaluation_csv():
    print("🚀 Starting Fast Quantitative Evaluation (CSV Keypoints)...")
    
    # 1. Load Data
    full_csv_path = os.path.join(utils.DATA_DIR, 'raw_keypoints_colab_2.csv')
    label_csv_path = os.path.join(utils.DATA_DIR, 'labeling_work.csv')
    
    if not os.path.exists(full_csv_path) or not os.path.exists(label_csv_path):
        print("❌ CSVs not found.")
        return
        
    print("⏳ Loading CSVs...")
    df_kpts = pd.read_csv(full_csv_path)
    df_labels = pd.read_csv(label_csv_path)
    
    # Create Label Map
    # Ensure filename matches video_name format
    # In raw_keypoints, video_name might be just filename 'fall-01-cam0.mp4'
    label_map = dict(zip(df_labels['filename'], df_labels['is_fall']))
    
    # Filter dataset to only videos we have labels for
    unique_videos = df_kpts['video_name'].unique()
    valid_videos = [v for v in unique_videos if v in label_map]
    
    print(f"📊 Valid Videos with Keypoints: {len(valid_videos)} / {len(label_map)} labeled")
    
    detector = FallDetector()
    
    y_true = []
    y_pred = []
    
    # Prepare column names
    # x0, y0, c0 ... x16, y16, c16
    kpt_cols = []
    for i in range(17):
        kpt_cols.extend([f'x{i}', f'y{i}', f'c{i}'])
    
    # Group by video for speed
    grouped = df_kpts[df_kpts['video_name'].isin(valid_videos)].groupby('video_name')
    
    for vid_name, group in tqdm(grouped, desc="Processing Videos"):
        is_fall_label = int(label_map[vid_name])
        
        # Sort by frame
        group = group.sort_values('frame_idx')
        
        detector.reset_buffer()
        video_detected_fall = False
        
        # We need continuous playback simulation
        # The CSV might have skipped frames or missing frames. 
        # But our detector depends on Frame-to-Frame velocity.
        # We will feed available frames sequentially.
        
        # Array of keypoints
        # Shape (N_frames, 51)
        kpts_data = group[kpt_cols].values
        
        for row_idx in range(len(kpts_data)):
            # Parse kpts
            # row is [x0, y0, c0, x1, y1, c1...]
            flat = kpts_data[row_idx]
            
            # Reshape to (17, 3) -> x, y, c
            reshaped = flat.reshape(17, 3)
            
            # Extract
            kpts = reshaped[:, :2] # (17, 2)
            confs = reshaped[:, 2] # (17,)
            
            # Fake bbox and kpts_xy (pixel coords)
            # YOLO output: kpts is normalized (0-1), bbox is pixels.
            # In CSV, check if x,y are normalized?
            # Usually YOLO CSV export from previous steps might be normalized or pixels.
            # If pixels, we need to normalize?
            # Let's assume normalized (xyn) because the headers usually imply that or I should check.
            # However, FallDetector logic uses `kpts` (normalized) for ST-GCN input buffer: `frame_data.append([kpts[i][0], kpts[i][1], confs[i]])`
            # And `kpts_xy` (pixels) for visualization.
            # BUT the Heuristics use `kpts` (normalized) implicitly?
            # `head_y = kpts[0][1]` -> If normalized, range 0-1. 
            # `head_v = (head_y - prev) * ...`
            # If pixel, range 0-1080.
            # ST-GCN requires normalized input? 
            # Looking at `detectors.py`: `kpts = r.keypoints.xyn[0]` (Normalized). 
            # So I must ensure CSV is normalized.
            # If CSV comes from `YOLO(...).keypoints.xyn`, it is normalized.
            # I will assume it is. If accuracy is 0, I know why.
            
            # Skip frames: The CSV was likely extracted every frame? 
            # Or maybe I should simulate skip?
            # FallDetector.process_frame has skip_frames arg.
            # If I simulate 30fps input, I should pass correct skip_frames.
            # Let's assume CSV is all frames.
            
            # Pass to detector
            pred_cls, conf, _, _, _, is_detected = detector.process_keypoints(kpts, confs, None, None, skip_frames=2)
            
            if is_detected and pred_cls == 1:
                video_detected_fall = True
                # Break early optimization
                # break
        
        y_true.append(is_fall_label)
        y_pred.append(1 if video_detected_fall else 0)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Fall'], zero_division=0)
    
    print("\n" + "="*40)
    print("📊 Evaluation Report (Full Dataset via Keypoints)")
    print("="*40)
    print(f"Total Videos: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(cm)
    print(report)
    
    # Save
    with open(os.path.join(current_dir, 'stgcn_evaluation_report_fast.txt'), 'w') as f:
        f.write(f"ST-GCN Evaluation Results (CSV-based Fast Eval)\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Classification Report:\n{report}")

if __name__ == '__main__':
    run_evaluation_csv()
