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

def run_evaluation_mini():
    print("🚀 Starting Mini Eval (10 videos)...")
    
    full_csv_path = os.path.join(utils.DATA_DIR, 'raw_keypoints_colab_2.csv')
    label_csv_path = os.path.join(utils.DATA_DIR, 'labeling_work.csv')
    
    if not os.path.exists(full_csv_path): return
        
    df_kpts = pd.read_csv(full_csv_path)
    df_labels = pd.read_csv(label_csv_path)
    label_map = dict(zip(df_labels['filename'], df_labels['is_fall']))
    
    unique_videos = df_kpts['video_name'].unique()
    valid_videos = [v for v in unique_videos if v in label_map]
    
    # Pick 10
    subset_videos = valid_videos[:10]
    
    detector = FallDetector()
    y_true = []
    y_pred = []
    
    kpt_cols = []
    for i in range(17): kpt_cols.extend([f'x{i}', f'y{i}', f'c{i}'])
    
    grouped = df_kpts[df_kpts['video_name'].isin(subset_videos)].groupby('video_name')
    
    for vid_name, group in tqdm(grouped):
        is_fall_label = int(label_map[vid_name])
        group = group.sort_values('frame_idx')
        detector.reset_buffer()
        video_detected_fall = False
        kpts_data = group[kpt_cols].values
        
        for row_idx in range(len(kpts_data)):
            flat = kpts_data[row_idx]
            reshaped = flat.reshape(17, 3)
            kpts = reshaped[:, :2]
            confs = reshaped[:, 2]
            
            p, c, _, _, _, is_det = detector.process_keypoints(kpts, confs, None, None, skip_frames=2)
            if is_det and p == 1:
                video_detected_fall = True
                # No break to ensure we process enough frames for logic? 
                # Actually break is fine if we detect
                break
        
        y_true.append(is_fall_label)
        y_pred.append(1 if video_detected_fall else 0)

    acc = accuracy_score(y_true, y_pred)
    print(f"Mini Accuracy: {acc}")
    print(y_true)
    print(y_pred)

if __name__ == '__main__':
    run_evaluation_mini()
