import os
import sys
import cv2
import numpy as np
import time

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import utils
# Use the updated FallDetector which uses ST-GCN
from core.detectors import FallDetector 

import os
import sys
import cv2
import numpy as np
import pandas as pd
import time

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import utils
from core.detectors import FallDetector 

def run_test_detection():
    print("🚀 Test Detection Started (ST-GCN) - Auto Iteration")
    
    # 1. Load CSV
    csv_path = os.path.join(utils.DATA_DIR, 'labeling_work.csv')
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    # Shuffle or just iterate. Let's iterate.
    
    # 2. Detector
    detector = FallDetector()
    
    # Loop videos
    for idx, row in df.iterrows():
        filename = row['filename']
        is_fall = row['is_fall']
        
        # Construct path: Try UR Fall path pattern first, then generic
        # UR Fall structure in data: data/urfall/fall/filename or data/urfall/adl/filename
        # labeling_work.csv has full absolute paths in 'filepath' column, let's use that if valid,
        # but the CSV paths might be from another machine. 
        # Let's try to find the file in local directories based on filename.
        
        video_path = None
        # Check standard locations
        candidates = [
            os.path.join(utils.BASE_DIR, '..', 'data', 'urfall', 'fall', filename),
            os.path.join(utils.BASE_DIR, '..', 'data', 'urfall', 'adl', filename),
            # Also check the path in CSV if it happens to match local
            row['filepath']
        ]
        
        for p in candidates:
            if os.path.exists(p):
                video_path = p
                break
        
        if not video_path:
            # print(f"⚠️ Video not found locally: {filename}")
            continue
            
        print(f"▶ Playing [{idx+1}/{len(df)}]: {filename} (Fall: {is_fall})")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        # Set Start/End
        start_sec = 0
        end_sec = 10 # Default 10s for normal
        
        if is_fall == 1:
            # Play a bit before and after fall
            s = row['start_sec']
            e = row['end_sec']
            start_sec = max(0, s - 2)
            end_sec = e + 2
            # Calculate duration
            duration = end_sec - start_sec
            # If duration is very long, maybe just clamp it, but user asked for fall segment
        
        # Seek
        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        detector.reset_buffer()
        
        frame_count = 0
        skip_frames = 2
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_pos_sec > end_sec:
                break
                
            frame = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()
            frame_count += 1
            
            # Inference
            status_text = "Monitoring..."
            status_color = (0, 255, 0)
            
            if frame_count % skip_frames == 0:
                pred_cls, conf, bbox, kpts, confs, is_detected = detector.process_frame(frame, skip_frames)
                
                if is_detected:
                    # Draw Skeleton
                    for i, (x, y) in enumerate(kpts):
                        if confs[i] > 0.5:
                            cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                    
                    # Draw BBox
                    if bbox is not None:
                        if pred_cls == 1:
                            status_text = f"FALL DETECTED! ({conf:.2f})"
                            status_color = (0, 0, 255)
                        else:
                            status_text = f"Normal ({conf:.2f})"
                            status_color = (0, 255, 0)
                            
                        cv2.rectangle(display_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), status_color, 2)
                        cv2.putText(display_frame, status_text, (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Info Overlay
            cv2.putText(display_frame, f"Video: {filename}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Type: {'FALL' if is_fall else 'ADL'}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Time: {current_pos_sec:.1f}s / {end_sec:.1f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("ST-GCN Fall Detection Test", display_frame)
            
            if cv2.waitKey(33) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        
    cv2.destroyAllWindows()
    print("👋 All videos finished.")

if __name__ == '__main__':
    run_test_detection()
