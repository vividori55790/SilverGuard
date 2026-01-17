
import os
import sys
import cv2
import numpy as np
import glob
from core.detectors import FallDetector
import utils

def compute_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    # Ensure numpy or list
    boxA = np.array(boxA).flatten()
    boxB = np.array(boxB).flatten()
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def main():
    # 1. Find a test image
    search_path = os.path.join(utils.ALERT_DIR, "*.jpg")
    files = glob.glob(search_path)
    if not files:
        print("❌ No test images found in alert_images.")
        return
        
    img_path = files[0] 
    print(f"🖼️ Testing on real sample: {os.path.basename(img_path)}")
    
    original_img = cv2.imread(img_path)
    if original_img is None:
        print("❌ Failed to load image.")
        return

    # 2. Initialize Detectors
    detector = FallDetector()
    
    # Run 480
    print("🔹 Running High-Res (480px)...")
    detector.YOLO_IMG_SIZE = 480
    # Warmup
    detector.process(original_img)
    # Actual
    # Returns: pred_cls, conf, bbox, kpts, confs, is_det, reason
    # kpts is typically [x, y] list for 17 keypoints
    try:
        _, conf480, bbox480, kpts480, _, _, _ = detector.process(original_img)
    except Exception as e:
        print(f"Error checking 480: {e}")
        return
    
    # Run 320
    print("🔹 Running Optimized (320px)...")
    detector.YOLO_IMG_SIZE = 320
    _, conf320, bbox320, kpts320, _, _, _ = detector.process(original_img)
    
    # 3. Compare Results
    print("\n📊 [Quantitative Accuracy Analysis]")
    
    # Check if detections exist
    # bbox usually [x1, y1, x2, y2]
    if bbox480 is None or len(bbox480) == 0:
        print("⚠️ 480px model failed to detect person.")
        return
        
    if bbox320 is None or len(bbox320) == 0:
        print("❌ 320px model LOST the detection! (Accuracy Failure)")
        return
        
    print(f"✅ Detection Success: Both models detected person.")
    
    # Compare Confidence
    print(f"   Confidence (480px): {float(conf480):.4f}")
    print(f"   Confidence (320px): {float(conf320):.4f}")
    
    # Compare BBox IOU
    iou = compute_iou(bbox480, bbox320)
    print(f"   Bounding Box IOU: {iou:.4f}")
    
    # Compare Keypoints
    # kpts480 expected to be (17, 2) or flattened?
    # detectors.py usually processes them. 
    # Let's try to convert and measure.
    try:
        kp_arr_480 = np.array(kpts480)
        kp_arr_320 = np.array(kpts320)
        
        # Check shapes
        if kp_arr_480.shape != kp_arr_320.shape:
             # Maybe (17,3) vs (17,2)?
             # Flatten?
             pass
             
        diff = kp_arr_480 - kp_arr_320
        # If shape is (17, 2), axis=1 gives distance per keypoint
        # If shape is (34,), reshape
        if kp_arr_480.ndim == 1:
             kp_arr_480 = kp_arr_480.reshape(-1, 2)
             kp_arr_320 = kp_arr_320.reshape(-1, 2)
             
        dist = np.linalg.norm(kp_arr_480 - kp_arr_320, axis=1)
        avg_pixel_shift = np.mean(dist)
        
        print(f"   Keypoint Shift: {avg_pixel_shift:.2f} pixels avg")
        
        if iou > 0.85:
            print("\n🎉 [Verdict] PASS: Accuracy is maintained.")
        else:
            print("\n⚠️ [Verdict] WARNING: Low IOU.")
            
    except Exception as e:
        print(f"   Keypoint comparison skipped: {e}")

if __name__ == "__main__":
    main()
