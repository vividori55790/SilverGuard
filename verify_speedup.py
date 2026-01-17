
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import utils
from core.detectors import FallDetector

def run_on_memory(frames, img_size):
    print(f"\n🧪 Testing YOLO Input Size: {img_size}px ({len(frames)} frames)...")
    
    detector = FallDetector()
    detector.YOLO_IMG_SIZE = img_size 
    
    times = []
    
    # Warmup
    if len(frames) > 0:
        detector.process(frames[0])

    for frame in tqdm(frames):
        start = time.perf_counter()
        
        # Detector process
        detector.process(frame)
        
        dt = time.perf_counter() - start
        times.append(dt)

    avg_time = np.mean(times) * 1000 # ms
    fps = 1.0 / np.mean(times)
    
    return {
        "size": img_size,
        "avg_ms": avg_time,
        "fps": fps
    }

def main():
    print("📷 Camera check...")
    frames = []
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            for _ in range(10):
                ret, frame = cap.read()
                if ret: frames.append(frame)
            cap.release()
    except: pass
    
    if len(frames) < 5:
        print("⚠️ Camera busy. Using SYNTHETIC RANDOM DATA.")
        print("   (Speed comparison is valid, Accuracy is N/A)")
        # 50 frames of random noise
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(30)]
    else:
        print(f"✅ Captured {len(frames)} live frames.")

    results = []
    res480 = run_on_memory(frames, 480)
    results.append(res480)
    
    res320 = run_on_memory(frames, 320)
    results.append(res320)
    
    print("\n📊 [Speed Benchmark Report]")
    print(f"{'Size':<10} {'Latency(ms)':<15} {'FPS':<10}")
    print("-" * 40)
    
    for r in results:
        print(f"{r['size']:<10} {r['avg_ms']:.1f}ms {'':<5} {r['fps']:.1f}")

    r480 = results[0]
    r320 = results[1]
    
    speedup = ((r320['fps'] - r480['fps']) / r480['fps']) * 100
    print(f"\n🚀 Estimated Speedup: +{speedup:.1f}%")

if __name__ == "__main__":
    main()
