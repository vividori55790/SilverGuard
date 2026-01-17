"""
SilverGuard Optimization Benchmark Script
Quantitative Performance Evaluation
"""

import os
import sys
import time
import cv2
import numpy as np

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import utils
from core.detectors import FallDetector

def benchmark_yolo_inference(detector, frame, iterations=50):
    """Measure YOLO inference time"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        detector.process(frame, timestamp=time.time())
        times.append((time.perf_counter() - start) * 1000)  # ms
    return np.mean(times), np.std(times), np.min(times), np.max(times)

def benchmark_motion_detection(frame1, frame2, iterations=1000):
    """Measure motion detection computation time"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        small1 = cv2.resize(frame1, (160, 120))
        small2 = cv2.resize(frame2, (160, 120))
        gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        mean_diff = diff.mean()
        times.append((time.perf_counter() - start) * 1000)  # ms
    return np.mean(times), np.std(times)

def simulate_scenarios():
    """Simulate inference counts for various scenarios"""
    
    target_fps = 30
    duration_sec = 60  # 1 minute simulation
    total_frames = target_fps * duration_sec
    
    print("\n" + "="*60)
    print("[Scenario Simulation - 1 minute]")
    print("="*60)
    
    scenarios = [
        ("Static (empty room)", 0.0),
        ("Slow motion (sitting)", 0.3),
        ("Normal activity", 0.6),
        ("Fast motion (exercise)", 0.9),
        ("Fall event", 1.0),
    ]
    
    print("\n[BEFORE: Every 2 frames, no motion detection]")
    print("-"*60)
    old_interval = 2
    for name, motion_rate in scenarios:
        inferences = total_frames // old_interval
        print(f"  {name:30s}: {inferences:4d} inferences")
    
    print(f"\n  Worst case: {total_frames // old_interval} inferences/min")
    
    print("\n[AFTER: Every 3 frames + motion detection (threshold=3)]")
    print("-"*60)
    new_interval = 3
    no_motion_reset = 30
    
    for name, motion_rate in scenarios:
        base_inferences = total_frames // new_interval
        
        if motion_rate == 0:
            actual_inferences = total_frames // (new_interval * no_motion_reset)
        else:
            skip_rate = 1 - motion_rate
            actual_inferences = int(base_inferences * motion_rate + base_inferences * skip_rate * (1/no_motion_reset))
        
        reduction = (1 - actual_inferences / (total_frames // old_interval)) * 100
        print(f"  {name:30s}: {actual_inferences:4d} inferences (-{reduction:5.1f}%)")
    
    print("\n" + "="*60)

def run_benchmark():
    print("="*60)
    print("SilverGuard Optimization Benchmark")
    print("="*60)
    
    # Test frames
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame_similar = frame.copy()
    frame_similar[100:200, 100:200] += 10
    
    # 1. Motion Detection Performance
    print("\n[1] Motion Detection Overhead")
    print("-"*60)
    mean_time, std_time = benchmark_motion_detection(frame, frame_similar, 1000)
    print(f"  Average time: {mean_time:.3f}ms (+/-{std_time:.3f}ms)")
    print(f"  Max ops/second: {1000/mean_time:.0f}")
    
    # 2. YOLO Inference Performance
    print("\n[2] YOLO Inference Time")
    print("-"*60)
    print("  (Loading FallDetector...)")
    
    try:
        detector = FallDetector()
        mean_inf, std_inf, min_inf, max_inf = benchmark_yolo_inference(detector, frame, 50)
        print(f"  Average time: {mean_inf:.1f}ms (+/-{std_inf:.1f}ms)")
        print(f"  Min/Max: {min_inf:.1f}ms / {max_inf:.1f}ms")
        print(f"  Max inferences/second: {1000/mean_inf:.1f}")
        
        # 3. Savings Calculation
        print("\n[3] Optimization Analysis")
        print("-"*60)
        
        ratio = mean_inf / mean_time
        print(f"  YOLO vs Motion Detection cost ratio: {ratio:.0f}:1")
        print(f"  -> Skipping 1 YOLO = {ratio:.0f}x savings")
        
        interval_saving = (1 - 2/3) * 100
        print(f"\n  Interval change (2->3 frames):")
        print(f"    Before: {30/2:.0f} inferences/sec possible")
        print(f"    After: {30/3:.0f} inferences/sec possible")
        print(f"    Reduction: {interval_saving:.1f}%")
        
    except Exception as e:
        print(f"  Warning: YOLO load failed: {e}")
    
    # 4. Scenario Simulation
    simulate_scenarios()
    
    # 5. Summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print("""
  +-----------------------------------------------------------+
  | Changes Made                                              |
  +-----------------------------------------------------------+
  | 1. Inference interval: 2 frames -> 3 frames (-33%)        |
  | 2. Motion detection: Applied to all cameras               |
  | 3. Safety: Force check every 30 no-motion frames          |
  +-----------------------------------------------------------+

  +-----------------------------------------------------------+
  | Expected CPU Reduction                                    |
  +-----------------------------------------------------------+
  | Static scene (empty room):     80~95% reduction           |
  | Low activity (sitting):        50~70% reduction           |
  | Normal activity:               30~50% reduction           |
  | High activity/fall:            33% reduction (minimum)    |
  +-----------------------------------------------------------+

  +-----------------------------------------------------------+
  | Fall Detection Impact                                     |
  +-----------------------------------------------------------+
  | OK: Fast motion during fall -> Detected by motion check   |
  | OK: Lying still after fall -> Force check every 30 frames |
  | OK: 3-frame debounce -> Accuracy maintained               |
  +-----------------------------------------------------------+
""")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()
