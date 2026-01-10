import cv2
import os
import glob
import numpy as np
import pandas as pd
import zipfile
import shutil
import random
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# [설정] 데이터 경로 지정
# ==========================================
# 1. 특정 압축 파일 지정: "./datasets/video01.zip"
# 2. 압축 파일들이 있는 폴더 지정: "./datasets/" (자동으로 하나 선택됨)
# 3. 이미지가 풀려있는 폴더 지정: "./datasets/extracted/"
SOURCE_PATH = "./sample_data"  # <- 여기에 데이터 경로 입력

# 결과 저장 설정
SAVE_RESULT = True
SAVE_DIR = "./output_result"
TEMP_DIR = "./temp_extract_data"  # 압축 풀 임시 폴더

# ==========================================
# 1. 범용 낙상 감지 클래스 (Universal Fall Detector)
# ==========================================
class UniversalFallDetector:
    def __init__(self):
        self.history = {}       # ID별 이전 프레임 좌표
        self.risk_buffer = {}   # ID별 위험 점수 버퍼

        # [감도 설정]
        self.FALL_CONFIDENCE = 0.60
        self.IMPACT_THRESH = 0.05

    def get_body_orientation(self, kpts):
        valid_pts = kpts[kpts[:, 2] > 0.5]
        if len(valid_pts) < 5: return 0.0
        
        x_coords, y_coords = valid_pts[:, 0], valid_pts[:, 1]
        std_x, std_y = np.std(x_coords), np.std(y_coords)
        
        if std_y == 0: return 0.0
        return std_x / (std_y + 1e-6)

    def update(self, keypoints, bbox, track_id=0):
        nose = keypoints[0][:2]
        shoulder_mid = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hip_mid = (keypoints[11][:2] + keypoints[12][:2]) / 2
        
        x1, y1, x2, y2 = bbox
        box_h = max(1, y2 - y1)

        # 1. Motion
        current_y = (shoulder_mid[1] + hip_mid[1]) / 2
        prev_y = self.history.get(track_id, current_y)
        self.history[track_id] = current_y
        velocity = (current_y - prev_y) / box_h

        impact_score = 0
        if velocity > self.IMPACT_THRESH: impact_score = 1.0
        elif velocity > self.IMPACT_THRESH * 0.5: impact_score = 0.5

        # 2. Pose & Topology
        dx = abs(shoulder_mid[0] - hip_mid[0])
        dy = abs(shoulder_mid[1] - hip_mid[1])
        is_spine_horizontal = dx > dy * 1.5
        
        head_inverted = nose[1] > hip_mid[1]
        head_on_floor = nose[1] > (y2 - box_h * 0.2)
        
        # 3. Orientation
        spread_ratio = self.get_body_orientation(keypoints)
        is_body_flat = spread_ratio > 1.2

        # Risk Calculation
        risk = 0.0
        risk += impact_score * 0.3
        if is_spine_horizontal or is_body_flat: risk += 0.4
        if head_inverted or head_on_floor: risk += 0.3
        if head_inverted and impact_score > 0: risk += 0.2

        prev_risk = self.risk_buffer.get(track_id, 0.0)
        smoothed_risk = prev_risk * 0.6 + risk * 0.4
        self.risk_buffer[track_id] = smoothed_risk

        is_fall = smoothed_risk > self.FALL_CONFIDENCE
        debug_msg = f"R:{smoothed_risk:.2f}"
        return is_fall, smoothed_risk, debug_msg

# ==========================================
# 2. 데이터 준비 유틸리티 (압축 해제 로직)
# ==========================================
def prepare_data(source_path):
    target_path = source_path
    
    # 0. 임시 폴더 초기화
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    selected_zip = None

    # 1. 입력이 .zip 파일인 경우
    if os.path.isfile(source_path) and source_path.lower().endswith('.zip'):
        selected_zip = source_path

    # 2. 입력이 폴더인 경우 (내부 탐색)
    elif os.path.isdir(source_path):
        # 먼저 이미지 파일이 있는지 확인
        extensions = ['*.jpg', '*.jpeg', '*.png']
        has_images = False
        for ext in extensions:
            if glob.glob(os.path.join(source_path, ext)) or glob.glob(os.path.join(source_path, '**', ext), recursive=True):
                has_images = True
                break
        
        if has_images:
            return source_path, None # 이미지가 바로 있으면 압축해제 안 함
        
        # 이미지가 없으면 zip 파일 탐색
        zip_files = glob.glob(os.path.join(source_path, '*.zip')) + glob.glob(os.path.join(source_path, '**', '*.zip'), recursive=True)
        if zip_files:
            print(f"📦 폴더 내에서 {len(zip_files)}개의 압축 파일을 발견했습니다.")
            selected_zip = random.choice(zip_files) # 랜덤 선택 (원하면 index 0으로 고정 가능)
        else:
            return source_path, None # 아무것도 없음

    # 3. 압축 해제 실행
    if selected_zip:
        print(f"🔓 압축 해제 중...: {os.path.basename(selected_zip)}")
        try:
            with zipfile.ZipFile(selected_zip, 'r') as z:
                z.extractall(TEMP_DIR)
            print("✅ 압축 해제 완료!")
            return TEMP_DIR, selected_zip
        except Exception as e:
            print(f"❌ 압축 해제 실패: {e}")
            return None, None

    return source_path, None

# ==========================================
# 3. 메인 실행 함수
# ==========================================
def main():
    print("⏳ 모델을 로드 중입니다...")
    try:
        model = YOLO('yolo11n-pose.pt')
    except:
        model = YOLO('yolov8n-pose.pt')

    detector = UniversalFallDetector()

    # 데이터 준비 (압축 해제 등)
    data_path, extracted_zip_name = prepare_data(SOURCE_PATH)
    if data_path is None:
        print("❌ 데이터를 준비하는 과정에서 오류가 발생했습니다.")
        return

    # 이미지/영상 로드
    frames = []
    cap = None
    
    if os.path.isfile(data_path) and not data_path.lower().endswith('.zip'):
        # 동영상 파일인 경우
        cap = cv2.VideoCapture(data_path)
        print(f"🎬 동영상 파일 로드: {data_path}")
    else:
        # 이미지 폴더인 경우 (압축 해제된 폴더 포함)
        extensions = ['*.jpg', '*.jpeg', '*.png']
        for ext in extensions:
            frames.extend(glob.glob(os.path.join(data_path, ext)))
            frames.extend(glob.glob(os.path.join(data_path, '**', ext), recursive=True))
        frames.sort()
        
        if not frames:
            print(f"❌ '{data_path}' 경로에서 이미지나 영상을 찾을 수 없습니다.")
            return
        print(f"📂 이미지 로드 완료: {len(frames)}장 (소스: {extracted_zip_name if extracted_zip_name else data_path})")

    # 결과 저장 준비
    if SAVE_RESULT:
        os.makedirs(SAVE_DIR, exist_ok=True)
        # 압축 파일명이나 폴더명으로 서브폴더 생성
        sub_name = os.path.splitext(os.path.basename(extracted_zip_name))[0] if extracted_zip_name else "manual_run"
        current_save_dir = os.path.join(SAVE_DIR, sub_name)
        os.makedirs(current_save_dir, exist_ok=True)
        print(f"💾 결과 저장 경로: {current_save_dir}")

    print("\n🚀 분석 시작! (화면 클릭 후 'q'로 종료, 'p'로 일시정지)")

    global_consecutive_fall_frames = 0
    frame_idx = 0
    paused = False

    while True:
        if not paused:
            if cap:
                ret, frame = cap.read()
                if not ret: break
            else:
                if frame_idx >= len(frames): break
                frame = cv2.imread(frames[frame_idx])
                frame_idx += 1

            if frame is None: continue

            # --- 분석 로직 시작 ---
            results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
            annotated_frame = frame.copy()
            if results[0].keypoints is not None:
                annotated_frame = results[0].plot(kpt_radius=5)

            max_risk_in_frame = 0.0
            status_text, status_color = "Safe", (0, 255, 0)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                kpts_data = results[0].keypoints.data.cpu().numpy()

                for box, track_id, kpts in zip(boxes, track_ids, kpts_data):
                    is_fall, risk, msg = detector.update(kpts, box, track_id)
                    max_risk_in_frame = max(max_risk_in_frame, risk)
                    
                    cx, cy = int(box[0]), int(box[1])
                    c = (0, 0, 255) if is_fall else (0, 255, 255)
                    cv2.putText(annotated_frame, f"ID:{track_id} {msg}", (cx, cy-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

            if max_risk_in_frame > detector.FALL_CONFIDENCE:
                global_consecutive_fall_frames += 1
            else:
                global_consecutive_fall_frames = max(0, global_consecutive_fall_frames - 1)

            if global_consecutive_fall_frames >= 5:
                status_text, status_color = "FALL DETECTED!", (0, 0, 255)
                cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
            elif global_consecutive_fall_frames > 2:
                status_text, status_color = "Warning...", (0, 165, 255)

            cv2.putText(annotated_frame, status_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
            # --- 분석 로직 끝 ---

            # 화면 출력용 리사이즈
            display_frame = annotated_frame.copy()
            if display_frame.shape[1] > 1280:
                ratio = 1280 / display_frame.shape[1]
                display_frame = cv2.resize(display_frame, (1280, int(display_frame.shape[0] * ratio)))
            
            cv2.imshow("Fall Detection", display_frame)

            if SAVE_RESULT:
                save_name = f"{frame_idx:04d}_result.jpg"
                cv2.imwrite(os.path.join(current_save_dir, save_name), annotated_frame)

        # 키 입력 처리
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused  # 일시정지 기능 추가

    if cap: cap.release()
    cv2.destroyAllWindows()
    # 임시 폴더 삭제 (옵션: 결과 확인 후 삭제하려면 주석 처리)
    # if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    print("✅ 테스트 종료")

if __name__ == "__main__":
    main()