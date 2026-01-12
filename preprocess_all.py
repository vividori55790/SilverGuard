import cv2
import csv
import os
import sys
from ultralytics import YOLO
from tqdm import tqdm

# ==============================================================================
# [설정] 경로 및 모델 설정
# ==============================================================================

# 1. 경로 설정 (도커 내부 기준)
# 이 파일(preprocess_extract_all.py)은 /app/SilverGuard 안에 위치함
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# [핵심] 모델 경로: SilverGuard/models/yolo11s-pose.pt
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolo11s-pose.pt')

# 데이터 루트 디렉토리 (모든 영상 검색)
DATA_ROOT_DIR = '/app/data'

# 결과 저장 CSV 경로
OUTPUT_CSV_PATH = os.path.join(DATA_ROOT_DIR, 'raw_keypoints_all.csv')

# ==============================================================================

def ensure_dirs():
    """데이터 저장 경로 폴더 생성"""
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

def run():
    print("🚀 [전처리] YOLO11s-Pose (기존 모델 대체) 데이터 추출 시작")
    print(f"   - 모델 경로: {MODEL_PATH}")
    print(f"   - 데이터 경로: {DATA_ROOT_DIR}")

    ensure_dirs()

    # 1. 모델 로드 및 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ [Error] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("   -> 'SilverGuard/models/' 폴더 안에 'yolo11s-pose.pt'를 넣어주세요.")
        # 자동 다운로드 시도 (옵션)
        try:
            print("   -> 모델 자동 다운로드를 시도합니다...")
            model = YOLO('yolo11s-pose.pt') # 현재 폴더에 다운로드
            # 다운로드된 파일을 models 폴더로 이동
            os.rename('yolo11s-pose.pt', MODEL_PATH)
            print("   -> ✅ 다운로드 및 이동 성공!")
        except Exception as e:
            print(f"   -> ❌ 자동 다운로드 실패. 직접 넣어주세요. ({e})")
            return
    
    try:
        model = YOLO(MODEL_PATH)
        print("✅ 모델 로드 성공!")
    except Exception as e:
        print(f"❌ 모델 로드 에러: {e}")
        return

    # 2. 결과 CSV 파일 준비
    # 라벨링 없이 순수 데이터(Raw Data)만 추출합니다.
    f = open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    
    # 헤더: [파일명, 프레임, 시간, 트랙ID] + [17개 관절 x,y,conf]
    header = ['video_name', 'frame_idx', 'time_sec', 'track_id']
    for i in range(17):
        header.extend([f'x{i}', f'y{i}', f'c{i}'])
    writer.writerow(header)

    # 3. 전체 영상 파일 탐색
    video_files = []
    for root, dirs, files in os.walk(DATA_ROOT_DIR):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))

    print(f"📂 처리할 영상 파일 수: {len(video_files)}개")

    total_frames = 0

    # 4. 추론 및 데이터 추출 (Tracking)
    for video_path in tqdm(video_files, desc="Extracting Keypoints"):
        filename = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30.0
        
        # YOLO Tracking 실행 (persist=True: ID 유지)
        results = model.track(source=video_path, persist=True, stream=True, verbose=False, device='cpu')
        
        frame_idx = 0
        for result in results:
            frame_idx += 1
            current_sec = frame_idx / fps
            
            if result.boxes.id is None or result.keypoints is None:
                continue

            # GPU -> CPU 변환
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            keypoints = result.keypoints.xyn.cpu().numpy()
            confs = result.keypoints.conf.cpu().numpy()

            # 감지된 모든 사람 저장
            for i, track_id in enumerate(track_ids):
                row = [filename, frame_idx, round(current_sec, 4), track_id]
                
                kpts = keypoints[i].flatten()
                conf = confs[i].flatten()
                
                for k in range(17):
                    row.extend([kpts[2*k], kpts[2*k+1], conf[k]])
                
                writer.writerow(row)
                total_frames += 1
        cap.release()

    f.close()
    print("="*50)
    print("✅ 데이터 추출 완료!")
    print(f"💾 저장 위치: {OUTPUT_CSV_PATH}")
    print(f"🔢 총 프레임: {total_frames}")
    print("="*50)

if __name__ == '__main__':
    run()