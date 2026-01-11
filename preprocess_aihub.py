import cv2
import csv
import os
import torch
from ultralytics import YOLO
from tqdm import tqdm
import utils 

# ==========================================
# [설정] AI Hub 데이터셋 경로
# ==========================================
AIHUB_ROOT_DIR = '/app/data/aihub_videos'

# 자동 라벨링 임계값 (너비 / 높이)
# 1.2 이상이면 가로로 긴 것(누움), 0.8 이하면 세로로 긴 것(서있음)
THRESHOLD_FALL_AR = 1.2 
THRESHOLD_NORMAL_AR = 0.8
# ==========================================

def run():
    print("🚀 [2단계] AI Hub 데이터 마이닝 시작 (데이터 증강)")
    utils.ensure_dirs()

    if not os.path.exists(utils.CSV_PATH):
        print("❌ 오류: dataset.csv가 없습니다. preprocess_urfall.py를 먼저 실행하세요.")
        return

    # YOLO 모델 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(utils.YOLO_MODEL_PATH)

    # CSV 파일 이어쓰기 ('a' 모드)
    f = open(utils.CSV_PATH, 'a', newline='')
    writer = csv.writer(f)
    
    # 모든 하위 폴더의 영상 파일 찾기
    video_files = []
    for root, dirs, files in os.walk(AIHUB_ROOT_DIR):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                video_files.append(os.path.join(root, file))

    print(f"📂 발견된 영상: {len(video_files)}개 (하위 폴더 포함)")

    total_extracted = 0
    
    for video_path in tqdm(video_files, desc="AI Hub Mining"):
        cap = cv2.VideoCapture(video_path)
        filename = os.path.basename(video_path)
        
        # 긴 영상 처리 속도를 위해 프레임 건너뛰기 (3프레임당 1번 처리)
        frame_skip = 3 
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue

            # AI Hub 영상은 자르지 않고 원본 사용
            
            # YOLO 추론
            results = model(frame, verbose=False, device=device)
            
            # 사람이 감지되었고, 박스 정보가 있을 때
            if results[0].keypoints is not None and len(results[0].boxes) > 0:
                # 첫 번째 사람 정보 가져오기
                box = results[0].boxes.xywh[0].cpu().numpy() # x, y, width, height
                kpts = results[0].keypoints.xyn[0].cpu().numpy().flatten()
                confs = results[0].keypoints.conf[0].cpu().numpy().flatten()

                # 사람이 너무 작거나 신뢰도가 낮으면 무시
                if confs.mean() < 0.5: continue

                # [자동 라벨링 로직] Aspect Ratio (가로/세로) 계산
                w, h = box[2], box[3]
                aspect_ratio = w / h
                
                auto_label = -1
                
                if aspect_ratio > THRESHOLD_FALL_AR:
                    auto_label = 1 # 확실히 누움 (낙상)
                elif aspect_ratio < THRESHOLD_NORMAL_AR:
                    auto_label = 0 # 확실히 서있음 (정상)
                
                # 애매한 구간(0.8 ~ 1.2)은 버림 (노이즈 방지)

                if auto_label != -1:
                    row = [auto_label, filename]
                    for x, y, c in zip(kpts[0::2], kpts[1::2], confs):
                        row.extend([x, y, c])
                    writer.writerow(row)
                    total_extracted += 1
                    
        cap.release()
    
    f.close()
    print(f"✅ AI Hub 처리 완료! (추가된 데이터: {total_extracted}장)")
    print(f"💾 최종 데이터셋: {utils.CSV_PATH}")

if __name__ == '__main__':
    run()