import cv2
import csv
import os
import torch
import math
from ultralytics import YOLO
from tqdm import tqdm
import utils 

# ==========================================
# [설정] AI Hub 데이터셋 경로 (도커 내부 경로 유지)
# ==========================================
AIHUB_ROOT_DIR = '/app/data/aihub_videos'

# 자동 라벨링 임계값
THRESHOLD_FALL_AR = 1.2 
THRESHOLD_NORMAL_AR = 0.8
# ==========================================

def calculate_angle(p1, p2):
    """ 두 점 사이의 각도 계산 """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(math.degrees(math.atan2(dx, dy)))

def run():
    print("🚀 [2단계] AI Hub 데이터 마이닝 시작 (Docker Env)")
    utils.ensure_dirs()

    # YOLO 모델 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   - Device: {device}")
    
    # 도커 내부 모델 경로 사용
    try:
        model = YOLO(utils.YOLO_MODEL_PATH)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {utils.YOLO_MODEL_PATH}")
        print("   -> 도커 내부에 모델 파일이 있는지 확인하세요.")
        return

    # CSV 파일 준비 (없으면 헤더 생성, 있으면 이어쓰기)
    file_exists = os.path.isfile(utils.CSV_PATH)
    mode = 'a' if file_exists else 'w'
    
    f = open(utils.CSV_PATH, mode, newline='')
    writer = csv.writer(f)
    
    # 파일이 새로 생성되는 경우 헤더 작성 (학습 코드와 포맷 통일)
    if not file_exists:
        print("📝 새로운 dataset.csv 파일을 생성합니다.")
        header = ['label', 'video_name']
        for i in range(17): header.extend([f'x{i}', f'y{i}', f'c{i}'])
        header.extend(['head_velocity', 'angle_velocity', 'torso_angle']) # 필수 파생변수
        writer.writerow(header)
    
    # 영상 파일 찾기
    video_files = []
    for root, dirs, files in os.walk(AIHUB_ROOT_DIR):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                video_files.append(os.path.join(root, file))

    print(f"📂 발견된 영상: {len(video_files)}개")
    if len(video_files) == 0:
        print(f"⚠️ 영상이 없습니다. 경로 확인: {AIHUB_ROOT_DIR}")
        f.close()
        return

    total_extracted = 0
    
    for video_path in tqdm(video_files, desc="AI Hub Mining"):
        cap = cv2.VideoCapture(video_path)
        filename = os.path.basename(video_path)
        
        frame_skip = 3 
        frame_idx = 0
        
        # 속도 계산을 위한 이전 프레임 정보
        prev_head_y = None
        prev_angle = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue

            # YOLO 추론
            results = model(frame, verbose=False, device=device)
            
            if results[0].keypoints is not None and len(results[0].boxes) > 0:
                box = results[0].boxes.xywh[0].cpu().numpy()
                kpts = results[0].keypoints.xyn[0].cpu().numpy().flatten()
                confs = results[0].keypoints.conf[0].cpu().numpy().flatten()

                if confs.mean() < 0.5: continue

                # [자동 라벨링] Aspect Ratio
                w, h = box[2], box[3]
                aspect_ratio = w / h
                
                auto_label = -1
                if aspect_ratio > THRESHOLD_FALL_AR: auto_label = 1
                elif aspect_ratio < THRESHOLD_NORMAL_AR: auto_label = 0
                
                if auto_label != -1:
                    # ------------------------------------------------
                    # [로직 추가] 파생 변수(Velocity, Angle) 계산
                    # ------------------------------------------------
                    head_y = kpts[1] # index 0의 y좌표
                    
                    # 어깨(5,6), 골반(11,12) 인덱스 주의 (flatten 상태이므로 *2 필요)
                    # kpts 배열: [x0, y0, x1, y1, ... x16, y16]
                    
                    # x5=kpts[10], y5=kpts[11] / x6=kpts[12], y6=kpts[13]
                    shoulder_mid_x = (kpts[10] + kpts[12]) / 2
                    shoulder_mid_y = (kpts[11] + kpts[13]) / 2
                    
                    # x11=kpts[22], y11=kpts[23] / x12=kpts[24], y12=kpts[25]
                    hip_mid_x = (kpts[22] + kpts[24]) / 2
                    hip_mid_y = (kpts[23] + kpts[25]) / 2
                    
                    current_angle = calculate_angle((shoulder_mid_x, shoulder_mid_y), (hip_mid_x, hip_mid_y))

                    if prev_head_y is not None:
                        head_velocity = (head_y - prev_head_y) * (30 / frame_skip)
                        angle_velocity = (current_angle - prev_angle) * (30 / frame_skip)
                    else:
                        head_velocity = 0
                        angle_velocity = 0

                    prev_head_y = head_y
                    prev_angle = current_angle
                    # ------------------------------------------------

                    row = [auto_label, filename]
                    for i in range(17):
                        row.extend([kpts[2*i], kpts[2*i+1], confs[i]])
                    
                    row.extend([head_velocity, angle_velocity, current_angle])
                    
                    writer.writerow(row)
                    total_extracted += 1
                    
        cap.release()
    
    f.close()
    print(f"✅ AI Hub 데이터 처리 완료! (총 {total_extracted} 프레임 추가됨)")
    print(f"💾 데이터셋 위치: {utils.CSV_PATH}")

if __name__ == '__main__':
    run()