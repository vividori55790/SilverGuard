import cv2
import csv
import os
import math
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import utils

# ==========================================
# [설정] UR Fall 데이터셋 경로
# ==========================================
DIR_FALL = '/app/data/urfall/fall'
DIR_ADL = '/app/data/urfall/adl'
# ==========================================

def calculate_angle(p1, p2):
    """ 두 점(p1, p2) 사이의 각도 계산 (수직선 기준) """
    # p1: 어깨 중점, p2: 골반 중점
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # 라디안 -> 도(degree) 변환. 서 있을 때 0도(또는 180도), 누우면 90도 근처
    return abs(math.degrees(math.atan2(dx, dy)))

def process_folder(folder_path, label, writer, model):
    if not os.path.exists(folder_path):
        print(f"⚠️ 폴더 없음: {folder_path}")
        return 0

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi'))]
    count = 0
    
    for filename in tqdm(video_files, desc=f"Processing {label}"):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        
        # [이전 프레임 정보 저장용 변수]
        prev_head_y = None
        prev_angle = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # UR Fall 특성상 오른쪽 RGB만 사용 (필요 시 수정)
            h, w, _ = frame.shape
            rgb_frame = frame[:, w//2:]
            
            # YOLO 추론
            results = model(rgb_frame, verbose=False)
            
            if results[0].keypoints is not None:
                # Keypoints 정규화 좌표 (0~1)
                kpts = results[0].keypoints.xyn[0].cpu().numpy()
                confs = results[0].keypoints.conf[0].cpu().numpy()
                
                # 데이터가 온전한지 확인 (17개 키포인트)
                if len(kpts) == 17:
                    # ------------------------------------------------
                    # [핵심] 파생 변수(Feature Engineering) 생성
                    # ------------------------------------------------
                    
                    # 1. 머리(코)의 Y좌표 (Keypoint 0)
                    head_y = kpts[0][1]
                    
                    # 2. 몸통 각도 (어깨 중점 ~ 골반 중점)
                    # 어깨: 5(L), 6(R) / 골반: 11(L), 12(R)
                    shoulder_mid = (kpts[5] + kpts[6]) / 2
                    hip_mid = (kpts[11] + kpts[12]) / 2
                    current_angle = calculate_angle(shoulder_mid, hip_mid)

                    # 3. 변화량 계산 (속도)
                    if prev_head_y is not None:
                        # 머리가 아래로 떨어지는 속도 (Y좌표 증가량)
                        # *30을 하는 이유: 대략 30fps 기준 초당 변화율처럼 보이게 스케일링
                        head_velocity = (head_y - prev_head_y) * 30 
                        
                        # 몸통 각도 변화 속도
                        angle_velocity = (current_angle - prev_angle) * 30
                    else:
                        head_velocity = 0
                        angle_velocity = 0

                    # 상태 업데이트
                    prev_head_y = head_y
                    prev_angle = current_angle

                    # ------------------------------------------------
                    # [CSV 저장] 
                    # 좌표(51개) + 속도정보(2개) + 각도정보(1개) = 총 54개 피처
                    # ------------------------------------------------
                    row = [label, filename]
                    
                    # (1) 기본 좌표 및 신뢰도
                    kpts_flat = kpts.flatten()
                    row.extend(kpts_flat)      # x, y 좌표들
                    row.extend(confs)          # confidence 값들 (뒤에 몰아서 넣거나 순서대로 넣거나 통일 필요)
                    # 여기서는 편의상 x,y만 평평하게 넣고 뒤에 추가 피처를 붙이겠습니다.
                    # -> train.py와 맞추기 위해 x,y,c 순서로 다시 정리
                    
                    dataset_row = [label, filename]
                    for i in range(17):
                        dataset_row.extend([kpts[i][0], kpts[i][1], confs[i]]) # x, y, c
                    
                    # (2) **중요** 파생 피처 추가
                    dataset_row.append(head_velocity)   # 머리 낙하 속도
                    dataset_row.append(angle_velocity)  # 몸통 회전 속도
                    dataset_row.append(current_angle)   # 현재 몸통 각도
                    
                    writer.writerow(dataset_row)
                    count += 1
                    
        cap.release()
    return count

def run():
    print("🚀 [UR Fall] 속도 기반 데이터 전처리 시작")
    utils.ensure_dirs()
    model = YOLO(utils.YOLO_MODEL_PATH)
    
    f = open(utils.CSV_PATH, 'w', newline='')
    writer = csv.writer(f)
    
    # 헤더 작성
    header = ['label', 'video_name']
    for i in range(17): header.extend([f'x{i}', f'y{i}', f'c{i}'])
    
    # 추가된 피처 헤더
    header.extend(['head_velocity', 'angle_velocity', 'torso_angle'])
    
    writer.writerow(header)

    # Fall -> Label 1
    c1 = process_folder(DIR_FALL, 1, writer, model)
    # ADL -> Label 0
    c2 = process_folder(DIR_ADL, 0, writer, model)
    
    f.close()
    print(f"✅ 완료. 데이터셋 생성됨: {utils.CSV_PATH}")

if __name__ == '__main__':
    run()