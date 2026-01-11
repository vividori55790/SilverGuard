import cv2
import csv
import os
import math
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import utils  # SilverGuard/utils.py

# ==========================================
# [설정] UR Fall 데이터셋 경로 (도커 볼륨 기준)
# ==========================================
DIR_FALL = '/app/data/urfall/fall'
DIR_ADL = '/app/data/urfall/adl'
# ==========================================

def calculate_angle(p1, p2):
    """ 
    두 점(p1, p2) 사이의 각도 계산 (수직선 기준) 
    p1: 상체(어깨), p2: 하체(골반)
    Return: 0~180도 (서있으면 0 or 180, 누우면 90 근처)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # 라디안 -> 도(degree) 변환
    return abs(math.degrees(math.atan2(dx, dy)))

def process_folder(folder_path, label, writer, model):
    """
    특정 폴더의 영상들을 처리하여 CSV에 기록하는 함수
    """
    if not os.path.exists(folder_path):
        print(f"⚠️ 폴더 없음: {folder_path} (건너뜀)")
        return 0

    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi'))]
    count = 0
    
    # 진행 상황 표시 (tqdm 라이브러리 사용)
    desc_text = f"Label {label} ({'Fall' if label==1 else 'ADL'}) 처리 중"
    for filename in tqdm(video_files, desc=desc_text):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        
        # [이전 프레임 정보 저장용 변수] - 속도 계산을 위해 필요
        prev_head_y = None
        prev_angle = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # UR Fall 데이터셋 특성상 영상이 좌우로 나뉘어 있는 경우 오른쪽(RGB)만 사용
            # 일반 웹캠 영상이라면 utils.CROP_RIGHT_HALF를 False로 설정해야 함
            if utils.CROP_RIGHT_HALF:
                h, w, _ = frame.shape
                frame = frame[:, w//2:]
            
            # YOLO 추론 (Pose Estimation)
            # verbose=False: 불필요한 로그 출력 억제
            results = model(frame, verbose=False)
            
            # 사람이 감지되지 않으면 건너뜀
            if results[0].keypoints is None or len(results[0].keypoints) == 0:
                continue
                
            # 첫 번째 사람의 데이터만 사용 (단순화)
            # xyn: 정규화된 좌표 (0~1 범위)
            kpts = results[0].keypoints.xyn[0].cpu().numpy() 
            confs = results[0].keypoints.conf[0].cpu().numpy()
            
            # 데이터 유효성 체크 (17개 키포인트가 모두 있어야 함)
            if len(kpts) == 17:
                # ------------------------------------------------
                # [핵심] 파생 변수(Feature Engineering) 생성
                # ------------------------------------------------
                
                # 1. 머리(코, Index 0)의 Y좌표
                head_y = kpts[0][1]
                
                # 2. 몸통 각도 (어깨 중점 ~ 골반 중점)
                # 어깨: 5(Left), 6(Right) / 골반: 11(Left), 12(Right)
                shoulder_mid = (kpts[5] + kpts[6]) / 2
                hip_mid = (kpts[11] + kpts[12]) / 2
                current_angle = calculate_angle(shoulder_mid, hip_mid)

                # 3. 변화량(속도) 계산
                if prev_head_y is not None:
                    # 머리가 아래로 떨어지는 속도 (Y좌표 증가량)
                    # *30: 프레임 간 차이를 초당 변화율로 스케일링 (30fps 가정)
                    head_velocity = (head_y - prev_head_y) * 30 
                    
                    # 몸통 각도 변화 속도
                    angle_velocity = (current_angle - prev_angle) * 30
                else:
                    # 첫 프레임은 비교 대상이 없으므로 속도 0
                    head_velocity = 0
                    angle_velocity = 0

                # 상태 업데이트 (현재 값을 다음 프레임의 과거 값으로 저장)
                prev_head_y = head_y
                prev_angle = current_angle

                # ------------------------------------------------
                # [CSV 저장] 
                # 데이터 구조: [Label, VideoName] + [x,y,c * 17] + [head_vel, angle_vel, angle]
                # 총 Feature 개수: 51 + 3 = 54개
                # ------------------------------------------------
                row = [label, filename]
                
                # (1) 기본 좌표 및 신뢰도 (x0, y0, c0, x1, y1, c1 ...)
                for i in range(17):
                    row.extend([kpts[i][0], kpts[i][1], confs[i]]) 
                
                # (2) 파생 피처 추가
                row.append(head_velocity)   # 머리 낙하 속도
                row.append(angle_velocity)  # 몸통 회전 속도
                row.append(current_angle)   # 현재 몸통 각도
                
                writer.writerow(row)
                count += 1
                    
        cap.release()
    return count

def run():
    print("🚀 [UR Fall] 속도(Velocity) 기반 데이터 전처리 시작")
    utils.ensure_dirs()
    
    # YOLO 모델 로드
    print(f"   - YOLO 모델 로드 중: {utils.YOLO_MODEL_PATH}")
    try:
        model = YOLO(utils.YOLO_MODEL_PATH)
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패. models 폴더에 yolov8n-pose.pt가 있는지 확인하세요.\n에러: {e}")
        return
    
    # CSV 파일 열기 (쓰기 모드)
    f = open(utils.CSV_PATH, 'w', newline='')
    writer = csv.writer(f)
    
    # CSV 헤더 작성
    header = ['label', 'video_name']
    for i in range(17): header.extend([f'x{i}', f'y{i}', f'c{i}'])
    
    # 추가된 피처 헤더
    header.extend(['head_velocity', 'angle_velocity', 'torso_angle'])
    writer.writerow(header)

    # 1. 낙상 폴더 처리 (Label 1)
    c1 = process_folder(DIR_FALL, 1, writer, model)
    
    # 2. 정상 폴더 처리 (Label 0)
    c2 = process_folder(DIR_ADL, 0, writer, model)
    
    f.close()
    print(f"✅ 전처리 완료!")
    print(f"   - 저장 위치: {utils.CSV_PATH}")
    print(f"   - 총 데이터 포인트: {c1 + c2} 프레임")

if __name__ == '__main__':
    run()