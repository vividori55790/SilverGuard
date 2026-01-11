# SilverGuard/preprocess.py
import cv2
import csv
import os
import torch
from ultralytics import YOLO
from tqdm import tqdm
import utils  # utils.py 임포트

def run():
    print("🚀 데이터 전처리(Preprocess) 시작...")
    utils.ensure_dirs()

    # 1. YOLO 모델 로드 (없으면 자동 다운로드)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   - 사용 장치: {device}")
    model = YOLO(utils.YOLO_MODEL_PATH)

    # 2. CSV 파일 생성 및 헤더 작성
    f = open(utils.CSV_PATH, 'w', newline='')
    writer = csv.writer(f)
    
    # 헤더: 라벨(0/1) + 비디오명 + 17개 관절의 (x, y, confidence)
    header = ['label', 'video_name']
    for i in range(17):
        header.extend([f'x{i}', f'y{i}', f'c{i}'])
    writer.writerow(header)

    # 3. 영상 파일 목록 가져오기
    if not os.path.exists(utils.VIDEO_DIR):
        print(f"❌ 오류: 비디오 폴더가 없습니다 ({utils.VIDEO_DIR})")
        return

    video_files = [f for f in os.listdir(utils.VIDEO_DIR) if f.endswith(('.mp4', '.avi'))]
    print(f"   - 발견된 영상: {len(video_files)}개")

    # 4. 각 영상 처리
    for filename in tqdm(video_files, desc="Processing"):
        video_path = os.path.join(utils.VIDEO_DIR, filename)
        cap = cv2.VideoCapture(video_path)
        
        # 라벨링 규칙: 파일명에 'fall'이 포함되면 1(낙상), 아니면 0(일상)
        label = 1 if 'fall' in filename.lower() else 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # UR Fall 데이터셋 특화 전처리 (오른쪽 절반만 사용)
            if utils.CROP_RIGHT_HALF:
                h, w, _ = frame.shape
                frame = frame[:, w//2:]
            
            # YOLO 추론 (Verbose=False로 로그 생략)
            results = model(frame, verbose=False, device=device)
            
            # 사람이 감지된 경우
            if results[0].keypoints is not None:
                # xyn: 정규화된 좌표 (0~1)
                kpts = results[0].keypoints.xyn[0].cpu().numpy().flatten()
                confs = results[0].keypoints.conf[0].cpu().numpy().flatten()
                
                # 데이터가 비어있지 않은지 확인
                if len(kpts) == 34: # 17개 점 * (x,y) = 34
                    row = [label, filename]
                    # x, y, conf 순서로 묶어서 저장
                    for x, y, c in zip(kpts[0::2], kpts[1::2], confs):
                        row.extend([x, y, c])
                    writer.writerow(row)
                    
        cap.release()
    
    f.close()
    print(f"✅ 완료! 데이터가 저장되었습니다: {utils.CSV_PATH}")

if __name__ == '__main__':
    run()