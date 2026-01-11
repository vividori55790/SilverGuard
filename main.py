# SilverGuard/main.py
import cv2
import joblib
import numpy as np
import time
import os
import datetime
from ultralytics import YOLO
import utils

def main():
    print("🚀 낙상 감지 시스템(Main) 가동...")
    utils.ensure_dirs()

    # 1. 모델 로드
    print("   - 모델 로딩 중...")
    yolo_model = YOLO(utils.YOLO_MODEL_PATH)
    
    if not os.path.exists(utils.ML_MODEL_PATH):
        print("❌ 오류: 분류기 모델이 없습니다. train.py를 먼저 실행하세요.")
        return
    classifier = joblib.load(utils.ML_MODEL_PATH)

    # 2. 영상 소스 설정
    # 파일이 있으면 파일 사용, 없으면 설정된 값(RTSP 등) 사용
    test_video_path = os.path.join(utils.VIDEO_DIR, utils.TEST_VIDEO_NAME)
    video_source = test_video_path if os.path.exists(test_video_path) else 0
    
    print(f"   - 영상 소스: {video_source}")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("❌ 영상을 열 수 없습니다.")
        return

    # 변수 초기화
    fall_start_time = None
    is_fall_state = False
    
    print("✅ 감시를 시작합니다. (로그 모니터링 중...)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("영상 종료. (테스트용이므로 종료합니다)")
            break
            # 실제 배포 시에는 아래 코드 주석 해제하여 무한 루프
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # continue

        # [주의] 테스트 영상이 UR Fall 데이터라면 잘라야 하고,
        # 일반 웹캠/CCTV라면 자르지 말아야 합니다.
        # 여기서는 테스트용으로 '자르기'를 활성화해둡니다. (필요 시 주석 처리)
        if utils.CROP_RIGHT_HALF:
             h, w, _ = frame.shape
             frame = frame[:, w//2:]

        # 성능 최적화: 3프레임당 1번만 추론 (필요 시 활성화)
        # if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 3 != 0: continue

        # YOLO 추론
        results = yolo_model(frame, verbose=False, stream=True)

        for r in results:
            if r.keypoints is None: continue

            # 감지된 모든 사람에 대해
            boxes = r.boxes.xyxy.cpu().numpy()
            keypoints_list = r.keypoints.xyn.cpu().numpy()
            confs_list = r.keypoints.conf.cpu().numpy()

            for i, kpts in enumerate(keypoints_list):
                confs = confs_list[i]
                
                # 데이터 전처리 (학습할 때와 똑같은 형태로 변환)
                row_data = []
                for x, y, c in zip(kpts[0::2], kpts[1::2], confs):
                    row_data.extend([x, y, c])
                
                # 입력 데이터가 유효한지 확인
                if len(row_data) == 51: # 17 * 3
                    # 머신러닝 예측 (2차원 배열 입력 필요)
                    pred = classifier.predict([row_data])[0]
                    
                    # 시각화 (도커라 화면엔 안 나오지만 저장된 이미지 확인용)
                    bbox = boxes[i]
                    color = (0, 0, 255) if pred == 1 else (0, 255, 0)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    label_text = "FALL" if pred == 1 else "Normal"
                    cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # [낙상 로직]
                    if pred == 1: # 낙상 감지
                        if not is_fall_state:
                            is_fall_state = True
                            fall_start_time = time.time()
                            print(f"⚠️ [{datetime.datetime.now().strftime('%H:%M:%S')}] 낙상 의심 동작 감지!")
                        
                        # 지속 시간 체크
                        elapsed = time.time() - fall_start_time
                        if elapsed >= utils.FALL_TIME_THRESHOLD:
                            # 진짜 사고로 판단 -> 이미지 저장 및 알림
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = os.path.join(utils.ALERT_DIR, f"FALL_{timestamp}.jpg")
                            
                            cv2.imwrite(save_path, frame)
                            print(f"🚨 [긴급] 낙상 사고 확정! 이미지 저장됨: {save_path}")
                            
                            # 알림 반복 방지 (타이머 리셋)
                            fall_start_time = time.time() 
                    else:
                        # 정상이면 타이머 초기화 (단, 아주 잠깐 일어난건 무시하도록 로직 추가 가능)
                        is_fall_state = False
                        fall_start_time = None

    cap.release()
    print("시스템 종료.")

if __name__ == '__main__':
    main()