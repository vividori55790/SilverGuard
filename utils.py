# SilverGuard/utils.py
import os

# ==================================================
# [1] 도커 컨테이너 내부 절대 경로 설정
# ==================================================
# 데이터 관련
DATA_DIR = '/app/data'
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
ALERT_DIR = os.path.join(DATA_DIR, 'alert_images')
CSV_PATH = os.path.join(DATA_DIR, 'dataset.csv')

# 모델 관련
MODEL_DIR = '/app/models'
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, 'yolov8n-pose.pt')
ML_MODEL_PATH = os.path.join(MODEL_DIR, 'fall_classifier.pkl')

# ==================================================
# [2] 시스템 설정값
# ==================================================
# 테스트용 영상 파일 이름 (data/videos 안에 있어야 함)
# 실제 배포 시에는 RTSP 주소(예: "rtsp://admin:1234@...")를 넣으세요.
TEST_VIDEO_NAME = 'fall_test.mp4' 

# UR Fall 데이터셋 학습 시, 영상 오른쪽(RGB)만 자를지 여부
CROP_RIGHT_HALF = True 

# 낙상 판단 기준 (초): 넘어진 상태로 이 시간 이상 유지되면 알림
FALL_TIME_THRESHOLD = 5.0 

# ==================================================
# [3] 유틸리티 함수
# ==================================================
def ensure_dirs():
    """필요한 폴더가 없으면 생성"""
    os.makedirs(ALERT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)