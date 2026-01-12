# download_model.py
from ultralytics import YOLO
import os

# 현재 파일(SilverGuard 폴더 등) 기준 models 폴더 경로 설정
# 이 코드를 SilverGuard 폴더 안에서 실행한다고 가정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 폴더가 없으면 생성
os.makedirs(MODELS_DIR, exist_ok=True)

model_name = 'yolo11s-pose.pt'
save_path = os.path.join(MODELS_DIR, model_name)

print(f"⬇️ {model_name} 다운로드 및 저장 위치 설정: {save_path}")

# 모델 다운로드 (로드 시 자동 다운로드됨)
model = YOLO(model_name)

# 다운로드된 파일을 models 폴더로 이동 (기본적으로 현재 폴더에 받아짐)
if os.path.exists(model_name):
    os.replace(model_name, save_path)
    print(f"✅ 모델 이동 완료: {save_path}")
else:
    # 이미 models 폴더에 있을 경우 등
    print(f"✅ 모델이 준비되었습니다.")