# SilverGuard/train.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import utils

def run():
    print("🚀 모델 학습(Training) 시작...")
    utils.ensure_dirs()

    # 1. CSV 데이터 로드
    if not os.path.exists(utils.CSV_PATH):
        print(f"❌ 오류: 데이터 파일이 없습니다. preprocess.py를 먼저 실행하세요.")
        return

    df = pd.read_csv(utils.CSV_PATH)
    print(f"   - 총 데이터 개수: {len(df)}")
    
    # 2. 입력(X)과 정답(y) 분리
    # label과 video_name 컬럼 제거 -> 순수 좌표값만 사용
    X = df.drop(['label', 'video_name'], axis=1)
    y = df['label']

    # 3. 학습용/테스트용 데이터 분리 (80% : 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Random Forest 모델 학습
    print("   - 학습 진행 중...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. 성능 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✨ 모델 정확도: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))

    # 6. 모델 저장
    joblib.dump(model, utils.ML_MODEL_PATH)
    print(f"💾 모델 저장 완료: {utils.ML_MODEL_PATH}")

if __name__ == '__main__':
    run()