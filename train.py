import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import utils

def run():
    print("🚀 모델 학습 (Velocity Feature 포함) 시작...")
    
    if not os.path.exists(utils.CSV_PATH):
        print("❌ 데이터 파일이 없습니다.")
        return

    df = pd.read_csv(utils.CSV_PATH)
    
    # 결측치(NaN) 제거 (첫 프레임은 속도 계산 불가라 0이거나 NaN일 수 있음)
    df = df.dropna()
    
    print(f"   - 총 데이터 개수: {len(df)}")
    
    # Feature와 Target 분리
    # label, video_name을 제외한 모든 컬럼이 입력값(X)
    X = df.drop(['label', 'video_name'], axis=1)
    y = df['label']
    
    # 그룹(영상) 정보: 같은 영상의 프레임이 Train/Test에 섞이지 않게 분리
    groups = df['video_name']
    
    # GroupKFold를 이용한 데이터 분할 (Data Leakage 방지)
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"   - 학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")

    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n✨ 모델 정확도: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    # [중요] 어떤 피처가 낙상 판단에 중요한지 확인
    print("\n🔍 Feature Importance (상위 5개):")
    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]
    
    for i in range(5):
        idx = sorted_idx[i]
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # 모델 저장
    joblib.dump(model, utils.ML_MODEL_PATH)
    print(f"💾 모델 저장 완료: {utils.ML_MODEL_PATH}")

if __name__ == '__main__':
    run()