import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import utils
from models import FallLSTM

# ==========================================
# [설정] 하이퍼파라미터
# ==========================================
SEQUENCE_LENGTH = 30  # 과거 30프레임(약 1초)을 하나의 시퀀스로 묶음
HIDDEN_SIZE = 64      # LSTM 은닉 노드 수
NUM_LAYERS = 2        # LSTM 레이어 층 수
BATCH_SIZE = 32
EPOCHS = 15           # 학습 반복 횟수
LEARNING_RATE = 0.001
# ==========================================

class FallDataset(Dataset):
    """ PyTorch용 데이터셋 클래스 """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(df, seq_length):
    """
    영상별로 그룹화하여 슬라이딩 윈도우 방식으로 시퀀스 데이터 생성
    Output: (Num_Samples, Sequence_Length, Num_Features)
    """
    sequences = []
    labels = []
    groups = [] 
    
    # 비디오 이름으로 그룹화 (서로 다른 영상의 프레임이 시퀀스로 섞이는 것 방지)
    video_groups = df.groupby('video_name')
    
    # label과 video_name을 제외한 모든 컬럼이 입력 Feature
    feature_cols = [c for c in df.columns if c not in ['label', 'video_name']]
    print(f"   - Feature 차원 수: {len(feature_cols)} (예상: 54)")
    
    for video_name, group in video_groups:
        if len(group) < seq_length: continue
            
        data = group[feature_cols].values
        label = group['label'].values
        
        # 슬라이딩 윈도우 (Sliding Window)
        # 예: [0~29], [1~30], [2~31] ... 이렇게 한 칸씩 밀어가며 데이터 생성
        for i in range(len(data) - seq_length + 1):
            seq = data[i : i+seq_length]
            
            # 시퀀스의 마지막 프레임 라벨을 정답(Target)으로 사용
            target = label[i + seq_length - 1] 
            
            sequences.append(seq)
            labels.append(target)
            groups.append(video_name)
            
    return np.array(sequences), np.array(labels), np.array(groups)

def run():
    print("🚀 LSTM 모델 학습 시작...")
    utils.ensure_dirs()
    
    # 1. 데이터 로드
    if not os.path.exists(utils.CSV_PATH):
        print(f"❌ 데이터 파일이 없습니다: {utils.CSV_PATH}")
        print("   -> preprocess_urfall_velocity.py를 먼저 실행하세요.")
        return
        
    df = pd.read_csv(utils.CSV_PATH)
    # NaN 값 제거 (속도 계산 시 첫 프레임 등에서 발생 가능)
    df = df.dropna()
    
    # 2. 시퀀스 데이터 생성
    print("   - 시퀀스 데이터(Window 30) 생성 중...")
    X, y, groups = create_sequences(df, SEQUENCE_LENGTH)
    
    if len(X) == 0:
        print("❌ 생성된 시퀀스가 없습니다. 데이터셋 크기를 확인하세요.")
        return

    print(f"   - 생성된 시퀀스 형태: {X.shape}") # (N, 30, 54)
    print(f"   - 총 샘플 수: {len(y)} (낙상: {sum(y==1)}, 정상: {sum(y==0)})")
    
    # 3. 데이터 분할 (GroupKFold)
    # 같은 비디오에 나온 프레임이 Train과 Test에 섞이면 안 되므로 그룹 기반 분할 사용
    # 이를 통해 Data Leakage(정답 유출)를 방지하고 일반화 성능을 높임
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"   - 학습용: {len(X_train)}개, 테스트용: {len(X_test)}개")
    
    # 4. 데이터 정규화 (Standard Scaling) - 필수
    # LSTM은 값의 크기에 민감하므로 평균 0, 분산 1로 조정
    scaler = StandardScaler()
    
    # 3차원 -> 2차원 변환 후 스케일링 -> 다시 3차원 복구
    N_train, L, F = X_train.shape
    X_train_reshaped = X_train.reshape(-1, F)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(N_train, L, F)
    
    N_test, L, F = X_test.shape
    X_test_reshaped = X_test.reshape(-1, F)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(N_test, L, F)
    
    # 추론 시 사용하기 위해 스케일러 저장
    scaler_path = os.path.join(utils.MODEL_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"   - 스케일러 저장됨: {scaler_path}")
    
    # 5. DataLoader 설정
    train_dataset = FallDataset(X_train_scaled, y_train)
    test_dataset = FallDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 6. 모델 및 학습 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FallLSTM(input_size=F, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 7. 학습 루프
    print(f"   - 학습 시작 (Device: {device}, Epochs: {EPOCHS})")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # 1 Epoch 끝날 때마다 출력
        print(f"     Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f}")
        
    # 8. 모델 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f"\n✨ 최종 검증 정확도: {acc:.2f}%")
    
    # 9. 모델 저장
    save_path = os.path.join(utils.MODEL_DIR, 'fall_lstm.pth')
    torch.save(model.state_dict(), save_path)
    print(f"💾 모델 저장 완료: {save_path}")

if __name__ == '__main__':
    run()