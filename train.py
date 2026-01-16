import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from model import LightSTGCN  # model.py에서 클래스 임포트

# 도커 경로 설정
DATA_DIR = '/app/data'
X_PATH = os.path.join(DATA_DIR, 'X_final_aug.npy')
Y_PATH = os.path.join(DATA_DIR, 'y_final_aug.npy')
MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'final_light_stgcn.pth')

BATCH_SIZE = 16
EPOCHS = 70
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def train_final():
    print(f"🚀 [Train] Light ST-GCN 학습 시작 (Device: {DEVICE})")
    
    if not os.path.exists(X_PATH):
        print("❌ 전처리된 데이터가 없습니다. preprocess.py가 먼저 실행되어야 합니다.")
        return

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    # Feature Selection: (x, y, conf, vx, vy, ax, ay) = 7 features * 17 joints = 119
    feature_dim = 17 * 7
    X = X[:, :, :feature_dim]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    class_counts = np.bincount(y_train)
    weights = 1. / (class_counts + 1e-6) # 0으로 나누기 방지
    samples_weights = [weights[t] for t in y_train]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(FallDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(FallDataset(X_val, y_val), batch_size=BATCH_SIZE)
    
    model = LightSTGCN(in_channels=7, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            # MixUp
            if np.random.random() > 0.5:
                lam = np.random.beta(1.0, 1.0)
                index = torch.randperm(inputs.size(0)).to(DEVICE)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, targets) + (1 - lam) * criterion(outputs, targets[index])
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        model.eval()
        preds_arr, targets_arr = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                preds_arr.extend(preds.cpu().numpy())
                targets_arr.extend(targets.cpu().numpy())
        
        val_f1 = f1_score(targets_arr, preds_arr, average='macro')
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("   --> ⭐ Model Saved!")

    print(f"\n✅ 학습 완료. 저장 경로: {MODEL_SAVE_PATH}")
    print(classification_report(targets_arr, preds_arr, target_names=['Normal', 'Fall'], zero_division=0))

if __name__ == "__main__":
    train_final()