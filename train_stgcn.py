import os
import sys
import io

# [Fix] Windows Unicode Error
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # [New]
from tqdm import tqdm
from stgcn import STGCN
import utils

# Config
WINDOW_SIZE = 30 # 1 second approx
STRIDE = 10      # Overlap for data augmentation
BATCH_SIZE = 32
EPOCHS = 30      # Increased from 10
LR = 0.001       # Reduced slightly for stability with 30 epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_field_data():
    """Load user-verified samples from VERIFIED_DIR and FALSE_ALARM_DIR"""
    field_data = []
    field_labels = []
    
    # Helper to process a directory
    def process_dir(dir_path, label_val):
        if not os.path.exists(dir_path): return
        
        files = [f for f in os.listdir(dir_path) if f.lower().endswith('.npy')]
        for f in files:
            path = os.path.join(dir_path, f)
            try:
                arr = np.load(path)
                
                # Robust shape handling
                # Expected: (T, 17, 3)
                if arr.ndim == 2:
                    # Possibly (T, 51) or (T, 34)
                    if arr.shape[1] == 51:
                         arr = arr.reshape(-1, 17, 3)
                    elif arr.shape[1] == 34: # No confidence
                         # Pad confidence with 1.0 or 0.0?
                         # Better to assume (T, 17, 2) -> (T, 17, 3) with c=0.5
                         T = len(arr)
                         arr = arr.reshape(T, 17, 2)
                         # Add confidence channel
                         confs = np.full((T, 17, 1), 0.5)
                         arr = np.concatenate((arr, confs), axis=2)
                
                if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] != 3:
                    continue # Skip invalid shapes
                    
                T = len(arr)
                if T < WINDOW_SIZE: continue
                
                # Extract windows
                # Use smaller stride for field data to maximize usage? Or same STRIDE?
                # Use same STRIDE for consistency.
                for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
                    end = start + WINDOW_SIZE
                    window = arr[start:end]
                    
                    field_data.append(window)
                    field_labels.append(label_val)
                    
            except Exception as e:
                print(f"⚠️ Failed to load field data {f}: {e}")

    print("🔍 Scanning User Verified Data...")
    process_dir(utils.VERIFIED_DIR, 1)    # Verified Falls
    process_dir(utils.FALSE_ALARM_DIR, 0) # Verified Normal
    
    if len(field_data) == 0:
        return np.array([]), np.array([])
        
    return np.array(field_data), np.array(field_labels)

def load_data():
    print("🚀 Loading CSV files...")
    label_df = pd.read_csv(os.path.join(utils.DATA_DIR, 'labeling_work.csv'))
    kp_df = pd.read_csv(os.path.join(utils.DATA_DIR, 'raw_keypoints_colab_2.csv'))
    
    data_list = []
    label_list = []
    importance_list = [] # [New] Weight for importance sampling
    
    grouped = kp_df.groupby(['video_name', 'track_id'])
    
    print("🔄 Processing sequences...")
    for (video_name, track_id), group in tqdm(grouped):
        video_info = label_df[label_df['filename'] == video_name]
        
        if len(video_info) == 0:
            continue
            
        group = group.sort_values('frame_idx')
        frame_labels = np.zeros(len(group))
        
        if video_info.iloc[0]['is_fall'] == 1:
            start_sec = video_info.iloc[0]['start_sec']
            end_sec = video_info.iloc[0]['end_sec']
            times = group['time_sec'].values
            mask = (times >= start_sec) & (times <= end_sec)
            frame_labels[mask] = 1
            
        feature_cols = []
        for i in range(17):
            feature_cols.extend([f'x{i}', f'y{i}', f'c{i}'])
            
        features = group[feature_cols].values
        features = features.reshape(-1, 17, 3) 
        
        num_frames = len(features)
        if num_frames < WINDOW_SIZE:
            continue
            
        for start in range(0, num_frames - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            window_data = features[start:end]
            window_labels = frame_labels[start:end]
            
            # Label Assignment
            if np.sum(window_labels) >= 5:
                final_label = 1
            else:
                final_label = 0
                
            data_list.append(window_data)
            label_list.append(final_label)
            importance_list.append(1.0) # Standard Data Weight
            
            data_list.append(window_data)
            label_list.append(final_label)
            importance_list.append(1.0) # Augmentation Weight
            
    # [NEW] Load Field Data (Self-Learning) -> High Importance
    field_X, field_y = load_field_data()
    if len(field_X) > 0:
        print(f"🌟 Found {len(field_X)} new field data samples from actual operation!")
        field_weights = np.full(len(field_X), 10.0) # [Booster] 10x weight for user feedback
        
        # Concatenate
        if len(data_list) > 0:
            data_list = np.concatenate((data_list, field_X), axis=0)
            label_list = np.concatenate((label_list, field_y), axis=0)
            importance_list = np.concatenate((importance_list, field_weights), axis=0)
        else:
            data_list = field_X
            label_list = field_y
            importance_list = field_weights
            
    return np.array(data_list), np.array(label_list), np.array(importance_list)

# ... (load_field_data remains same) ...
# ... (FallDataset remains same) ...

def evaluate_metrics(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            # Inputs: (N, T, V, C) -> (N, C, T, V, M)
            inputs = inputs.float().unsqueeze(-1).permute(0, 3, 1, 2, 4).to(DEVICE)
            targets = targets.long().to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds) * 100
    p = precision_score(all_targets, all_preds, zero_division=0) * 100
    r = recall_score(all_targets, all_preds, zero_division=0) * 100
    f1 = f1_score(all_targets, all_preds, zero_division=0) * 100
        
    return acc, p, r, f1

def train():
    X, y, weights = load_data()
    if len(X) == 0:
        print("❌ No data found.")
        return

    print(f"📊 Dataset Shape: {X.shape}, Labels: {y.shape}")
    
    # Split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dataset = FallDataset(X_train, y_train)
    test_dataset = FallDataset(X_test, y_test)
    
    # Weighted Sampler
    class_counts = np.bincount(y_train)
    if len(class_counts) < 2: class_counts = np.array([1, 1]) # Safety
    class_weights = 1. / class_counts 
    sample_weights = class_weights[y_train] * w_train # Apply Priority Weights
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 🔎 [Step 1] Evaluate Existing Model (Baseline)
    print("\n------------------------------------------------")
    print("🔎 [1/3] 기존 모델 성능 검증 (Baseline Check)")
    old_acc, old_p, old_r, old_f1 = 0, 0, 0, 0
    model_path = os.path.join(utils.MODEL_DIR, 'stgcn_fall.pth')
    
    if os.path.exists(model_path):
        try:
            old_model = STGCN(in_channels=3, num_class=2).to(DEVICE)
            old_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            old_acc, old_p, old_r, old_f1 = evaluate_metrics(old_model, test_loader)
            print(f"   ► 기존 모델: Acc={old_acc:.2f}%, F1={old_f1:.2f}% (Recall={old_r:.2f}%)")
        except:
            print("   ⚠️ 기존 모델 로드 실패 (신규 학습 진행)")

    # 🚀 [Step 2] Train New Model
    print("\n🚀 [2/3] 신규 모델 학습 시작...")
    model = STGCN(in_channels=3, num_class=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4) # Regularization

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", ncols=80):
            inputs = inputs.float().unsqueeze(-1).permute(0, 3, 1, 2, 4).to(DEVICE)
            targets = targets.long().to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        if (epoch+1) % 5 == 0:
            print(f"   Ep {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={100*correct/total:.2f}%")

    # 🔎 [Step 3] Evaluate New Model
    print("\n🔎 [3/3] 신규 모델 성능 검증 (New Model Check)")
    new_acc, new_p, new_r, new_f1 = evaluate_metrics(model, test_loader)
    print(f"   ► 신규 모델: Acc={new_acc:.2f}%, F1={new_f1:.2f}% (Recall={new_r:.2f}%)")
    
    # ⚖️ [Final Comparison]
    print("\n⚖️ [최종 결과 리포트]")
    print(f"   - 기존 F1: {old_f1:.2f}% vs 신규 F1: {new_f1:.2f}%")
    print(f"   - 기존 Acc: {old_acc:.2f}% vs 신규 Acc: {new_acc:.2f}%")
    
    # Improvement Condition:
    # 1. Old doesn't exist (0) -> Save
    # 2. New F1 is better OR (F1 same and Acc better)
    # Bias towards New if metrics are identical (learning new data is good) 
    improved = False
    if old_f1 == 0:
        improved = True
        print("✅ 초기 모델 생성 완료.")
    elif new_f1 >= old_f1 and new_acc >= old_acc:
        improved = True
        print("✅ 성능 향상 (또는 동등) 확인! 정밀도/재현율 개선.")
    elif new_f1 > old_f1:
        improved = True
        print("✅ F1 점수(낙상 감지 능력) 향상 확인!")
        
    if improved:
        print("💾 새로운, 더 똑똑한 모델을 채택하고 저장합니다...")
        torch.save(model.state_dict(), model_path)
        print("🎉 [성공] 재학습 및 모델 교체 완료.")
    else:
        print("⚠️ 성능 향상이 관찰되지 않았습니다. 기존 모델을 유지합니다.")
        print("   (Tip: 검증된 데이터를 더 많이 추가하면 성능이 오를 수 있습니다.)")

if __name__ == "__main__":
    train()
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        acc = 100 * correct / total
        print(f"   - Loss: {total_loss/len(train_loader):.4f}, Train Acc: {acc:.2f}%")
        
        # Validation
        val_acc = evaluate(model, test_loader)
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            # Temp save
            torch.save(model.state_dict(), "temp_best.pth")
            
    print(f"🏁 Training Finished. New Best Acc: {best_acc:.2f}% (Old: {current_acc:.2f}%)")
    
    if best_acc >= current_acc: # Allow equal if it trained on more data
        print("✅ New model is better or equal. Updating system model.")
        if os.path.exists("temp_best.pth"):
            # Load bytes and save to final
            try:
                state = torch.load("temp_best.pth")
                torch.save(state, model_path)
                print("💾 Model Updated Successfully.")
            except: pass
    else:
        print("⚠️ New model performance is worse. Discarding changes.")
        
    if os.path.exists("temp_best.pth"):
        os.remove("temp_best.pth")

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.unsqueeze(-1).to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

if __name__ == '__main__':
    train()
