import os
import sys
import io
import json # [New]

# [Fix] Windows Unicode Error
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import pandas as pd
import numpy as np
import torch

# ... existing imports ...
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from stgcn import STGCN
import utils

# Config
WINDOW_SIZE = 30
STRIDE = 10
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optimize_thresholds(model):
    """
    Find optimal confidence threshold using User Verified Data (Field Data).
    Instead of relying on a fixed 0.65, we adapt to the user's specific environment.
    """
    # Load ONLY field data for calibration
    print("   ... 사용자 피드백 데이터 로드 중 ...")
    X_field, y_field = load_field_data()
    if len(X_field) == 0:
        print("   ⚠️ 튜닝할 사용자 데이터(오작동/실제낙상)가 부족하여 최적화를 건너뜁니다.")
        return None
        
    # Get confidences
    model.eval()
    fall_probs = []
    
    # Batch processing to avoid OOM if field data is huge (unlikely but safe)
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X_field), batch_size):
            batch_X = X_field[i:i+batch_size]
            inputs = torch.tensor(batch_X, dtype=torch.float32).unsqueeze(-1).permute(0, 3, 1, 2, 4).to(DEVICE)
            outputs = model(inputs) # (N, 2)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            fall_probs.extend(probs[:, 1].cpu().numpy())
            
    fall_probs = np.array(fall_probs)
    
    # Grid Search for Best Threshold (0.40 to 0.95)
    best_t = 0.65
    best_f1 = 0.0
    
    # We prioritize Precision slightly more to avoid False Alarms in real home usage?
    # Or Recall? Falls are critical. Recall is Key.
    # But F1 balances both.
    
    for t in np.arange(0.40, 0.96, 0.01):
        preds = (fall_probs >= t).astype(int)
        f1 = f1_score(y_field, preds, zero_division=0)
        
        # If F1 is equal, prefer higher threshold (Conservative / Less False Alarms)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
        elif f1 == best_f1 and f1 > 0:
            # Tie-breaking: Choose the one closer to default or Higher?
            # Higher threshold is safer against false alarms.
            best_t = max(best_t, t)

    print(f"   ► 사용자 데이터 기준 최적 F1: {best_f1*100:.1f}% (최적 임계값: {best_t:.2f})")
    
    # Update settings.json
    try:
        if os.path.exists(utils.SETTINGS_PATH):
            with open(utils.SETTINGS_PATH, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        else:
            settings = {}
            
        old_t = settings.get("AI_CONFIDENCE", 0.65)
        
        # Apply change if meaningful diff (> 0.02)
        if abs(old_t - best_t) >= 0.01:
            print(f"   💡 [시스템 최적화] AI 민감도 자동 조정: {old_t:.2f} -> {best_t:.2f}")
            settings["AI_CONFIDENCE"] = float(best_t)
            with open(utils.SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            return best_t
        else:
            print(f"   (현재 민감도 {old_t:.2f}가 이미 최적입니다)")
    except Exception as e:
        print(f"   ⚠️ 설정 저장 실패: {e}")
        
    return None



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
    
    # [NEW] Scan Archives (Cumulative Learning)
    process_dir(utils.ARCHIVE_VERIFIED, 1)
    process_dir(utils.ARCHIVE_FALSE, 0)
    
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

    # 🔧 [Step 4] Sensitivity Auto-Tuning
    print("\n🔧 [4/4] 민감도(Threshold) 정밀 튜닝 (Heuristic Optimization)")
    
    # Pick the winner model
    active_model = model if improved else None
    
    # If not improved, try to load old model to tune it
    if not active_model and os.path.exists(model_path):
        try:
             active_model = STGCN(in_channels=3, num_class=2).to(DEVICE)
             active_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except: pass

    if active_model:
        optimize_thresholds(active_model)
        
    # [Step 5] Archive Data (Clean up active folders)
    print("\n🧹 학습 데이터 아카이빙 (데이터 정리)...")
    utils.ensure_dirs()
    
    def archive_files(src_dir, dst_dir):
        if not os.path.exists(src_dir): return
        files = os.listdir(src_dir)
        count = 0
        for f in files:
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f)
            try:
                if os.path.isfile(src):
                    if os.path.exists(dst): os.remove(dst) # Overwrite
                    shutil.move(src, dst)
                    count += 1
            except Exception as e:
                print(f"   ⚠️ 파일 이동 실패 ({f}): {e}")
        if count > 0:
            print(f"   ► {count}개 파일을 아카이브로 이동: {os.path.basename(dst_dir)}")

    archive_files(utils.VERIFIED_DIR, utils.ARCHIVE_VERIFIED)
    archive_files(utils.FALSE_ALARM_DIR, utils.ARCHIVE_FALSE)
    
    print("✅ 모든 작업 완료.")

if __name__ == "__main__":
    train()
