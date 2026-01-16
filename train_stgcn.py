import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
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

def load_data():
    print("🚀 Loading CSV files...")
    label_df = pd.read_csv(os.path.join(utils.DATA_DIR, 'labeling_work.csv'))
    kp_df = pd.read_csv(os.path.join(utils.DATA_DIR, 'raw_keypoints_colab_2.csv'))
    
    data_list = []
    label_list = []
    
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
            
            # Label Assignment: Lower threshold to capture more fall phases
            # If > 5 frames (approx 0.16s) are labeled fall, consider it a Fall window
            if np.sum(window_labels) >= 5:
                final_label = 1
            else:
                final_label = 0
                
            data_list.append(window_data)
            label_list.append(final_label)
            
    return np.array(data_list), np.array(label_list)

class FallDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx] 
        sample = sample.permute(2, 0, 1) 
        label = self.labels[idx]
        return sample, label

def train():
    X, y = load_data()
    print(f"📊 Dataset Shape: {X.shape}, Labels: {y.shape}")
    print(f"   - Fall samples: {np.sum(y==1)}")
    print(f"   - Normal samples: {np.sum(y==0)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = FallDataset(X_train, y_train)
    test_dataset = FallDataset(X_test, y_test)
    
    # ⚖️ Handle Class Imbalance with WeightedRandomSampler
    # Calculate weights for each class
    class_counts = np.bincount(y_train)
    print(f"   - Train counts: Normal={class_counts[0]}, Fall={class_counts[1]}")
    
    # Weight = 1 / count
    class_weights = 1. / class_counts 
    # Assign a weight to each sample corresponding to its class
    sample_weights = class_weights[y_train]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Use sampler in DataLoader (shuffle must be False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = STGCN(in_channels=3, num_class=2).to(DEVICE)
    # Since we use a balanced sampler, we can use standard CrossEntropy
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4) # Added weight_decay
    
    # Training Loop
    print("🚀 Start Training (Imbalance fix enabled)...")
    model.train()
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        # Train
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs = inputs.unsqueeze(-1).to(DEVICE)
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
        print(f"   - Val Acc: {val_acc:.2f}%")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(utils.MODEL_DIR, 'stgcn_fall.pth')
            torch.save(model.state_dict(), save_path)
    
    print(f"💾 Best Model saved with Val Acc: {best_acc:.2f}%")

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
