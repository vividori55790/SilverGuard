
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
from stgcn import STGCN
import utils

# Config (Must match training config)
WINDOW_SIZE = 30 # 1 second approx
STRIDE = 10      # Overlap for data augmentation
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    print("🚀 Loading CSV files...")
    label_df = pd.read_csv(os.path.join(utils.DATA_DIR, 'labeling_work.csv'))
    kp_df = pd.read_csv(os.path.join(utils.DATA_DIR, 'raw_keypoints_colab_2.csv'))
    
    # Process each video sequence
    data_list = []
    label_list = []
    
    grouped = kp_df.groupby(['video_name', 'track_id'])
    
    print("🔄 Processing sequences...")
    for (video_name, track_id), group in tqdm(grouped):
        # Find label info
        video_info = label_df[label_df['filename'] == video_name]
        
        if len(video_info) == 0:
            continue
            
        group = group.sort_values('frame_idx')
        
        # Determine labels for each frame in this group
        # Default 0
        frame_labels = np.zeros(len(group))
        
        # If video is fall (is_fall == 1)
        if video_info.iloc[0]['is_fall'] == 1:
            start_sec = video_info.iloc[0]['start_sec']
            end_sec = video_info.iloc[0]['end_sec']
            
            # Check timestamps
            times = group['time_sec'].values
            # Label frames within interval as 1
            mask = (times >= start_sec) & (times <= end_sec)
            frame_labels[mask] = 1
            
        # Extract features: (Frames, 17, 3)
        # Columns x0,y0,c0 ... x16,y16,c16
        # Construct array
        feature_cols = []
        for i in range(17):
            feature_cols.extend([f'x{i}', f'y{i}', f'c{i}'])
            
        features = group[feature_cols].values
        features = features.reshape(-1, 17, 3) # (T, V, C)
        
        # Sliding Window
        num_frames = len(features)
        if num_frames < WINDOW_SIZE:
            continue
            
        for start in range(0, num_frames - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            window_data = features[start:end] # (30, 17, 3)
            window_labels = frame_labels[start:end]
            
            # Label Assignment: Align with training (>= 5 frames)
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
        sample = self.data[idx] # (T, V, C)
        sample = sample.permute(2, 0, 1) # (C, T, V)
        label = self.labels[idx]
        return sample, label

def evaluate():
    print("📊 Starting Quantitative Evaluation...")
    
    # 1. Load Data
    X, y = load_data()
    print(f"   - Total Samples: {len(X)}")
    print(f"   - Fall samples: {np.sum(y==1)}")
    print(f"   - Normal samples: {np.sum(y==0)}")
    
    # 2. Split (Must match training split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   - Test Samples: {len(X_test)}")
    
    test_dataset = FallDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Load Model
    model_path = os.path.join(utils.MODEL_DIR, 'stgcn_fall.pth')
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    model = STGCN(in_channels=3, num_class=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print("🤖 Model Loaded. Running Inference on Test Set...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            # Add M dimension: (N, C, T, V) -> (N, C, T, V, 1)
            inputs = inputs.unsqueeze(-1).to(DEVICE)
            targets = targets.to(DEVICE)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # 4. Metrics
    acc = accuracy_score(all_targets, all_preds)
    conf_mat = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=['Normal', 'Fall'], digits=4)
    
    print("\n" + "="*50)
    print("📊 ST-GCN Evaluation Results (Improved Logic)")
    print("="*50)
    print(f"✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(conf_mat)
    print("\nClassification Report:")
    print(report)
    print("="*50)
    
    # Save results
    with open("stgcn_evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("ST-GCN Evaluation Results (Improved Logic)\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_mat) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    print("💾 Report saved to stgcn_evaluation_report.txt")

if __name__ == '__main__':
    evaluate()
