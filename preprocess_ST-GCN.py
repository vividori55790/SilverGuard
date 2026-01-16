import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 도커 내부 경로 설정
DATA_DIR = '/app/data'
# 사용자가 제공한 파일명에 맞춰 수정 (예: raw_keypoints.csv)
RAW_CSV_PATH = os.path.join(DATA_DIR, 'raw_keypoints_colab_2.csv') 
LABEL_CSV_PATH = os.path.join(DATA_DIR, 'labeling_work.csv')

SAVE_X_PATH = os.path.join(DATA_DIR, 'X_final_aug.npy')
SAVE_Y_PATH = os.path.join(DATA_DIR, 'y_final_aug.npy')

WINDOW_SIZE = 120
STRIDE = 30
AUGMENTATION = True

def normalize_body(pose_data):
    left_hip = pose_data[:, 11, :2]
    right_hip = pose_data[:, 12, :2]
    hip_center = (left_hip + right_hip) / 2.0  
    
    left_shoulder = pose_data[:, 5, :2]
    right_shoulder = pose_data[:, 6, :2]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    
    torso_size = np.linalg.norm(shoulder_center - hip_center, axis=1)
    torso_size = np.where(torso_size < 0.01, 1.0, torso_size).reshape(-1, 1, 1)

    norm_xy = (pose_data[:, :, :2] - hip_center.reshape(-1, 1, 2)) / torso_size
    return np.concatenate([norm_xy, pose_data[:, :, 2:]], axis=2)

def flip_keypoints(pose_data):
    flipped = pose_data.copy()
    flipped[:, :, 0] = -flipped[:, :, 0]
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    for (left, right) in pairs:
        temp = flipped[:, left, :].copy()
        flipped[:, left, :] = flipped[:, right, :]
        flipped[:, right, :] = temp
    return flipped

def create_dataset():
    print("🚀 [Preprocess] 데이터 변환 및 증강 시작...")
    
    if not os.path.exists(RAW_CSV_PATH):
        print(f"❌ 오류: {RAW_CSV_PATH} 파일이 없습니다. data 폴더를 확인하세요.")
        return

    try:
        df = pd.read_csv(RAW_CSV_PATH)
        labels = pd.read_csv(LABEL_CSV_PATH)
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return

    label_map = labels.set_index('filename').to_dict('index')
    X_list, y_list = [], []
    grouped = df.groupby(['video_name', 'track_id'])
    
    for (fname, tid), group in tqdm(grouped):
        if len(group) < 10: continue
        group = group.sort_values('frame_idx')
        
        raw_kpts = []
        for i in range(17):
            # 컬럼명이 x0, y0, c0 ... 형태라고 가정
            raw_kpts.append(group[f'x{i}'].values)
            raw_kpts.append(group[f'y{i}'].values)
            raw_kpts.append(group[f'c{i}'].values)
        
        npy_kpts = np.array(raw_kpts).T.reshape(-1, 17, 3)
        norm_kpts = normalize_body(npy_kpts)
        
        def make_features(kpts_input):
            pos = kpts_input[:, :, :2]
            vel = np.gradient(pos, axis=0)
            acc = np.gradient(vel, axis=0)
            feat_motion = np.concatenate([pos, vel, acc], axis=2).reshape(len(pos), -1)
            confs = kpts_input[:, :, 2]
            # (Frames, 119) -> 17 * 7 (x,y,c,vx,vy,ax,ay)
            return np.concatenate([feat_motion, confs], axis=1)

        base_features = make_features(norm_kpts)
        aug_features = make_features(flip_keypoints(norm_kpts)) if AUGMENTATION else None
        
        is_fall_video = False
        fall_start, fall_end = -1, -1
        if fname in label_map and label_map[fname]['is_fall'] == 1:
            is_fall_video = True
            fall_start = label_map[fname]['start_sec']
            fall_end = label_map[fname]['end_sec']
            
        timestamps = group['time_sec'].values

        def slice_and_add(features_arr):
            # Padding for short videos
            if len(features_arr) < WINDOW_SIZE:
                pad_len = WINDOW_SIZE - len(features_arr)
                padded = np.pad(features_arr, ((0, pad_len), (0, 0)), mode='edge')
                label = 0
                if is_fall_video:
                    t_s, t_e = timestamps[0], timestamps[-1]
                    if (t_s <= fall_start <= t_e) or (fall_start <= t_s and t_e <= fall_end):
                        label = 1
                X_list.append(padded)
                y_list.append(label)
                return

            # Sliding Window
            for i in range(0, len(features_arr) - WINDOW_SIZE + 1, STRIDE):
                window = features_arr[i : i + WINDOW_SIZE]
                t_start = timestamps[i]
                t_end = timestamps[i + WINDOW_SIZE - 1]
                
                label = 0
                if is_fall_video:
                    if (t_start <= fall_start <= t_end) or (max(t_start, fall_start) <= min(t_end, fall_end)):
                         label = 1
                
                X_list.append(window)
                y_list.append(label)

        slice_and_add(base_features)
        if AUGMENTATION: slice_and_add(aug_features)

    X_final = np.array(X_list)
    y_final = np.array(y_list)

    print(f"✅ 데이터셋 생성 완료: {X_final.shape}")
    np.save(SAVE_X_PATH, X_final)
    np.save(SAVE_Y_PATH, y_final)

if __name__ == "__main__":
    create_dataset()