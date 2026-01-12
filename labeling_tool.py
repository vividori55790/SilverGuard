import cv2
import pandas as pd
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog

# ================= 설정 =================
OUTPUT_FILE = 'labeling_work.csv'
TARGET_WIDTH = 1280  # 화면에 보여줄 가로 크기
# ========================================

def select_video_folder():
    """ 윈도우 폴더 선택 창을 띄웁니다 """
    root = tk.Tk()
    root.withdraw() 
    print("📂 팝업창에서 '동영상들이 들어있는 폴더(data)'를 선택해주세요...")
    folder_selected = filedialog.askdirectory(title="[SilverGuard] 영상이 있는 폴더를 선택하세요")
    root.destroy()
    return folder_selected

def get_all_video_files(root_dir):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    video_list = []
    
    print(f"🔎 '{root_dir}' 내부를 검색 중...", end='')
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                full_path = os.path.join(root, file)
                video_list.append(full_path)
    
    print(f" 완료! (총 {len(video_list)}개 발견)")
    return video_list

def load_existing_work():
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            return df, df['filepath'].tolist()
        except:
            return pd.DataFrame(), []
    return pd.DataFrame(), []

def resize_frame(frame, target_width):
    """ 비율을 유지하며 가로 길이에 맞춰 리사이즈 """
    h, w = frame.shape[:2]
    if w > target_width:
        scale = target_width / w
        new_h = int(h * scale)
        return cv2.resize(frame, (target_width, new_h))
    return frame

def run_labeling():
    # 1. 폴더 선택
    video_root_dir = select_video_folder()
    if not video_root_dir:
        print("❌ 폴더가 선택되지 않았습니다.")
        return

    # 2. 영상 리스트 확보
    all_videos = get_all_video_files(video_root_dir)
    if not all_videos:
        print("❌ 영상 파일이 없습니다.")
        return

    # 3. 이어하기 기능
    existing_df, done_files = load_existing_work()
    data_list = existing_df.to_dict('records')
    
    done_filenames = [os.path.basename(p) for p in done_files]
    todo_videos = [v for v in all_videos if os.path.basename(v) not in done_filenames]
    
    if not todo_videos:
        print("🎉 이미 모든 라벨링이 완료되었습니다!")
        return

    print(f"▶ 남은 작업량: {len(todo_videos)}개")
    print("="*50)
    print("   [조작법]")
    print("   SPACE : 재생 / 일시정지")
    print("   S     : 낙상 시작 (Start)")
    print("   E     : 낙상 끝 (End)")
    print("   R     : 기록 초기화")
    print("   A / D : 1초 뒤로 / 1초 앞으로")
    print("   Z / C : 5초 뒤로 / 5초 앞으로")
    print("   N     : 저장 후 다음 영상")
    print("   Q     : 저장 후 종료")
    print("="*50)

    for idx, video_path in enumerate(todo_videos):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 전체 재생 시간 계산
        duration = total_frames / fps if fps > 0 else 0
        
        display_name = os.path.basename(video_path)
        
        start_time = None
        end_time = None
        is_fall = 0
        paused = False
        
        # 첫 프레임 미리 읽기
        ret, frame = cap.read()
        if not ret:
            print(f"❌ 영상 로드 실패: {display_name}")
            continue

        while True:
            # 일시정지가 아닐 때만 프레임을 계속 읽음
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    paused = True # 영상 끝나면 일시정지
            
            # 리사이즈 및 화면 출력 준비
            if frame is not None:
                display_frame = resize_frame(frame, TARGET_WIDTH)
            else:
                display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # 현재 위치 (초 단위)
            current_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            
            # 1. 파일명 표시
            cv2.putText(display_frame, f"[{idx+1}/{len(todo_videos)}] {display_name}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 2. 라벨링 상태 표시
            status_text = f"Start: {start_time if start_time else '-'} | End: {end_time if end_time else '-'}"
            status_color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.putText(display_frame, status_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # 3. 시간 정보 표시 (현재시간 / 전체시간)
            time_text = f"Time: {current_pos:.1f}s / {duration:.1f}s"
            cv2.putText(display_frame, time_text, (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if paused:
                cv2.putText(display_frame, "PAUSED", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('SilverGuard Labeling Tool v4', display_frame)

            # 키 입력 대기
            key = cv2.waitKey(30 if not paused else 100) & 0xFF

            # === 키 조작 로직 ===
            if key == 32: # SPACE
                paused = not paused
            
            elif key == ord('s'): 
                start_time = round(current_pos, 2)
                print(f"📌 Start: {start_time}")
            
            elif key == ord('e'): 
                end_time = round(current_pos, 2)
                is_fall = 1
                paused = True
                print(f"📌 End: {end_time}")
            
            elif key == ord('r'): 
                start_time=None; end_time=None; is_fall=0
                print("🔄 Reset")
            
            # A키: 뒤로 1초
            elif key == ord('a'): 
                cur_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur_f - fps))
                ret, frame = cap.read()
            
            # D키: 앞으로 1초
            elif key == ord('d'): 
                cur_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames, cur_f + fps))
                ret, frame = cap.read()

            # [추가] Z키: 뒤로 5초
            elif key == ord('z'): 
                cur_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur_f - (fps * 5)))
                ret, frame = cap.read()

            # [추가] C키: 앞으로 5초
            elif key == ord('c'): 
                cur_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames, cur_f + (fps * 5)))
                ret, frame = cap.read()

            elif key == ord('n'): 
                break 
            
            elif key == ord('q'): 
                cap.release()
                cv2.destroyAllWindows()
                save_csv(data_list)
                return
            
            # 창 닫힘 감지
            if cv2.getWindowProperty('SilverGuard Labeling Tool v4', cv2.WND_PROP_VISIBLE) < 1:
                cap.release()
                save_csv(data_list)
                return

        data_list.append({
            'filepath': video_path,
            'filename': display_name,
            'is_fall': is_fall,
            'start_sec': start_time,
            'end_sec': end_time
        })
        print(f"✅ 저장됨: {display_name}")
        cap.release()
        save_csv(data_list)

    cv2.destroyAllWindows()
    print("\n🎉 모든 라벨링 완료!")

def save_csv(data):
    if not data: return
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    run_labeling()