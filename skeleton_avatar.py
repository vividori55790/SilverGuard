# SilverGuard/skeleton_avatar.py
import cv2
import numpy as np
import os
import json
import time
import utils
import math

# ==========================================
# [1] 설정 및 라이브러리
# ==========================================
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("❌ MediaPipe가 없습니다. 자동 누끼 기능이 제한됩니다.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_IMAGE_PATH = os.path.join(BASE_DIR, "background.png") 
FACE_IMAGE_PATH = os.path.join(BASE_DIR, "face.png")

SKELETON_CONNECTIONS = [
    (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12)
]

# 캐싱 변수
_cached_bg = None
_cached_face = None

def check_privacy_mode():
    try:
        if os.path.exists(utils.SETTINGS_PATH):
            with open(utils.SETTINGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("PRIVACY_MODE", False)
    except:
        pass
    return False

def remove_background_from_image(image):
    """(핵심) 사진 파일의 배경을 지워주는 함수"""
    if not HAS_MEDIAPIPE: return image # 라이브러리 없으면 원본 반환
    
    print("✂️ [자동 누끼] 아바타 사진의 배경을 제거하는 중...")
    try:
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        # 정밀도 높음(1) 모드 사용
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
            # BGR -> RGB
            results = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            mask = results.segmentation_mask
            
            # 마스크가 너무 흐릿하면 확실하게 만듦 (0.1보다 크면 사람)
            mask = np.where(mask > 0.1, 1.0, 0.0).astype(np.float32)
            
            # 3채널 이미지를 4채널(BGRA)로 변환
            b, g, r = cv2.split(image)
            rgba = [b, g, r, mask * 255] # 알파 채널에 마스크 적용
            dst = cv2.merge(rgba, 4)
            
            print("✅ 배경 제거 완료!")
            return dst.astype(np.uint8)
    except Exception as e:
        print(f"⚠️ 배경 제거 실패: {e}")
        return image

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """투명 이미지를 배경 위에 합성하는 함수"""
    try:
        bg_h, bg_w, _ = background.shape
        if overlay_size is not None:
            overlay = cv2.resize(overlay, (overlay_size, overlay_size))

        h, w = overlay.shape[:2]

        # 오버레이 이미지가 3채널(불투명)이면 4채널로 변환
        if overlay.shape[2] < 4:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        # 화면 밖으로 나가는 좌표 처리
        if x + w > bg_w: w = bg_w - x
        if y + h > bg_h: h = bg_h - y
        if x < 0 or y < 0 or w <= 0 or h <= 0: return background

        # 알파 블렌딩 (합성)
        alpha_s = overlay[:h, :w, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            background[y:y+h, x:x+w, c] = (alpha_s * overlay[:h, :w, c] +
                                           alpha_l * background[y:y+h, x:x+w, c])
        return background
    except:
        return background

def draw_virtual_avatar(frame, kpts_xy, confs):
    global _cached_bg, _cached_face
    
    # ----------------------------------------------------
    # 1. 배경 이미지 준비 (실제 카메라는 여기서 버려짐!)
    # ----------------------------------------------------
    if _cached_bg is None or _cached_bg.shape[:2] != frame.shape[:2]:
        if os.path.exists(BG_IMAGE_PATH):
            img = cv2.imread(BG_IMAGE_PATH)
            if img is not None:
                _cached_bg = cv2.resize(img, (frame.shape[1], frame.shape[0]))
                _cached_bg = cv2.convertScaleAbs(_cached_bg, alpha=0.5, beta=0) # 어둡게
            else:
                _cached_bg = np.full_like(frame, (30, 30, 30))
        else:
            _cached_bg = np.full_like(frame, (30, 30, 30))

    # [중요] 실제 모습(frame) 대신 배경 이미지(_cached_bg)를 캔버스로 사용
    canvas = _cached_bg.copy()

    # ----------------------------------------------------
    # 2. 얼굴 이미지 로드 및 '자동 누끼'
    # ----------------------------------------------------
    if _cached_face is None:
        if os.path.exists(FACE_IMAGE_PATH):
            # 일단 투명도 포함해서 읽기 시도
            img = cv2.imread(FACE_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # 만약 투명 배경이 없는 사진(3채널)이라면 -> AI로 배경 지우기 시도
                if img.shape[2] == 3:
                    img = remove_background_from_image(img)
                _cached_face = img
            else:
                print("⚠️ face.png 파일을 읽을 수 없습니다.")

    # ----------------------------------------------------
    # 3. UI 및 아바타 그리기
    # ----------------------------------------------------
    cv2.putText(canvas, "SILVERGUARD VIRTUAL CORE", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
    
    if int(time.time() * 2) % 2 == 0:
        cv2.putText(canvas, "[ON] LIVE PROTECTION", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(canvas, "[  ] LIVE PROTECTION", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

    if kpts_xy is not None and len(kpts_xy) > 0:
        # (1) 뼈대 그리기 (몸통)
        for p1, p2 in SKELETON_CONNECTIONS:
            if len(kpts_xy) > max(p1, p2) and confs[p1] > 0.5 and confs[p2] > 0.5:
                pt1 = (int(kpts_xy[p1][0]), int(kpts_xy[p1][1]))
                pt2 = (int(kpts_xy[p2][0]), int(kpts_xy[p2][1]))
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

        # (2) 관절 그리기 (얼굴 0~4번 제외)
        for idx, (x, y) in enumerate(kpts_xy):
            if idx < 5: continue 
            if idx < len(confs) and confs[idx] > 0.5:
                cv2.circle(canvas, (int(x), int(y)), 5, (0, 255, 255), -1)

        # (3) 얼굴 이미지 합성
        nose_x, nose_y = kpts_xy[0]
        nose_conf = confs[0]
        
        # 어깨 너비로 얼굴 크기 계산
        face_size = 120
        if confs[5] > 0.5 and confs[6] > 0.5:
            shoulder_width = math.dist(kpts_xy[5], kpts_xy[6])
            face_size = int(shoulder_width * 1.8) # 1.8배 크기 (대두 효과)

        if nose_conf > 0.5 and _cached_face is not None:
            top_left_x = int(nose_x - face_size // 2)
            top_left_y = int(nose_y - face_size // 2)
            canvas = overlay_transparent(canvas, _cached_face, top_left_x, top_left_y, face_size)
        elif nose_conf > 0.5:
            # 이미지가 없으면 기본 원
            cv2.circle(canvas, (int(nose_x), int(nose_y)), 25, (255, 255, 255), -1)
            
    else:
        # 대기 화면
        text = "SEARCHING TARGET..."
        font_scale = 0.7
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        center_x = (frame.shape[1] - text_w) // 2
        center_y = (frame.shape[0] + text_h) // 2
        cv2.putText(canvas, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 255, 100), 1)

    return canvas