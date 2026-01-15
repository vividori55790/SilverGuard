# SilverGuard/skeleton_avatar.py
import cv2
import numpy as np
import os
import json
import utils
import math

# ==========================================
# [1] 설정 및 파일 경로
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_IMAGE_PATH = os.path.join(BASE_DIR, "background.png")

IMAGE_FILES = {
    'head': 'head.png',
    'torso': 'torso.png',
    'l_arm_up': 'l_arm_up.png',   'l_arm_low': 'l_arm_low.png',
    'r_arm_up': 'r_arm_up.png',   'r_arm_low': 'r_arm_low.png',
    'l_leg_up': 'l_leg_up.png',   'l_leg_low': 'l_leg_low.png',
    'r_leg_up': 'r_leg_up.png',   'r_leg_low': 'r_leg_low.png'
}

# [핵심] 각 이미지 파일이 원래 가리키는 방향 (단위: 도)
# 오른쪽(0도), 아래(90도), 왼쪽(180도), 위(-90도) 기준
PART_SOURCE_ANGLES = {
    'head': -90,        # 머리: 위쪽
    'torso': -90,       # 몸통: 위쪽
    'l_arm_up': 180,    # 왼팔 사진: 왼쪽(←)을 가리킴
    'l_arm_low': 180,   # 왼팔 하박: 왼쪽(←)을 가리킴
    'r_arm_up': 0,      # 오른팔 사진: 오른쪽(→)을 가리킴
    'r_arm_low': 0,     # 오른팔 하박: 오른쪽(→)을 가리킴
    'l_leg_up': -90,    # 다리: 위쪽 (골반->무릎 벡터에 맞추기 위해)
    'l_leg_low': -90,
    'r_leg_up': -90,
    'r_leg_low': -90
}

_cached_bg = None
_assets = {}
_assets_loaded = False

def check_privacy_mode():
    try:
        if os.path.exists(utils.SETTINGS_PATH):
            with open(utils.SETTINGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("PRIVACY_MODE", False)
    except:
        pass
    return False

# ==========================================
# [2] 이미지 처리 로직
# ==========================================
def load_assets_safe():
    global _assets, _assets_loaded
    if not _assets_loaded:
        for key, filename in IMAGE_FILES.items():
            path = os.path.join(BASE_DIR, filename)
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None: _assets[key] = img
        _assets_loaded = True

def get_vector_angle(p1, p2):
    """두 점(p1->p2)의 벡터 각도를 계산 (0=Right, 90=Down, 180=Left, -90=Up)"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def rotate_image(image, angle):
    """이미지 회전 (잘림 방지)"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # OpenCV 회전은 반시계 방향이 양수
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def overlay_transparent_safe(background, overlay, x, y):
    """안전 합성 함수"""
    try:
        bg_h, bg_w, _ = background.shape
        h, w = overlay.shape[:2]

        if x < 0: w += x; overlay = overlay[:, -x:]; x = 0
        if y < 0: h += y; overlay = overlay[-y:, :]; y = 0
        if x + w > bg_w: w = bg_w - x; overlay = overlay[:, :w]
        if y + h > bg_h: h = bg_h - y; overlay = overlay[:h, :]
            
        if w <= 0 or h <= 0: return background

        if overlay.shape[2] < 4: overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
        alpha_s = overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            background[y:y+h, x:x+w, c] = (alpha_s * overlay[:, :, c] + alpha_l * background[y:y+h, x:x+w, c])
        return background
    except: return background

def process_part(canvas, img_key, p1, p2, width_ref, scale_w, overlap=1.2):
    """
    [핵심 수정] 파츠별 원래 방향(PART_SOURCE_ANGLES)을 고려하여 회전
    """
    if img_key not in _assets: return canvas
    img = _assets[img_key]

    # 1. 뼈대(Target) 각도 계산
    target_angle = get_vector_angle(p1, p2)
    
    # 2. 이미지(Source) 각도 가져오기
    source_angle = PART_SOURCE_ANGLES.get(img_key, -90)

    # 3. 회전할 각도 계산 (Target - Source)
    # 예: 왼팔(180도) -> 아래(90도)로 가려면 -90도 회전 필요
    rotation_angle = target_angle - source_angle

    # *추가 보정*: OpenCV 회전 함수는 반시계가 양수이므로 부호 확인
    # 수학적으로 맞추기 위해 음수 부호 적용 (y축이 아래로 증가하는 좌표계 특성)
    final_angle = -rotation_angle

    # 4. 크기 조절
    length = math.dist(p1, p2)
    target_h = int(length * overlap)
    target_w = int(width_ref * scale_w)
    
    if target_w < 5 or target_h < 5: return canvas

    resized = cv2.resize(img, (target_w, target_h))
    rotated = rotate_image(resized, final_angle)

    # 5. 중심점 배치
    cx, cy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    x = int(cx - rotated.shape[1] // 2)
    y = int(cy - rotated.shape[0] // 2)

    return overlay_transparent_safe(canvas, rotated, x, y)

def process_head(canvas, img_key, nose, ears, width_ref):
    """머리 그리기 (180도 뒤집힘 방지 & 기울기 반영)"""
    if img_key not in _assets: return canvas
    img = _assets[img_key]

    target_size = int(width_ref * 0.9)
    if target_size < 20: target_size = 20
    
    angle = 0
    if ears[0] is not None and ears[1] is not None:
        dx = ears[1][0] - ears[0][0] # 왼쪽 귀 -> 오른쪽 귀 벡터
        dy = ears[1][1] - ears[0][1]
        raw_angle = math.degrees(math.atan2(dy, dx))
        
        # [강력 보정] 머리가 뒤집히지 않도록 각도 제한 (-45 ~ +45)
        # 180도 근처 값이 나오면 무시하고 0으로 처리하거나 제한함
        if raw_angle > 45: raw_angle = 45
        elif raw_angle < -45: raw_angle = -45
        
        # OpenCV 회전 방향 보정
        angle = -raw_angle

    resized = cv2.resize(img, (target_size, int(target_size * 1.2)))
    rotated = rotate_image(resized, angle)

    x = int(nose[0] - rotated.shape[1] // 2)
    y = int(nose[1] - rotated.shape[0] // 2)

    return overlay_transparent_safe(canvas, rotated, x, y)

def process_torso(canvas, img_key, shoulders, hips, width_ref):
    """몸통 그리기"""
    if img_key not in _assets: return canvas
    img = _assets[img_key]

    neck = (shoulders[0] + shoulders[1]) / 2
    pelvis = (hips[0] + hips[1]) / 2
    
    target_angle = get_vector_angle(neck, pelvis)
    source_angle = PART_SOURCE_ANGLES.get('torso', -90)
    final_angle = -(target_angle - source_angle)

    length = math.dist(neck, pelvis)
    target_w = int(width_ref * 1.6)
    target_h = int(length * 1.5)
    
    resized = cv2.resize(img, (target_w, target_h))
    rotated = rotate_image(resized, final_angle)

    cx, cy = (neck[0] + pelvis[0]) / 2, (neck[1] + pelvis[1]) / 2
    x = int(cx - rotated.shape[1] // 2)
    y = int(cy - rotated.shape[0] // 2)

    return overlay_transparent_safe(canvas, rotated, x, y)

# ==========================================
# [3] 메인 그리기 실행
# ==========================================
def draw_virtual_avatar(frame, kpts_xy, confs):
    global _cached_bg
    load_assets_safe()

    if _cached_bg is None or _cached_bg.shape[:2] != frame.shape[:2]:
        if os.path.exists(BG_IMAGE_PATH):
            img = cv2.imread(BG_IMAGE_PATH)
            if img is not None:
                _cached_bg = cv2.resize(img, (frame.shape[1], frame.shape[0]))
                _cached_bg = cv2.convertScaleAbs(_cached_bg, alpha=0.5, beta=0)
            else: _cached_bg = np.full_like(frame, (30, 30, 30))
        else: _cached_bg = np.full_like(frame, (30, 30, 30))
    canvas = _cached_bg.copy()

    if kpts_xy is not None and len(kpts_xy) > 0:
        # 기준: 어깨 너비
        if confs[5]>0.5 and confs[6]>0.5:
            shoulder_w = math.dist(kpts_xy[5], kpts_xy[6])
        else:
            shoulder_w = 150

        # --- Layer 1: 다리 ---
        if confs[11]>0.5 and confs[13]>0.5:
            canvas = process_part(canvas, 'l_leg_up', kpts_xy[11], kpts_xy[13], shoulder_w, 0.45)
        if confs[12]>0.5 and confs[14]>0.5:
            canvas = process_part(canvas, 'r_leg_up', kpts_xy[12], kpts_xy[14], shoulder_w, 0.45)
        if confs[13]>0.5 and confs[15]>0.5:
            canvas = process_part(canvas, 'l_leg_low', kpts_xy[13], kpts_xy[15], shoulder_w, 0.35)
        if confs[14]>0.5 and confs[16]>0.5:
            canvas = process_part(canvas, 'r_leg_low', kpts_xy[14], kpts_xy[16], shoulder_w, 0.35)

        # --- Layer 2: 몸통 ---
        if confs[5]>0.5 and confs[6]>0.5 and confs[11]>0.5 and confs[12]>0.5:
            shoulders = (kpts_xy[5], kpts_xy[6])
            hips = (kpts_xy[11], kpts_xy[12])
            canvas = process_torso(canvas, 'torso', shoulders, hips, shoulder_w)

        # --- Layer 3: 머리 ---
        if confs[0] > 0.5:
            nose = kpts_xy[0]
            l_ear = kpts_xy[3] if confs[3]>0.5 else None
            r_ear = kpts_xy[4] if confs[4]>0.5 else None
            canvas = process_head(canvas, 'head', nose, (l_ear, r_ear), shoulder_w)

        # --- Layer 4: 팔 (방향 보정 자동 적용됨) ---
        if confs[5]>0.5 and confs[7]>0.5:
            canvas = process_part(canvas, 'l_arm_up', kpts_xy[5], kpts_xy[7], shoulder_w, 0.35)
        if confs[7]>0.5 and confs[9]>0.5:
            canvas = process_part(canvas, 'l_arm_low', kpts_xy[7], kpts_xy[9], shoulder_w, 0.30)

        if confs[6]>0.5 and confs[8]>0.5:
            canvas = process_part(canvas, 'r_arm_up', kpts_xy[6], kpts_xy[8], shoulder_w, 0.35)
        if confs[8]>0.5 and confs[10]>0.5:
            canvas = process_part(canvas, 'r_arm_low', kpts_xy[8], kpts_xy[10], shoulder_w, 0.30)
    else:
        cv2.putText(canvas, "SEARCHING...", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

    cv2.putText(canvas, "SILVERGUARD AVATAR", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
    return canvas