# SilverGuard/skeleton_avatar.py
# 2D 파츠(머리/몸통/팔/다리) + YOLO Pose Keypoints로 "종이인형" 아바타를 합성합니다.

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

# =========================================================
# [핵심] 파츠가 "원래" 가리키는 방향(단위: 도)
# 0=Right, 90=Down, 180=Left, -90=Up
# =========================================================
PART_SOURCE_ANGLES = {
    'head': 0,
    'torso': 90,        
    
    'l_arm_up': 180,    # 왼쪽 파일에 '오른쪽 팔(←)' 이미지
    'l_arm_low': 180,
    
    'r_arm_up': 0,      # 오른쪽 파일에 '왼쪽 팔(→)' 이미지
    'r_arm_low': 0,

    'l_leg_up': 90,     
    'l_leg_low': 90,   
    'r_leg_up': 90,
    'r_leg_low': 90,
}

# =========================================================
# [설정 가이드] 관절 위치(Pivot) 미세조정 (여기서 숫자를 바꾸세요!)
# =========================================================
# (x, y) 좌표는 이미지 내에서의 비율입니다. (0.0 ~ 1.0)
# 이미지가 회전되므로 X/Y가 헷갈릴 수 있습니다. 직접 값을 0.05씩 바꿔보며 확인하세요.
#
# - 상박(up): 어깨와 연결되는 지점
# - 하박(low): 팔꿈치와 연결되는 지점
# =========================================================
PART_PIVOTS = {
    'head': (0.50, 0.88),
    'torso': (0.50, 0.12),

    # ▼ [왼쪽 팔 조절] 
    # X값(0.95): 줄이면 팔이 바깥으로 나가고, 늘리면 안쪽으로 들어옵니다.
    # Y값(0.60): 줄이면 팔이 화면상 아래로, 늘리면 위로 올라갑니다.
    'l_arm_up': (0.95, 0.60),   
    'l_arm_low': (1.45, 0.60),

    # ▼ [오른쪽 팔 조절]
    # X값(0.05): 늘리면 팔이 바깥으로 나가고, 줄이면 안쪽으로 들어옵니다.
    # Y값(0.60): 줄이면 팔이 화면상 아래로, 늘리면 위로 올라갑니다.
    'r_arm_up': (0.05, 0.60),   
    'r_arm_low': (-0.35, 0.60),

    # ▼ [다리 조절] (필요하면 수정)
    'l_leg_up': (0.50, 0.18),
    'l_leg_low': (0.50, 0.06),
    'r_leg_up': (0.50, 0.18),
    'r_leg_low': (0.50, 0.06),
}

# 디버그용(관절/뼈대 선 확인)
DEBUG_DRAW_SKELETON = False

# =========================================================
# 팔이 몸통과 겹칠 때 조절하는 파라미터 (어깨 벌어짐 정도)
# =========================================================
ARM_OFFSET_OUT_RATIO = 0.02   # 숫자가 클수록 팔이 몸통 바깥으로 밀려납니다.
ARM_OFFSET_DOWN_RATIO = 0.01  # 숫자가 클수록 팔이 아래로 내려갑니다.
ARM_OFFSET_ELBOW_SCALE = 0.80

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
                if img is not None:
                    if img.shape[2] < 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    _assets[key] = img
        _assets_loaded = True


def get_vector_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def rotate_image_with_pivot(image, angle, pivot_xy):
    h, w = image.shape[:2]
    cx, cy = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    px, py = pivot_xy
    rx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
    ry = M[1, 0] * px + M[1, 1] * py + M[1, 2]

    return rotated, (rx, ry)


def overlay_transparent_safe(background, overlay, x, y):
    try:
        bg_h, bg_w, _ = background.shape
        h, w = overlay.shape[:2]

        if x < 0:
            w += x
            overlay = overlay[:, -x:]
            x = 0
        if y < 0:
            h += y
            overlay = overlay[-y:, :]
            y = 0
        if x + w > bg_w:
            w = bg_w - x
            overlay = overlay[:, :w]
        if y + h > bg_h:
            h = bg_h - y
            overlay = overlay[:h, :]

        if w <= 0 or h <= 0:
            return background

        if overlay.shape[2] < 4:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        alpha_s = overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        # Vectorized alpha blending (approx 3x faster than loop)
        overlay_rgb = overlay[:, :, :3]
        bg_roi = background[y:y+h, x:x+w]
        
        # Expand alpha to (H, W, 1) for broadcasting
        alpha_s_exp = alpha_s[:, :, np.newaxis]
        alpha_l_exp = alpha_l[:, :, np.newaxis]
        
        # Perform blending
        background[y:y+h, x:x+w] = (alpha_s_exp * overlay_rgb + alpha_l_exp * bg_roi).astype(np.uint8)
        
        return background
    except:
        return background


def _is_horizontal_part(img):
    h, w = img.shape[:2]
    return w >= h


def process_part(canvas, img_key, p1, p2, width_ref, thickness_ratio, overlap=1.15):
    """
    ✅ 파츠 합성 함수
    """
    if img_key not in _assets:
        return canvas

    img = _assets[img_key]

    target_angle = get_vector_angle(p1, p2)
    source_angle = PART_SOURCE_ANGLES.get(img_key, 90)
    rotation_angle = target_angle - source_angle
    final_angle = -rotation_angle

    length = math.dist(p1, p2)
    main = int(length * overlap)
    thick = int(width_ref * thickness_ratio)

    if main < 5 or thick < 5:
        return canvas

    horizontal = _is_horizontal_part(img)

    if horizontal:
        target_w, target_h = main, thick
    else:
        target_w, target_h = thick, main

    target_w = max(5, target_w)
    target_h = max(5, target_h)

    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    pv = PART_PIVOTS.get(img_key, (0.5, 0.5))
    px = float(pv[0]) * float(max(1, target_w - 1))
    py = float(pv[1]) * float(max(1, target_h - 1))
    px = max(0.0, min(px, float(target_w - 1)))
    py = max(0.0, min(py, float(target_h - 1)))
    pivot_px = (px, py)

    rotated, pivot_rot = rotate_image_with_pivot(resized, final_angle, pivot_px)

    x = int(p1[0] - pivot_rot[0])
    y = int(p1[1] - pivot_rot[1])

    return overlay_transparent_safe(canvas, rotated, x, y)


def _unwrap_angle_to_90(angle_deg):
    a = angle_deg
    if a > 90:
        a -= 180
    elif a < -90:
        a += 180
    return a


def process_head(canvas, img_key, shoulders, nose, ears, width_ref):
    if img_key not in _assets:
        return canvas

    img = _assets[img_key]

    if shoulders is not None and shoulders[0] is not None and shoulders[1] is not None:
        neck = ((shoulders[0][0] + shoulders[1][0]) / 2.0, (shoulders[0][1] + shoulders[1][1]) / 2.0)
    else:
        neck = (float(nose[0]), float(nose[1]))

    head_w = int(width_ref * 0.95)
    head_w = max(40, head_w)
    head_h = int(head_w * 1.15)

    angle = 0.0
    l_ear, r_ear = ears
    if l_ear is not None and r_ear is not None:
        dx = r_ear[0] - l_ear[0]
        dy = r_ear[1] - l_ear[1]
        raw = math.degrees(math.atan2(dy, dx))
        raw = _unwrap_angle_to_90(raw)
        raw = max(-25.0, min(25.0, raw))
        angle = -raw

    resized = cv2.resize(img, (head_w, head_h), interpolation=cv2.INTER_AREA)

    pv = PART_PIVOTS.get('head', (0.5, 0.88))
    pivot_px = (pv[0] * head_w, pv[1] * head_h)

    rotated, pivot_rot = rotate_image_with_pivot(resized, angle, pivot_px)

    x = int(neck[0] - pivot_rot[0])
    y = int(neck[1] - pivot_rot[1])

    return overlay_transparent_safe(canvas, rotated, x, y)


def process_torso(canvas, img_key, shoulders, hips, width_ref):
    if img_key not in _assets:
        return canvas

    img = _assets[img_key]

    neck = ((shoulders[0][0] + shoulders[1][0]) / 2.0, (shoulders[0][1] + shoulders[1][1]) / 2.0)
    pelvis = ((hips[0][0] + hips[1][0]) / 2.0, (hips[0][1] + hips[1][1]) / 2.0)

    target_angle = get_vector_angle(neck, pelvis)
    source_angle = PART_SOURCE_ANGLES.get('torso', 90)
    final_angle = -(target_angle - source_angle)

    length = math.dist(neck, pelvis)
    
    # [설정 가이드] 몸통 크기 조절 (직접 수정하세요!)
    # 아래 1.58과 1.48 숫자를 줄이면 몸통이 작아지고, 키우면 커집니다.
    target_w = int(width_ref * 1.3)  # 너비 (어깨너비 대비 비율)
    target_h = int(length * 1.2)     # 높이 (목-골반길이 대비 비율)

    target_w = max(40, target_w)
    target_h = max(60, target_h)

    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    pv = PART_PIVOTS.get('torso', (0.5, 0.12))
    pivot_px = (pv[0] * target_w, pv[1] * target_h)

    rotated, pivot_rot = rotate_image_with_pivot(resized, final_angle, pivot_px)

    x = int(neck[0] - pivot_rot[0])
    y = int(neck[1] - pivot_rot[1])

    return overlay_transparent_safe(canvas, rotated, x, y)


def _draw_debug_skeleton(canvas, kpts_xy, confs):
    if kpts_xy is None or confs is None:
        return canvas

    def ok(i):
        return confs[i] > 0.5

    for i in range(len(kpts_xy)):
        if ok(i):
            x, y = int(kpts_xy[i][0]), int(kpts_xy[i][1])
            cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1)

    bones = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (5, 6), (11, 12),
        (5, 11), (6, 12),
    ]
    for a, b in bones:
        if ok(a) and ok(b):
            p1 = (int(kpts_xy[a][0]), int(kpts_xy[a][1]))
            p2 = (int(kpts_xy[b][0]), int(kpts_xy[b][1]))
            cv2.line(canvas, p1, p2, (0, 200, 0), 2)

    return canvas


def draw_skeleton_only(canvas, kpts_xy, confs, thr=0.5):
    if kpts_xy is None or confs is None:
        return canvas

    def ok(i):
        return i < len(confs) and confs[i] > thr

    bones = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (5, 6), (11, 12),
        (5, 11), (6, 12),
    ]

    for a, b in bones:
        if ok(a) and ok(b):
            p1 = (int(kpts_xy[a][0]), int(kpts_xy[a][1]))
            p2 = (int(kpts_xy[b][0]), int(kpts_xy[b][1]))
            cv2.line(canvas, p1, p2, (0, 255, 0), 3, lineType=cv2.LINE_AA)

    for i in range(len(kpts_xy)):
        if ok(i):
            x, y = int(kpts_xy[i][0]), int(kpts_xy[i][1])
            cv2.circle(canvas, (x, y), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    return canvas


def render_privacy_frame(frame_shape, kpts_xy, confs):
    h, w = frame_shape[:2]
    canvas = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
    canvas = draw_skeleton_only(canvas, kpts_xy, confs)
    cv2.putText(canvas, "PRIVACY SKELETON", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
    return canvas


def _lr_index_map(kpts_xy, confs, thr=0.5):
    swap = False
    if confs is not None and len(confs) > 6 and confs[5] > thr and confs[6] > thr:
        swap = (kpts_xy[5][0] > kpts_xy[6][0])

    if not swap:
        return {
            'LSH': 5, 'RSH': 6,
            'LEL': 7, 'REL': 8,
            'LWR': 9, 'RWR': 10,
            'LHIP': 11, 'RHIP': 12,
            'LKN': 13, 'RKN': 14,
            'LAN': 15, 'RAN': 16,
            'LEAR': 3, 'REAR': 4,
        }
    else:
        return {
            'LSH': 6, 'RSH': 5,
            'LEL': 8, 'REL': 7,
            'LWR': 10, 'RWR': 9,
            'LHIP': 12, 'RHIP': 11,
            'LKN': 14, 'RKN': 13,
            'LAN': 16, 'RAN': 15,
            'LEAR': 4, 'REAR': 3,
        }


import time # [New Import]

# ... existing imports ...

# Global variables for smart capturing
_prev_frame = None
_ref_stable_frame = None # [New] Reference frame for drift detection
_stable_since = 0

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_scene_stable(current, prev, threshold=10.0):
    if prev is None: return False
    # Resize for faster processing
    h, w = current.shape[:2]
    curr_small = cv2.resize(current, (64, 64))
    prev_small = cv2.resize(prev, (64, 64))
    
    gray1 = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(diff)
    return mean_diff < threshold

def draw_virtual_avatar(frame, kpts_xy, confs):
    global _cached_bg, _prev_frame, _stable_since, _ref_stable_frame
    load_assets_safe()

    # 감지 여부 판단 (어깨와 코 기준)
    is_detected = False
    if kpts_xy is not None and len(kpts_xy) > 0 and confs is not None:
        if confs[0] > 0.4 or (confs[5] > 0.4 and confs[6] > 0.4):
            is_detected = True

    current_time = time.time()

    # [Smart Background Capture]
    if not is_detected:
        # 1. Stability Check (움직임이 아주 적을 때만 update)
        stable = is_scene_stable(frame, _prev_frame, threshold=2.5) 
        
        if stable:
            if _stable_since == 0:
                _stable_since = current_time
                _ref_stable_frame = frame.copy() # 기준 프레임 저장
            
            # [NEW] Drift Check (서서히 변하는 움직임 감지)
            if _ref_stable_frame is not None:
                # 시작 시점과 비교해 너무 많이 변했으면(누적 오차) 리셋
                if not is_scene_stable(frame, _ref_stable_frame, threshold=8.0):
                     _stable_since = current_time 
                     _ref_stable_frame = frame.copy()

            # 2. Duration Check (3.0초 이상 '완벽한' 정적 상태 유지 시)
            if (current_time - _stable_since) > 3.0:
                # 3. Sharpness & Change Check
                sharpness = calculate_sharpness(frame)
                
                needs_update = False
                if _cached_bg is None: needs_update = True
                # 기존 저장된 배경과 '확실히' 다를 때만 업데이트
                elif not is_scene_stable(frame, _cached_bg, threshold=15.0): 
                    needs_update = True
                
                if needs_update and sharpness > 50.0:
                    _cached_bg = frame.copy()
                    # [Long-term Storage] Save to disk for persistence
                    try:
                        cv2.imwrite(BG_IMAGE_PATH, _cached_bg)
                        print("📸 [Avatar] Clean background captured and saved to disk.")
                    except: pass
        else:
            # 움직임 감지되면 타이머 리셋
            _stable_since = 0
            _ref_stable_frame = None

        _prev_frame = frame.copy()
        
        # 라이브 뷰 리턴
        return frame
    else:
        # [Fix] 사람이 감지되는 동안은 안정화 타이머 강제 리셋 (퇴장 직후 캡쳐 방지)
        _stable_since = 0
        _ref_stable_frame = None

    # 사람이 감지되었으면, 캐시된 배경 사용
    if _cached_bg is None or _cached_bg.shape[:2] != frame.shape[:2]:
         _cached_bg = np.full_like(frame, (30, 30, 30)) # Fallback
    if _cached_bg is None or _cached_bg.shape[:2] != frame.shape[:2]:
         _cached_bg = np.full_like(frame, (30, 30, 30)) # Fallback

    canvas = _cached_bg.copy()
    
    # 아바타 합성 시작 (is_detected is True here)
    if is_detected:
        if confs[5] > 0.5 and confs[6] > 0.5:
            shoulder_w = math.dist(kpts_xy[5], kpts_xy[6])
        else:
            shoulder_w = 150

        if DEBUG_DRAW_SKELETON:
            canvas = _draw_debug_skeleton(canvas, kpts_xy, confs)

        LR = _lr_index_map(kpts_xy, confs)

        # --- Layer 1: Legs (Behind) ---
        if confs[LR['LHIP']] > 0.5 and confs[LR['LKN']] > 0.5:
            canvas = process_part(canvas, 'l_leg_up', kpts_xy[LR['LHIP']], kpts_xy[LR['LKN']], shoulder_w, thickness_ratio=0.55, overlap=1.06)
        if confs[LR['RHIP']] > 0.5 and confs[LR['RKN']] > 0.5:
            canvas = process_part(canvas, 'r_leg_up', kpts_xy[LR['RHIP']], kpts_xy[LR['RKN']], shoulder_w, thickness_ratio=0.55, overlap=1.06)
        if confs[LR['LKN']] > 0.5 and confs[LR['LAN']] > 0.5:
            canvas = process_part(canvas, 'l_leg_low', kpts_xy[LR['LKN']], kpts_xy[LR['LAN']], shoulder_w, thickness_ratio=0.45, overlap=1.10)
        if confs[LR['RKN']] > 0.5 and confs[LR['RAN']] > 0.5:
            canvas = process_part(canvas, 'r_leg_low', kpts_xy[LR['RKN']], kpts_xy[LR['RAN']], shoulder_w, thickness_ratio=0.45, overlap=1.10)

        # ---- Arm Offsets (Reduce Outward) ----
        l_sh = kpts_xy[LR['LSH']] if confs[LR['LSH']] > 0.5 else None
        r_sh = kpts_xy[LR['RSH']] if confs[LR['RSH']] > 0.5 else None
        neck = None
        if l_sh is not None and r_sh is not None:
            neck = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)

        def _outward(pt, center, amount):
            if pt is None or center is None:
                return pt
            v = np.array([pt[0] - center[0], pt[1] - center[1]], dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                return pt
            v = v / n
            return (pt[0] + v[0] * amount, pt[1] + v[1] * amount)

        arm_out = shoulder_w * ARM_OFFSET_OUT_RATIO
        arm_down = shoulder_w * ARM_OFFSET_DOWN_RATIO

        l_sh2 = _outward(tuple(l_sh), neck, arm_out) if l_sh is not None else None
        r_sh2 = _outward(tuple(r_sh), neck, arm_out) if r_sh is not None else None
        if l_sh2 is not None:
            l_sh2 = (l_sh2[0], l_sh2[1] + arm_down)
        if r_sh2 is not None:
            r_sh2 = (r_sh2[0], r_sh2[1] + arm_down)

        l_el2 = _outward(tuple(kpts_xy[LR['LEL']]), neck, arm_out * ARM_OFFSET_ELBOW_SCALE) if confs[LR['LEL']] > 0.5 else None
        r_el2 = _outward(tuple(kpts_xy[LR['REL']]), neck, arm_out * ARM_OFFSET_ELBOW_SCALE) if confs[LR['REL']] > 0.5 else None
        l_wr2 = _outward(tuple(kpts_xy[LR['LWR']]), neck, arm_out * ARM_OFFSET_ELBOW_SCALE) if confs[LR['LWR']] > 0.5 else None
        r_wr2 = _outward(tuple(kpts_xy[LR['RWR']]), neck, arm_out * ARM_OFFSET_ELBOW_SCALE) if confs[LR['RWR']] > 0.5 else None

        # --- Layer 2: Upper Arms (Behind Torso) ---
        if l_sh2 is None and confs[LR['LSH']] > 0.5:
            l_sh2 = tuple(kpts_xy[LR['LSH']])
        if r_sh2 is None and confs[LR['RSH']] > 0.5:
            r_sh2 = tuple(kpts_xy[LR['RSH']])
        if l_el2 is None and confs[LR['LEL']] > 0.5:
            l_el2 = tuple(kpts_xy[LR['LEL']])
        if r_el2 is None and confs[LR['REL']] > 0.5:
            r_el2 = tuple(kpts_xy[LR['REL']])

        if l_sh2 is not None and l_el2 is not None:
            canvas = process_part(canvas, 'l_arm_up', l_sh2, l_el2, shoulder_w, thickness_ratio=0.29, overlap=1.14)
        if r_sh2 is not None and r_el2 is not None:
            canvas = process_part(canvas, 'r_arm_up', r_sh2, r_el2, shoulder_w, thickness_ratio=0.29, overlap=1.14)

        # --- Layer 3: Torso ---
        if confs[LR['LSH']] > 0.5 and confs[LR['RSH']] > 0.5 and confs[LR['LHIP']] > 0.5 and confs[LR['RHIP']] > 0.5:
            shoulders = (kpts_xy[LR['LSH']], kpts_xy[LR['RSH']])
            hips = (kpts_xy[LR['LHIP']], kpts_xy[LR['RHIP']])
            canvas = process_torso(canvas, 'torso', shoulders, hips, shoulder_w)

        # --- Layer 4: Lower Arms (Front) ---
        if l_el2 is None and confs[LR['LEL']] > 0.5:
            l_el2 = tuple(kpts_xy[LR['LEL']])
        if r_el2 is None and confs[LR['REL']] > 0.5:
            r_el2 = tuple(kpts_xy[LR['REL']])
        if l_wr2 is None and confs[LR['LWR']] > 0.5:
            l_wr2 = tuple(kpts_xy[LR['LWR']])
        if r_wr2 is None and confs[LR['RWR']] > 0.5:
            r_wr2 = tuple(kpts_xy[LR['RWR']])

        if l_el2 is not None and l_wr2 is not None:
            canvas = process_part(canvas, 'l_arm_low', l_el2, l_wr2, shoulder_w, thickness_ratio=0.25, overlap=1.14)
        if r_el2 is not None and r_wr2 is not None:
            canvas = process_part(canvas, 'r_arm_low', r_el2, r_wr2, shoulder_w, thickness_ratio=0.25, overlap=1.14)

        # --- Layer 5: Head ---
        if confs[0] > 0.5:
            nose = kpts_xy[0]
            shoulders = (kpts_xy[LR['LSH']] if confs[LR['LSH']] > 0.5 else None,
                         kpts_xy[LR['RSH']] if confs[LR['RSH']] > 0.5 else None)
            l_ear = kpts_xy[LR['LEAR']] if confs[LR['LEAR']] > 0.5 else None
            r_ear = kpts_xy[LR['REAR']] if confs[LR['REAR']] > 0.5 else None
            canvas = process_head(canvas, 'head', shoulders, nose, (l_ear, r_ear), shoulder_w)

    else:
        cv2.putText(canvas, "SEARCHING...", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

    cv2.putText(canvas, "SILVERGUARD AVATAR", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
    return canvas