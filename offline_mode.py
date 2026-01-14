import os
import socket
import datetime
import threading
import winsound
import csv
import utils

QUEUE_FILE = os.path.join(utils.DATA_DIR, "pending_alerts.csv")

def is_internet_available():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=1.0)
        return True
    except OSError:
        return False

def save_to_queue(image_path, message):
    file_exists = os.path.isfile(QUEUE_FILE)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(QUEUE_FILE, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["시간", "이미지경로", "메시지"])
        writer.writerow([timestamp, image_path, message])
    print(f"💾 [오프라인 저장] {timestamp} 사고 기록을 대기열에 저장했습니다.")

def play_siren_async():
    def siren_logic():
        for _ in range(3):
            winsound.Beep(2500, 500)
    threading.Thread(target=siren_logic, daemon=True).start()

def sync_unsent_data():
    """인터넷 복구 시 전송 시도 (에러 방지 기능 추가)"""
    if not os.path.exists(QUEUE_FILE) or not is_internet_available():
        return

    unsent_items = []
    try:
        with open(QUEUE_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            unsent_items = list(reader)
    except Exception as e:
        print(f"⚠️ 대기열 파일을 읽는 중 오류 발생(파일을 삭제합니다): {e}")
        os.remove(QUEUE_FILE)
        return

    if not unsent_items: return

    print(f"🔄 [온라인 복구] 미전송 알림 {len(unsent_items)}건 전송 시작...")
    still_pending = []
    for item in unsent_items:
        # 안전하게 항목 확인: '이미지경로'가 없으면 예전 파일이므로 건너뜁니다.
        path = item.get('이미지경로')
        msg = item.get('메시지', '사고 기록')
        
        if path and utils.send_telegram_alert(path, f"[복구 전송] {msg}"):
            print(f"✅ [전송 완료] {item.get('시간')} 기록 전송 성공")
        else:
            still_pending.append(item)

    if not still_pending:
        os.remove(QUEUE_FILE)
        print("✨ 모든 데이터 전송 완료! 대기열을 비웠습니다.")
    else:
        with open(QUEUE_FILE, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["시간", "이미지경로", "메시지"])
            writer.writeheader()
            writer.writerows(still_pending)

def activate_offline_safety_mode(image_path, result_type):
    save_to_queue(image_path, f"낙상 감지! 결과: {result_type}")
    play_siren_async()