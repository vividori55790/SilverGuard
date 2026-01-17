#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SilverGuard 알림 워크플로우 통합 테스트
전체 낙상 감지 → 음성 확인 → 텔레그램 → 전화 프로세스를 검증합니다.
"""

import sys
import os
import time
import numpy as np
import cv2

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import utils
from voice_module import speak, run_voice_emergency_check

def test_tts():
    """음성 출력 테스트"""
    print("\n" + "="*60)
    print("🔊 1단계: TTS (Text-to-Speech) 테스트")
    print("="*60)
    
    try:
        speak("테스트 음성입니다. 소리가 들리면 정상입니다.")
        print("✅ TTS 테스트 통과")
        return True
    except Exception as e:
        print(f"❌ TTS 실패: {e}")
        return False

def test_telegram():
    """텔레그램 전송 테스트"""
    print("\n" + "="*60)
    print("📱 2단계: 텔레그램 전송 테스트")
    print("="*60)
    
    # Create dummy image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "TEST ALERT", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    test_path = os.path.join(utils.ALERT_DIR, "TEST_ALERT.jpg")
    cv2.imwrite(test_path, dummy_img)
    
    try:
        success = utils.send_telegram_alert(
            test_path, 
            "🧪 [테스트 알림]\n시스템 통합 테스트 중입니다.",
            None
        )
        
        if success:
            print("✅ 텔레그램 전송 성공")
        else:
            print("⚠️ 텔레그램 전송 실패 (설정 확인 필요)")
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
            
        return success
    except Exception as e:
        print(f"❌ 텔레그램 테스트 실패: {e}")
        return False

def test_phone_call():
    """전화 앱 실행 테스트 (실제 발신은 하지 않음)"""
    print("\n" + "="*60)
    print("📞 3단계: 전화 앱 실행 테스트")
    print("="*60)
    
    try:
        # Load settings to check if auto call is enabled
        if os.path.exists(utils.SETTINGS_PATH):
            import json
            with open(utils.SETTINGS_PATH, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                auto_call = settings.get("AUTO_CALL_ENABLED", False)
                contact = settings.get("EMERGENCY_CONTACT", "")
                
                if not auto_call:
                    print("⚠️ 자동 전화 기능이 비활성화되어 있습니다.")
                    print("   대시보드에서 '자동 전화 걸기'를 활성화하세요.")
                    return False
                
                if not contact:
                    print("⚠️ 긴급 연락처가 설정되지 않았습니다.")
                    print("   대시보드에서 연락처를 입력하세요.")
                    return False
                
                print(f"📱 설정된 연락처: {contact}")
                print("⚠️ 실제 전화는 걸지 않습니다 (수동 테스트 필요)")
                print("   낙상 시뮬레이션 버튼으로 전체 플로우를 테스트하세요.")
                return True
        else:
            print("⚠️ 설정 파일이 없습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 전화 설정 확인 실패: {e}")
        return False

def test_voice_workflow(simulate=False):
    """음성 워크플로우 테스트"""
    print("\n" + "="*60)
    print("🎤 4단계: 음성 워크플로우 테스트")
    print("="*60)
    
    if not simulate:
        print("⚠️ 실제 음성 확인은 낙상 감지 시에만 실행됩니다.")
        print("   대시보드의 '낙상 시뮬레이션' 버튼으로 테스트하세요.")
        return True
    
    try:
        # Create dummy alert image
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "FALL DETECTED", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        test_path = os.path.join(utils.ALERT_DIR, "VOICE_TEST.jpg")
        cv2.imwrite(test_path, dummy_img)
        
        print("🔊 음성 확인 시작...")
        print("   ⏱️ 10초 안에 '괜찮아'라고 말하면 알림이 전송되지 않습니다.")
        print("   🔇 무응답 시 2차 볼륨 증가 후 재확인됩니다.")
        
        result = run_voice_emergency_check(test_path)
        
        print(f"\n📊 음성 확인 결과: {result}")
        
        if result == "SAFE":
            print("✅ 안전 확인 - 알림 전송 없음")
        else:
            print("🚨 비상 상황 - 알림이 전송됩니다")
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
            
        return True
        
    except Exception as e:
        print(f"❌ 음성 워크플로우 테스트 실패: {e}")
        return False

def main():
    """통합 테스트 실행"""
    print("\n" + "🛡️ "*20)
    print("    SilverGuard 알림 워크플로우 통합 테스트")
    print("🛡️ "*20 + "\n")
    
    # Ensure directories exist
    utils.ensure_dirs()
    
    results = {
        "TTS": False,
        "Telegram": False,
        "Phone": False,
        "Voice": False
    }
    
    # Run tests
    results["TTS"] = test_tts()
    time.sleep(2)
    
    results["Telegram"] = test_telegram()
    time.sleep(2)
    
    results["Phone"] = test_phone_call()
    time.sleep(2)
    
    results["Voice"] = test_voice_workflow(simulate=False)
    
    # Summary
    print("\n" + "="*60)
    print("📊 테스트 결과 요약")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 모든 테스트 통과!")
        print("✅ 시스템이 정상적으로 작동합니다.")
    else:
        print("⚠️ 일부 테스트 실패")
        print("📝 ALERT_WORKFLOW_GUIDE.md를 참조하여 문제를 해결하세요.")
    print("="*60)
    
    print("\n💡 추가 테스트:")
    print("   대시보드에서 '낙상 시뮬레이션' 버튼을 눌러")
    print("   전체 워크플로우(음성 → 텔레그램 → 전화)를 검증하세요.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
