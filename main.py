# SilverGuard/main.py
import sys
import os

# Ensure current directory is in path (especially if run from outside)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core.engine import SilverGuardEngine

def main():
    engine = SilverGuardEngine()
    try:
        engine.run()
    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 시스템이 종료됩니다.")
    except Exception as e:
        print(f"\n❌ 시스템 오류 발생: {e}")
    finally:
        # engine.release_resources() # If such method exists, or just let python cleanup
        print("✅ 모든 프로세스가 안전하게 종료되었습니다.")

if __name__ == '__main__':
    main()