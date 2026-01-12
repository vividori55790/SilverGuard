import os

# 문제가 되는 폴더 경로를 여기에 넣으세요 (앞에 r 꼭 붙이기!)
check_path = r"C:\Users\vivid\Documents\Git Project\PythonUtil\data\aihub_videos\Vid\[원천]inside_H11H21H31\H11H21H31" 

print(f"📂 경로 확인: {check_path}")

if not os.path.exists(check_path):
    print("❌ 에러: 폴더 자체가 존재하지 않는다고 나옵니다. 경로 오타를 확인하세요.")
else:
    print("✅ 폴더는 존재합니다. 내부 파일 목록:")
    files = os.listdir(check_path)
    if not files:
        print("   -> 폴더가 비어 있습니다!")
    else:
        for f in files:
            print(f"   - {f}")