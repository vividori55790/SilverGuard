
import subprocess
import threading
import sys

def test_ps_speak(text):
    print(f"Thread {threading.get_ident()}: PS Speak '{text}'")
    try:
        cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
        subprocess.run(["powershell", "-Command", cmd], check=True)
        print("PS Speak finished.")
    except Exception as e:
        print(f"PS Error: {e}")

if __name__ == "__main__":
    print("Main: Starting PS test")
    t = threading.Thread(target=test_ps_speak, args=("테스트 음성입니다.",))
    t.start()
    t.join(timeout=10)
    if t.is_alive():
        print("Main: Thread timed out!")
    else:
        print("Main: Finished")
