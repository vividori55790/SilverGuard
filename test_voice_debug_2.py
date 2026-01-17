
import threading
import time
import pyttsx3
import pythoncom

def test_speak(text):
    print(f"Thread {threading.get_ident()}: Speak '{text}'")
    try:
        pythoncom.CoInitialize()
        engine = pyttsx3.init('sapi5')
        engine.setProperty('rate', 160)
        engine.say(text)
        engine.runAndWait()
        print("Speak finished.")
    except Exception as e:
        print(f"Speak Error: {e}")
    finally:
        pythoncom.CoUninitialize()

if __name__ == "__main__":
    print("Main: Starting independent test")
    t = threading.Thread(target=test_speak, args=("테스트 음성입니다.",))
    t.start()
    t.join(timeout=5)
    if t.is_alive():
        print("Main: Thread timed out!")
    else:
        print("Main: Finished")
