
import threading
import pythoncom
import win32com.client

def test_sapi_speak(text):
    print(f"Thread {threading.get_ident()}: SAPI Speak '{text}'")
    try:
        pythoncom.CoInitialize()
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Speak(text)
        print("SAPI Speak finished.")
    except Exception as e:
        print(f"SAPI Error: {e}")
    finally:
        pythoncom.CoUninitialize()

if __name__ == "__main__":
    print("Main: Starting SAPI test")
    t = threading.Thread(target=test_sapi_speak, args=("테스트 음성입니다.",))
    t.start()
    t.join(timeout=5)
    if t.is_alive():
        print("Main: Thread timed out!")
    else:
        print("Main: Finished")
