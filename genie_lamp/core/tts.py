import pyttsx3
class TTS:
    def __init__(self, cfg): self.eng = pyttsx3.init()
    def speak(self, text: str):
        if text: self.eng.say(text); self.eng.runAndWait()
