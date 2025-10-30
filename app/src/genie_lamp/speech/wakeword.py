import sounddevice as sd, numpy as np
from openwakeword import Model

class WakeWord:
    def __init__(self, phrase="lumaeth"):
        self.model = Model(wakeword_models=["hey_jarvis.tflite"])  # placeholder model
        self.phrase = phrase
    def listen_once(self, seconds=1.0, sr=16000):
        audio = sd.rec(int(seconds*sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait(); return audio[:,0]
    def detected(self) -> bool:
        audio = self.listen_once()
        scores = self.model.predict(audio)
        return max(scores.values()) > 0.6
