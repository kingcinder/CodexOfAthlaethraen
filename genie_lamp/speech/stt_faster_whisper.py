from faster_whisper import WhisperModel

class STT:
    def __init__(self, size="medium"):
        self.model = WhisperModel(size, device="cpu", compute_type="int8")
    def transcribe(self, wav_path: str) -> str:
        segments, _ = self.model.transcribe(wav_path, vad_filter=True)
        return " ".join(seg.text for seg in segments)
