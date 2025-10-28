"""Genie Lamp text-to-speech orchestration."""
from __future__ import annotations

from loguru import logger

from voice import LinuxTTSSynthesizer, SynthesizerConfig, SynthesizerError


class TTS:
    """Facade around the open-source Linux synthesizer project."""

    def __init__(self, cfg):
        voice_cfg = (cfg or {}).get("tts", {})
        self._config = SynthesizerConfig.from_dict(voice_cfg)
        self._synth = LinuxTTSSynthesizer(self._config)
        preload = voice_cfg.get("preload_samples", [])
        if preload:
            logger.debug("Preloading %d voice samples", len(preload))
            self._synth.preload(preload)

    def speak(self, text: str) -> None:
        if not text:
            return
        try:
            blocking = not self._config.async_playback
            self._synth.speak(text, blocking=blocking)
        except SynthesizerError as exc:
            logger.warning("Synthesizer error: {}", exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to render speech: {}", exc)

    def available_voices(self):
        try:
            return self._synth.available_voices()
        except Exception as exc:  # pragma: no cover - discovery is best-effort
            logger.warning("Could not enumerate voices: {}", exc)
            return []

    def set_voice(self, voice_id: str) -> bool:
        try:
            self._synth.set_voice(voice_id)
            return True
        except Exception as exc:  # pragma: no cover - best effort setter
            logger.warning("Unable to switch to voice '{}': {}", voice_id, exc)
            return False
