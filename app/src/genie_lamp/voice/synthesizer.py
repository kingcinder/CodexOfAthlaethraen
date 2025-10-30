"""Linux-focused open-source TTS synthesizer integration for Genie Lamp."""
from __future__ import annotations

import hashlib
import platform
import re
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional

import pyttsx3
from loguru import logger
from pydub import AudioSegment
from pydub.playback import play


class SynthesizerError(RuntimeError):
    """Raised when the synthesizer encounters a recoverable error."""


class PlaybackBackend(str, Enum):
    """Playback options supported by the synthesizer."""

    ENGINE = "engine"
    PYDUB = "pydub"


@dataclass
class VoiceProfile:
    """Metadata describing an installed TTS voice."""

    id: str
    name: str
    languages: List[str]
    gender: Optional[str]


@dataclass
class SynthesizerConfig:
    """Configuration for the Linux TTS synthesizer."""

    voice: Optional[str] = None
    rate: Optional[int] = None
    volume: Optional[float] = None
    cache_dir: Path = Path("./data/audio_cache")
    persist_audio: bool = False
    playback_backend: PlaybackBackend = PlaybackBackend.PYDUB
    driver: Optional[str] = None
    fallback_voice: Optional[str] = None
    async_playback: bool = False

    @classmethod
    def from_dict(cls, cfg: dict) -> "SynthesizerConfig":
        base = dict(cfg or {})
        cache_dir = Path(base.get("cache_dir", cls.cache_dir))
        playback = base.get("playback_backend", cls.playback_backend.value)
        try:
            playback_backend = PlaybackBackend(playback)
        except ValueError:
            logger.warning(
                "Unknown playback backend '{}', defaulting to {}",
                playback,
                cls.playback_backend.value,
            )
            playback_backend = cls.playback_backend
        return cls(
            voice=base.get("voice"),
            rate=base.get("rate"),
            volume=base.get("volume"),
            cache_dir=cache_dir,
            persist_audio=bool(base.get("persist_audio", False)),
            playback_backend=playback_backend,
            driver=base.get("driver"),
            fallback_voice=base.get("fallback_voice"),
            async_playback=bool(base.get("async_playback", False)),
        )


class LinuxTTSSynthesizer:
    """High-level synthesizer that wraps the system's speech engine."""

    def __init__(self, config: SynthesizerConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        driver = self._select_driver()
        logger.debug("Initialising pyttsx3 with driver={}", driver or "auto")
        self._engine = pyttsx3.init(driverName=driver) if driver else pyttsx3.init()
        self._engine_lock = threading.Lock()
        self._apply_base_settings()

    # ------------------------------------------------------------------
    # voice discovery and selection
    # ------------------------------------------------------------------
    def available_voices(self) -> List[VoiceProfile]:
        with self._engine_lock:
            voices = self._engine.getProperty("voices")
        profiles: List[VoiceProfile] = []
        for voice in voices:
            languages = []
            raw_langs = getattr(voice, "languages", []) or []
            for lang in raw_langs:
                if isinstance(lang, bytes):
                    try:
                        languages.append(lang.decode("utf-8"))
                    except UnicodeDecodeError:
                        continue
                else:
                    languages.append(str(lang))
            profiles.append(
                VoiceProfile(
                    id=str(getattr(voice, "id", voice.name)),
                    name=str(getattr(voice, "name", voice.id)),
                    languages=languages,
                    gender=getattr(voice, "gender", None),
                )
            )
        return profiles

    def set_voice(self, voice_id: str) -> None:
        with self._engine_lock:
            self._engine.setProperty("voice", voice_id)

    # ------------------------------------------------------------------
    # synthesis pipeline
    # ------------------------------------------------------------------
    def speak(self, text: str, *, blocking: bool = True) -> Optional[Path]:
        """Generate speech for *text* and play it back."""

        if not text:
            return None

        if self.config.playback_backend == PlaybackBackend.ENGINE:
            self._speak_with_engine(text, blocking=blocking)
            return None

        audio_path = self.synthesize_to_file(text)
        if blocking:
            self._play_path(audio_path)
            if not self.config.persist_audio:
                audio_path.unlink(missing_ok=True)
            return audio_path if self.config.persist_audio else None

        thread = threading.Thread(
            target=self._play_and_cleanup,
            args=(audio_path,),
            daemon=True,
        )
        thread.start()
        return audio_path if self.config.persist_audio else None

    def synthesize_to_file(self, text: str, *, cache_key: Optional[str] = None) -> Path:
        if not text:
            raise SynthesizerError("Cannot synthesise empty text")
        cache_path = self._cache_path(cache_key or text)
        with self._engine_lock:
            self._engine.save_to_file(text, str(cache_path))
            self._engine.runAndWait()
        return cache_path

    def preload(self, samples: Iterable[str]) -> None:
        for sample in samples:
            try:
                self.synthesize_to_file(sample)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to preload '{}': {}", sample, exc)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _play_and_cleanup(self, audio_path: Path) -> None:
        try:
            self._play_path(audio_path)
        finally:
            if not self.config.persist_audio:
                audio_path.unlink(missing_ok=True)

    def _play_path(self, audio_path: Path) -> None:
        try:
            segment = AudioSegment.from_file(audio_path)
            play(segment)
        except FileNotFoundError as exc:  # pragma: no cover - defensive
            raise SynthesizerError(f"Audio file missing: {audio_path}") from exc
        except Exception as exc:  # pragma: no cover - playback backend issues
            raise SynthesizerError(f"Playback failed: {exc}") from exc

    def _speak_with_engine(self, text: str, *, blocking: bool) -> None:
        with self._engine_lock:
            self._engine.say(text)
            if blocking:
                self._engine.runAndWait()
            else:
                thread = threading.Thread(target=self._engine.runAndWait, daemon=True)
                thread.start()

    def _apply_base_settings(self) -> None:
        self._apply_rate()
        self._apply_volume()
        self._apply_voice()

    def _apply_rate(self) -> None:
        if self.config.rate is None:
            return
        with self._engine_lock:
            self._engine.setProperty("rate", int(self.config.rate))

    def _apply_volume(self) -> None:
        if self.config.volume is None:
            return
        volume = max(0.0, min(float(self.config.volume), 1.0))
        with self._engine_lock:
            self._engine.setProperty("volume", volume)

    def _apply_voice(self) -> None:
        target_voice = self.config.voice
        if not target_voice and self.config.fallback_voice:
            target_voice = self.config.fallback_voice
        if not target_voice:
            return
        voices = self.available_voices()
        match = next((v for v in voices if v.id == target_voice or v.name == target_voice), None)
        if not match:
            logger.warning(
                "Requested voice '{}' not available. Using default voice.",
                target_voice,
            )
            return
        self.set_voice(match.id)

    def _cache_path(self, cache_key: str) -> Path:
        safe_key = self._sanitize_cache_key(cache_key)
        filename = f"{safe_key}.wav"
        return self.config.cache_dir / filename

    @staticmethod
    def _sanitize_cache_key(text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        text_snippet = re.sub(r"[^a-z0-9]+", "-", text.lower())[:24]
        text_snippet = text_snippet.strip("-")
        return f"{text_snippet or 'utt'}-{digest[:8]}"

    @staticmethod
    def _is_linux() -> bool:
        return platform.system().lower() == "linux"

    def _select_driver(self) -> Optional[str]:
        if self.config.driver:
            return self.config.driver
        if self._is_linux():
            # espeak is widely available; fallback to auto-detection otherwise
            return "espeak"
        return None


__all__ = [
    "LinuxTTSSynthesizer",
    "SynthesizerConfig",
    "VoiceProfile",
    "SynthesizerError",
    "PlaybackBackend",
]
