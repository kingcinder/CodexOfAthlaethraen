"""Voice synthesis subsystem for Genie Lamp."""

from .synthesizer import (
    SynthesizerConfig,
    VoiceProfile,
    LinuxTTSSynthesizer,
    SynthesizerError,
    PlaybackBackend,
)

__all__ = [
    "SynthesizerConfig",
    "VoiceProfile",
    "LinuxTTSSynthesizer",
    "SynthesizerError",
    "PlaybackBackend",
]
