#!/usr/bin/env python3
"""Build consolidated voice references and optional training datasets.

This helper combines multiple clean recordings of a speaker into a single
reference file that works well with Coqui XTTS style voice cloning.  It can also
normalise, resample, and export aligned ``metadata.csv`` files compatible with
Coqui's fine-tuning recipes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


class VoiceProfileError(Exception):
    """Raised when the input arguments are inconsistent."""


def _discover_audio_paths(samples: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for entry in samples:
        path = Path(entry).expanduser().resolve()
        if not path.exists():
            raise VoiceProfileError(f"Input path does not exist: {entry}")
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    paths.append(child)
        else:
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise VoiceProfileError(f"Unsupported audio format: {path.suffix} ({entry})")
            paths.append(path)
    if not paths:
        raise VoiceProfileError("No audio files discovered. Provide wav/flac/mp3/ogg recordings.")
    return paths


def _load_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = _resample_audio(audio, sr, target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    if audio.size == 0:
        return audio
    duration = audio.shape[0] / float(src_sr)
    target_length = max(1, int(round(duration * dst_sr)))
    source_times = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
    resampled = np.interp(target_times, source_times, audio)
    return resampled.astype(np.float32)


def _insert_silence(segments: Sequence[np.ndarray], silence_length: int) -> np.ndarray:
    valid_segments = [segment for segment in segments if segment.size > 0]
    chunks: List[np.ndarray] = []
    silence = np.zeros(silence_length, dtype=np.float32) if silence_length > 0 else None
    for index, segment in enumerate(valid_segments):
        chunks.append(segment)
        if silence is not None and index < len(valid_segments) - 1:
            chunks.append(silence)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def _normalise(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = np.max(np.abs(audio))
    if max_abs == 0:
        return audio
    return (audio / max_abs) * peak


def _load_transcripts(
    inline: Optional[Sequence[str]], transcript_file: Optional[str], count: int
) -> Optional[List[str]]:
    if inline and transcript_file:
        raise VoiceProfileError("Provide transcripts inline or via file, not both.")
    transcripts: Optional[List[str]] = None
    if inline:
        transcripts = [t.strip() for t in inline]
    if transcript_file:
        file_path = Path(transcript_file).expanduser().resolve()
        transcripts = [
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if transcripts is not None and len(transcripts) != count:
        raise VoiceProfileError(
            f"Number of transcripts ({len(transcripts)}) does not match number of audio clips ({count})."
        )
    return transcripts


def build_voice_profile(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = _discover_audio_paths(args.samples)
    processed: List[np.ndarray] = []
    for path in audio_paths:
        audio, _ = _load_audio(path, args.target_sr)
        if args.trim_silence:
            audio = _trim_silence(audio, args.trim_threshold)
        processed.append(audio)

    silence_len = int(round(args.silence_ms / 1000.0 * args.target_sr))
    combined = _insert_silence(processed, silence_len)
    if not args.no_normalise:
        combined = _normalise(combined)

    reference_path = output_dir / f"{args.name}_reference.wav"
    sf.write(reference_path, combined, args.target_sr)

    transcripts = _load_transcripts(args.transcripts, args.transcript_file, len(processed))
    metadata_path: Optional[Path] = None
    if args.dataset_dir:
        if transcripts is None:
            raise VoiceProfileError("Transcripts are required when exporting a fine-tuning dataset.")
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        metadata_lines: List[str] = []
        for index, (clip, transcript) in enumerate(zip(processed, transcripts)):
            clip_name = f"{args.name}_{index:03d}.wav"
            clip_path = dataset_dir / clip_name
            if not args.no_normalise:
                clip = _normalise(clip)
            sf.write(clip_path, clip, args.target_sr)
            metadata_lines.append(f"{clip_name}|{transcript}")
        metadata_path = dataset_dir / "metadata.csv"
        metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    profile = {
        "name": args.name,
        "reference_wav": str(reference_path),
        "sample_rate": args.target_sr,
        "clips": [str(path) for path in audio_paths],
    }
    if metadata_path:
        profile["metadata_csv"] = str(metadata_path)
    profile_path = output_dir / f"{args.name}_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    return reference_path


def _trim_silence(audio: np.ndarray, threshold: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    abs_audio = np.abs(audio)
    mask = abs_audio > threshold
    if not np.any(mask):
        return audio
    first = int(np.argmax(mask))
    last = len(audio) - int(np.argmax(mask[::-1]))
    return audio[first:last]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True, help="Audio files or directories containing recordings.")
    parser.add_argument("--name", required=True, help="Short identifier for the resulting voice profile.")
    parser.add_argument(
        "--output-dir",
        default="build/voice_profiles",
        help="Directory where the combined reference wav and profile metadata will be stored.",
    )
    parser.add_argument(
        "--dataset-dir",
        help="Optional directory to export per-clip wav files plus metadata.csv for fine-tuning.",
    )
    parser.add_argument("--transcripts", nargs="*", help="Transcripts that align with the provided audio clips.")
    parser.add_argument(
        "--transcript-file",
        help="Text file with one transcript per line to pair with the discovered audio clips.",
    )
    parser.add_argument("--target-sr", type=int, default=22050, help="Sample rate to resample audio to (Hz).")
    parser.add_argument("--silence-ms", type=float, default=150.0, help="Silence inserted between clips when combining (ms).")
    parser.add_argument(
        "--trim-silence",
        action="store_true",
        help="Enable naive silence trimming based on amplitude threshold before processing.",
    )
    parser.add_argument(
        "--trim-threshold",
        type=float,
        default=0.02,
        help="Absolute amplitude threshold for silence trimming (0-1 range).",
    )
    parser.add_argument(
        "--no-normalise",
        action="store_true",
        help="Skip peak normalisation. Disable if you already processed loudness externally.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        reference_path = build_voice_profile(args)
    except VoiceProfileError as exc:
        parser.error(str(exc))
        return
    print(f"Voice profile reference saved to: {reference_path}")


if __name__ == "__main__":
    main()
