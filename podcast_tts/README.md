# Local Podcast TTS Pipeline

This guide walks you through setting up a fully offline text-to-speech (TTS) workflow on Linux using the open-source [Coqui TTS](https://github.com/coqui-ai/TTS) project. The included `synthesize_podcast.py` helper script turns a raw transcript into an alternating back-and-forth dialogue with expressive delivery.

Your hardware (Ryzen 7 3700X CPU, Radeon 5700 XT GPU, 32 GB RAM, NVMe + HDD storage) is sufficient for high-quality results. The 5700 XT is not officially supported by AMD's ROCm GPU runtime, so the default setup runs on CPU; optional GPU instructions are included for experimentation.

## 1. Prerequisites

1. **Operating system**: A recent 64-bit Linux distribution (Ubuntu 22.04 LTS is used in the commands below).
2. **System packages**:
   ```bash
   sudo apt update
   sudo apt install python3.10 python3.10-venv build-essential ffmpeg libsndfile1
   ```
3. **Optional GPU runtime (experimental)**:
   * AMD ROCm 5.4+ can accelerate PyTorch on RDNA GPUs, but stability on the RX 5700 XT is not guaranteed.
   * If you already have ROCm installed and working with PyTorch, you may use the `--device rocm` flag when running the script.
   * Without ROCm the CPU-only pipeline is reliable; the Ryzen 7 3700X generates ~4–6× real-time audio with XTTS v2.

## 2. Create an isolated Python environment

```bash
cd /path/to/this/repository
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

## 3. Install Python dependencies

Install PyTorch, torchaudio, and Coqui TTS. Replace the CPU wheel with a ROCm build if your system has functional ROCm drivers.

```bash
# CPU wheels (recommended for RX 5700 XT users)
pip install torch==2.1.2+cpu torchaudio==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Core tooling
pip install TTS==0.21.3 soundfile numpy
```

**ROCm alternative (optional):**

```bash
pip install torch==2.1.2+rocm5.4.2 torchaudio==2.1.2+rocm5.4.2 --index-url https://download.pytorch.org/whl/rocm5.4.2
```

If you see `HIP` or `ROCm` errors, fall back to the CPU instructions.

## 4. Download expressive multi-speaker models

The `xtts_v2` multilingual model provides strong emotional range and speaker adaptation.

```bash
python - <<'PY'
from TTS.utils.manage import ModelManager
manager = ModelManager()
manager.download_model('tts_models/multilingual/multi-dataset/xtts_v2')
PY
```

The model files will be cached under `~/.local/share/tts/`. Subsequent runs reuse the cache automatically.

## 5. Prepare speaker reference audio

XTTS v2 produces its most natural results when you provide 20–60 seconds of clean speech for each voice you want to emulate.

1. Record or collect WAV/FLAC audio for each voice (16 kHz or 22.05 kHz mono recommended). Trim silence and background noise with any editor (Audacity is convenient).
2. Store the references somewhere accessible, e.g. `assets/voice_alex.wav` and `assets/voice_ai.wav`.

You may also use the model's built-in synthetic voices by passing speaker names (e.g. `--speakers en_female_01 en_male_01`). However, custom references usually sound more natural for podcast-style narration.

## 6. Create custom voice profiles and training datasets

When you have multiple recordings of a speaker, combine them into a single,
clean reference that XTTS can imitate consistently. The
`create_voice_profile.py` helper normalises, resamples, and stitches the clips
while optionally exporting a fine-tuning dataset.

```bash
python podcast_tts/create_voice_profile.py \
  --samples recordings/host_voice \
  --name host \
  --output-dir assets/voices \
  --dataset-dir data/host_xtts \
  --transcript-file recordings/host_transcripts.txt \
  --trim-silence
```

Outputs:

* `assets/voices/host_reference.wav`: feed this file to
  `--speaker-wavs` for high-fidelity cloning.
* `assets/voices/host_profile.json`: metadata documenting how the profile was
  generated and where the source clips live.
* `data/host_xtts/metadata.csv` plus cleaned per-clip WAV files (when
  `--dataset-dir` is set). This structure follows Coqui's LJSpeech-style
  expectations for further training.

You can point `--samples` at one or more directories or individual files. If
you would rather enter transcripts inline, swap `--transcript-file` for a
series of `--transcripts "First sentence" "Second sentence" ...` arguments.

Use the generated reference file directly with the synthesiser:

```bash
python podcast_tts/synthesize_podcast.py transcripts/episode1.txt \
  --speaker-wavs assets/voices/host_reference.wav assets/voice_ai.wav
```

## 7. Optional: fine-tune XTTS on your recordings

Fine-tuning lets the model internalise a speaker's timbre and cadence instead
of relying solely on few-shot voice cloning. After exporting a dataset with the
previous step:

1. Copy the baseline XTTS configuration so you can customise it:
   ```bash
   mkdir -p podcast_tts/configs
   cp ~/.local/share/tts/tts_models/multilingual/multi-dataset/xtts_v2/config.json \
     podcast_tts/configs/xtts_host_finetune.json
   ```
2. Edit `podcast_tts/configs/xtts_host_finetune.json` and update:
   * `output_path`: where checkpoints should be saved (e.g., `"output/xtts_host"`).
   * `run_name`: a descriptive label for TensorBoard logs.
   * The first entry in `datasets`:
     ```json
     "datasets": [
       {
         "name": "host_voice",
         "meta_file_train": "data/host_xtts/metadata.csv",
         "path": "data/host_xtts"
       }
     ]
     ```
   * `trainer.max_steps` or `trainer.max_epochs` for the amount of training you
     want (1 000–3 000 steps is a good starting point for <30 minutes of speech).
   * `trainer.save_checkpoints`: set to `true` to keep intermediate checkpoints.
   * (Optional) Freeze components to speed up training:
     ```json
     "model_args": {
       "freeze_text_encoder": true,
       "freeze_style_encoder": true,
       "freeze_speaker_encoder": false
     }
     ```
3. Launch training inside the virtual environment:
   ```bash
   python -m TTS.bin.train_tts --config_path podcast_tts/configs/xtts_host_finetune.json
   ```

Training on CPU is slow; consider enabling ROCm if available. The resulting
checkpoint directory contains `best_model.pth`. Point the synthesiser to it via
`--model /path/to/output/xtts_host/best_model.pth` and reuse your
`host_reference.wav` for cloning or the learned speaker embeddings inside the
checkpoint.

## 8. Format the transcript

* Save the dialogue as UTF-8 plain text (e.g. `transcripts/episode1.txt`).
* Separate each speaker's turn with a blank line, or enable sentence-level splitting with `--split-mode sentence`.
* Ensure the dialogue is already in chronological order; the script alternates speakers automatically and does not attempt to infer identity from context.

Example snippet:

```
I can't believe how quickly that model responded.

Well, you did phrase the prompt perfectly.

Maybe, but the nuance was all in the follow-up question.
```

## 9. Generate the podcast audio

```bash
python podcast_tts/synthesize_podcast.py transcripts/episode1.txt \
  --output-dir build/episode1 \
  --model tts_models/multilingual/multi-dataset/xtts_v2 \
  --language en \
  --speaker-wavs assets/voice_alex.wav assets/voice_ai.wav \
  --turn-threshold 1 \
  --progress \
  --gap-ms 150
```

Key options:

* `--speaker-wavs`: alternating list of reference clips. You can supply more than two to rotate through additional guests.
* `--speakers`: use built-in voice names if you prefer synthetic voices.
* `--turn-threshold`: increases beyond 1 when you expect the same person to speak for multiple turns in a row.
* `--split-mode sentence`: split by sentences instead of paragraphs when transcripts lack blank lines.
* `--emotion`: for models that support style tokens (e.g. `happy`, `sad`, `narration`).
* `--gap-ms`: amount of silence inserted between turns in the combined master mix. Increase for more breathing room.
* `--device`: force `cpu`, `cuda`, or `rocm`. The default `auto` picks the best available backend.

The script creates individual `turn_XXX.wav` files plus a `podcast_mix.wav` file that lines up the dialogue sequentially (with configurable silence). You can import the stems into any DAW for mastering.

## 10. Fine-tuning cadence and pacing

* **Timing gaps**: Insert ellipses (`...`) or stage directions like `(pause)` inside the transcript to encourage longer pauses. Coqui TTS honours punctuation strongly.
* **Emotion**: Experiment with alternative reference clips (e.g., excited vs. calm readings) for bigger emotional swings. You can also run multiple passes with different references for the same speaker and choose the best take.
* **Noise floor**: Keep reference audio free of background hum. The model will mirror the input timbre, including microphone coloration.

## 11. Automating production

For batch processing, create a shell script or Makefile that activates the virtual environment, runs `synthesize_podcast.py` for each transcript, and copies the resulting audio into your editing workflow. The Python script prints the location of the combined mix so it is easy to wire into larger pipelines.

## 12. Troubleshooting

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| `ModuleNotFoundError: No module named 'TTS'` | Virtual environment not active | Run `source .venv/bin/activate` before executing the script. |
| `RuntimeError: No audio turns generated` | Transcript lacks blank lines or punctuation | Add blank lines or use `--split-mode sentence`. |
| `RuntimeError: Error(s) in loading state_dict` | Model download incomplete | Delete `~/.local/share/tts` and re-run the download step. |
| Crackling or clipping in `podcast_mix.wav` | Mixed waveform exceeding 0 dBFS | Lower `turn-threshold`, reduce per-turn loudness via `--emotion narration`, or normalize in a DAW. |

## 13. Keeping everything offline

All models and dependencies are cached locally. Disconnect from the network after downloading the Python wheels and model files if you require an air-gapped workflow. The script itself performs no network requests.

---

Once you have the references, transcripts, and environment in place, you can generate audiobook-quality dialogue entirely on your Linux workstation without additional coding.
