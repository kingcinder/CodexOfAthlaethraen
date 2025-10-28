# The Genie Lamp (Lumaeth) — Unified Build

This project merges the base framework and the recursive meta-controller into a single, runnable local assistant with:
- Self-referential recursion (Propose→Critic→Select→Execute→Reflect)
- Persistent memory (Chroma/FAISS + embeddings)
- Voice (TTS), Vision (OCR), Desktop Actions
- Safety Lantern (guardrails), Wake-word + faster-whisper STT
- Nightly "Dream compression" + optional DreamGlass GUI

## Quickstart
PowerShell:  ./scripts/setup_windows.ps1  
Linux:       bash ./scripts/setup_linux.sh  
Then:        python main.py
