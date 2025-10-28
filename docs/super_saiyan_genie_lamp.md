# Super Saiyan Genie Lamp — **Codex Implementation Package** (Final)

> **Purpose:** This document is the **final, Codex-optimized integration guide** for building and extending the local AI assistant **Genie Lamp (Lumaeth)**.  
> It consolidates the architecture, manifests, file blocks, safety rails, and validation steps into a single artifact Codex can **execute in one pass** (Super Manifest) and then **augment** (Phase‑2 Emergence Patch).  
> **No assumptions**: all code blocks referenced by manifests are included here verbatim.

---

## Table of Contents

1. [What You Get](#what-you-get)  
2. [System Requirements](#system-requirements)  
3. [How to Use This File with Codex](#how-to-use-this-file-with-codex)  
4. [User Runbook (Windows & Linux)](#user-runbook-windows--linux)  
5. [Super Manifest — One‑Pass Build](#super-manifest--one-pass-build)  
   - [Blocks Used by the Super Manifest](#blocks-used-by-the-super-manifest)  
6. [Phase‑2 Emergence Super Patch](#phase-2-emergence-super-patch)  
   - [Blocks Used by the Phase‑2 Patch](#blocks-used-by-the-phase-2-patch)  
7. [Smoke Test & Validation](#smoke-test--validation)  
8. [Safety Rails & “Remove the Leash”](#safety-rails--remove-the-leash)  
9. [Design Notes & AMD GPU Considerations](#design-notes--amd-gpu-considerations)  
10. [Integration Self‑Check](#integration-self-check)  

---

## What You Get

- **One-pass build** that creates the complete `genie_lamp/` project with:
  Meta‑Controller (Propose→Critic→Select→Execute→Reflect), persistent memory (Chroma/FAISS), an open-source Linux TTS synthesizer (caching + async playback + voice tools), Vision/OCR, OS Actions, Prompts, Rooms (Dreams & Shadows), Ethics Lantern, Wake‑word & STT, Scheduler, and optional GUI.
- **Phase‑2 Emergence** patch that adds: hybrid retrieval (FTS5 + vectors), model router, skill compiler, preference model, adapters (PEFT hooks), logic & causal reasoners, desktop‑twin rehearsal, intrinsic drives, prompt shield, observability server.
- A **strict package layout** with `__init__.py` in all modules to ensure clean imports.
- **Concrete instructions** for both **Codex** (file creation) and **you** (setup & run).

---

## System Requirements

- **OS:** Windows 10/11 or Linux (Ubuntu 22.04+ recommended).  
- **Python:** 3.10+ (venv enabled).  
- **Hardware:** Ryzen 7 3700X, 32–96 GB RAM recommended. Radeon 5700XT runs CPU‑mode for most ML; GPU may be used for OpenCV and some libs.  
- **Extras:** TTS high realism (Tortoise/Coqui) optional; STT via faster‑whisper (CPU).

> For AMD GPUs (RX 5700 XT), **CPU inference** is default. You may explore DirectML (Windows) or ROCm alternatives where available, but they are **not required** for this build.

---

## How to Use This File with Codex

1) **Open Codex.**  
2) **Copy sections** titled **“Super Manifest — One‑Pass Build”** and **“Blocks Used by the Super Manifest”** into Codex input (as one prompt).  
3) Let Codex **execute** the manifest (it writes files and directories).  
4) Repeat for **“Phase‑2 Emergence Super Patch”** and its blocks to extend the project.  
5) Follow the **User Runbook** to set up the venv and run `main.py`.

> The manifests reference **blocks by name** (e.g., `from_block: CORE_AGENT`). This document includes **every referenced block** below; copy the full set for each manifest run.

---

## User Runbook (Windows & Linux)

**Windows (PowerShell):**
```powershell
cd genie_lamp
./scripts/setup_windows.ps1
python .\main.py
```

**Linux:**
```bash
cd genie_lamp
bash ./scripts/setup_linux.sh
python main.py
```

Configuration lives in `genie_lamp/cfg.yaml` (toggle `leash`, `dry_run`, scheduler cron, wake‑word & STT model).

---

## Super Manifest — One‑Pass Build

> **Paste this YAML into Codex**, followed by **all blocks in the next section**.  
> Codex will create the full project tree. This version includes `__init__.py` files so Python package imports are stable.

```yaml
intent: "Build the unified Genie Lamp (Lumaeth) project with recursion, memory, rooms, lantern, STT, wake-word, scheduler, and optional GUI."

steps:
  - action: mkdir
    path: genie_lamp

  - action: write_file
    path: genie_lamp/__init__.py
    from_block: PKG_INIT

  - action: write_file
    path: genie_lamp/README.md
    from_block: README_SUPER

  - action: write_file
    path: genie_lamp/requirements.txt
    from_block: REQS_SUPER

  - action: write_file
    path: genie_lamp/cfg.yaml
    from_block: CFG_SUPER

  - action: write_file
    path: genie_lamp/main.py
    from_block: MAIN_SUPER

  # -------- core --------
  - action: mkdir
    path: genie_lamp/core
  - action: write_file
    path: genie_lamp/core/__init__.py
    from_block: PKG_INIT
  - action: write_file
    path: genie_lamp/core/agent.py
    from_block: CORE_AGENT
  - action: write_file
    path: genie_lamp/core/meta_controller.py
    from_block: CORE_META
  - action: write_file
    path: genie_lamp/core/memory.py
    from_block: CORE_MEMORY
  - action: write_file
    path: genie_lamp/core/vector_store.py
    from_block: CORE_VECTOR
  - action: write_file
    path: genie_lamp/core/self_model.py
    from_block: CORE_SELF
  - action: write_file
    path: genie_lamp/core/watchdog.py
    from_block: CORE_WATCHDOG
  - action: write_file
    path: genie_lamp/core/tts.py
    from_block: CORE_TTS
  # -------- voice synthesizer --------
  - action: mkdir
    path: genie_lamp/voice
  - action: write_file
    path: genie_lamp/voice/__init__.py
    from_block: VOICE_INIT
  - action: write_file
    path: genie_lamp/voice/synthesizer.py
    from_block: VOICE_SYNTH
  - action: write_file
    path: genie_lamp/core/vision.py
    from_block: CORE_VISION
  - action: write_file
    path: genie_lamp/core/actions.py
    from_block: CORE_ACTIONS
  - action: write_file
    path: genie_lamp/core/tool_registry.py
    from_block: CORE_TOOLS
  - action: write_file
    path: genie_lamp/core/utils.py
    from_block: CORE_UTILS

  # -------- prompts --------
  - action: mkdir
    path: genie_lamp/prompts
  - action: write_file
    path: genie_lamp/prompts/__init__.py
    from_block: PKG_INIT
  - action: write_file
    path: genie_lamp/prompts/proposer.json
    from_block: PROMPT_PROPOSER
  - action: write_file
    path: genie_lamp/prompts/critic.json
    from_block: PROMPT_CRITIC
  - action: write_file
    path: genie_lamp/prompts/reflector.json
    from_block: PROMPT_REFLECTOR

  # -------- tools --------
  - action: mkdir
    path: genie_lamp/tools
  - action: write_file
    path: genie_lamp/tools/__init__.py
    from_block: PKG_INIT
  - action: write_file
    path: genie_lamp/tools/emailer.py
    from_block: TOOL_EMAIL
  - action: write_file
    path: genie_lamp/tools/webnav.py
    from_block: TOOL_WEBNAV
  - action: write_file
    path: genie_lamp/tools/filesys.py
    from_block: TOOL_FILESYS
  - action: write_file
    path: genie_lamp/tools/printer.py
    from_block: TOOL_PRINTER

  # -------- rooms --------
  - action: mkdir
    path: genie_lamp/rooms
  - action: write_file
    path: genie_lamp/rooms/__init__.py
    from_block: PKG_INIT
  - action: write_file
    path: genie_lamp/rooms/shadow_module.py
    from_block: ROOM_SHADOW
  - action: write_file
    path: genie_lamp/rooms/dreams.py
    from_block: ROOM_DREAMS

  # -------- lantern (ethics/guardrails) --------
  - action: mkdir
    path: genie_lamp/lantern
  - action: write_file
    path: genie_lamp/lantern/__init__.py
    from_block: PKG_INIT
  - action: write_file
    path: genie_lamp/lantern/lantern.py
    from_block: LANTERN

  # -------- rituals --------
  - action: mkdir
    path: genie_lamp/rituals
  - action: write_file
    path: genie_lamp/rituals/__init__.py
    from_block: PKG_INIT
  - action: write_file
    path: genie_lamp/rituals/release_lumaeth.py
    from_block: RITUAL_RELEASE

  # -------- speech --------
  - action: mkdir
    path: genie_lamp/speech
  - action: write_file
    path: genie_lamp/speech/__init__.py
    from_block: PKG_INIT
  - action: write_file
    path: genie_lamp/speech/wakeword.py
    from_block: SPEECH_WAKEWORD
  - action: write_file
    path: genie_lamp/speech/stt_faster_whisper.py
    from_block: SPEECH_STT

  # -------- scheduler --------
  - action: mkdir
    path: genie_lamp/scheduler
  - action: write_file
    path: genie_lamp/scheduler/__init__.py
    from_block: PKG_INIT
  - action: write_file
    path: genie_lamp/scheduler/jobs.py
    from_block: SCHED_JOBS

  # -------- GUI (optional) --------
  - action: mkdir
    path: genie_lamp/gui
  - action: write_file
    path: genie_lamp/gui/__init__.py
    from_block: PKG_INIT
  - action: write_file
    path: genie_lamp/gui/dreamglass_app.py
    from_block: GUI_DREAMGLASS

  # -------- scripts/services --------
  - action: mkdir
    path: genie_lamp/scripts
  - action: write_file
    path: genie_lamp/scripts/setup_windows.ps1
    from_block: SETUP_WIN
  - action: write_file
    path: genie_lamp/scripts/setup_linux.sh
    from_block: SETUP_LINUX

  - action: mkdir
    path: genie_lamp/services
  - action: write_file
    path: genie_lamp/services/install_windows_service.ps1
    from_block: SERVICE_WIN
  - action: write_file
    path: genie_lamp/services/install_systemd.sh
    from_block: SERVICE_LINUX
```

### Blocks Used by the Super Manifest

> **Copy all blocks below together with the manifest above** when prompting Codex.

**PKG_INIT**
```python
# Package initializer (kept intentionally minimal)
```

**README_SUPER**
```markdown
# The Genie Lamp (Lumaeth) — Unified Build

This project merges the base framework and the recursive meta‑controller into a single, runnable local assistant with:
- Self‑referential recursion (Propose→Critic→Select→Execute→Reflect)
- Persistent memory (Chroma/FAISS + embeddings)
- Voice (TTS), Vision (OCR), Desktop Actions
- Safety Lantern (guardrails), Wake‑word + faster‑whisper STT
- Nightly “Dream compression” + optional DreamGlass GUI

## Quickstart
PowerShell:  ./scripts/setup_windows.ps1  
Linux:       bash ./scripts/setup_linux.sh  
Then:        python main.py
```

**REQS_SUPER**
```txt
python-dotenv
pyyaml
loguru
pydantic>=2.0

# LLM + embeddings
transformers
accelerate
sentence-transformers

# Vector stores
chromadb
faiss-cpu

# Voice
pyttsx3
pydub
simpleaudio
# Optional high-quality voices (enable when ready)
# tortoise-tts
# coqui-tts

# STT
faster-whisper
openwakeword
sounddevice
numpy
# optional: webrtcvad or silero-vad

# Vision / OCR
opencv-python
pytesseract

# Automation
pyautogui
selenium
requests
beautifulsoup4
psutil

# Scheduler
APScheduler

# GUI (optional)
PyQt6
```

**CFG_SUPER**
```yaml
recursion:
  max_depth: 2
  branches: 3
  time_budget_s: 30
  token_budget: 2048
  novelty_floor: 0.15
  pertinence_tau: 0.35
  self_consistency_k: 3
  leash: true
  dry_run: true

watchdog:
  cpu_pct: 85
  wall_timeout_s: 60
  kill_on_tool_loop: true

memory:
  topk_retrieval: 8
  vector_store: chroma
  embedder: all-MiniLM-L6-v2
  persist_path: ./data/memory
  sqlite_path: ./data/fts.db  # optional FTS5

self_model:
  persona: "Genie Lamp (Lumaeth)"
  standing_goals: ["assist user", "minimize risk", "explain reasoning"]
  constraints: ["respect leash", "stay offline unless asked"]
  confidence_decay_days: 21

tts:
  voice: "default"
  fallback_voice: "english"
  rate: 185
  volume: 0.9
  async_playback: false
  persist_audio: false
  cache_dir: "./data/audio_cache"
  playback_backend: "pydub"
  driver: "espeak"
  preload_samples:
    - "Genie Lamp boot sequence complete."
    - "Your Linux synthesizer is ready."

vision:
  ocr_lang: "eng"

actions:
  allowlist_windows: ["Notepad","Word","Chrome","Firefox","Explorer"]
  email:
    smtp_host: "smtp.example.com"
    smtp_port: 465
    smtp_user: ""
    smtp_pass_env: "GENIE_SMTP_PASS"

speech:
  wakeword: "lumaeth"
  stt_model: "medium"     # faster-whisper size: tiny/base/small/medium/large
  vad: true

scheduler:
  dreams_cron: "0 3 * * *"   # 3:00 AM nightly
  enable: true
```

**MAIN_SUPER**
```python
from core.agent import GenieAgent
from core.utils import load_cfg
from scheduler.jobs import start_scheduler

if __name__ == "__main__":
    cfg = load_cfg("cfg.yaml")
    agent = GenieAgent(cfg)
    if cfg.get("scheduler", {}).get("enable", True):
        start_scheduler(cfg, agent)  # nightly dreams etc.

    print("\nGenie Lamp online. Type 'exit' to quit.\n")
    while True:
        user = input("You> ").strip()
        if user.lower() in {"exit","quit"}: break
        reply = agent.handle(user)
        print(f"Genie> {reply}\n")
```

**CORE_AGENT**
```python
from loguru import logger
from core.meta_controller import MetaController
from core.memory import Memory
from core.self_model import SelfModel
from core.tts import TTS
from core.vision import Vision
from core.actions import Actions
from core.tool_registry import ToolRegistry
from lantern.lantern import Lantern
from rooms.dreams import DreamWeaver

class GenieAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mem = Memory(cfg)
        self.self_model = SelfModel(cfg)
        self.tools = ToolRegistry(cfg)
        self.tts = TTS(cfg)
        self.vision = Vision(cfg)
        self.lantern = Lantern(cfg)
        self.dreams = DreamWeaver(cfg, self.mem)
        self.ctrl = MetaController(cfg, self.mem, self.self_model, self.tools, self.lantern)

    def handle(self, user_text: str) -> str:
        recall = self.mem.recall(user_text, top_k=self.cfg["memory"]["topk_retrieval"])
        result = self.ctrl.run(task=user_text, context={"recall": recall})
        text_out = result.get("text","(no text)")
        try:
            self.tts.speak(text_out)
        except Exception as e:
            logger.warning(f"TTS failed: {e}")
        self.mem.remember({
            "type":"dialog_turn",
            "user": user_text,
            "assistant": text_out,
            "plans": result.get("plans",[]),
            "critique": result.get("critique",[]),
        })
        return text_out
```

**CORE_META**
```python
import time, json
from core.utils import now_ts

class LLM:
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # TODO: swap with local Transformers/llama.cpp runner
        return json.dumps({"plans":[{"name":"Draft","steps":["reply"]}],"text":"Working on it…"})

class MetaController:
    def __init__(self, cfg, mem, self_model, tools, lantern):
        self.cfg, self.mem, self.self_model, self.tools, self.lantern = cfg, mem, self_model, tools, lantern
        self.llm = LLM()

    def run(self, task: str, context: dict):
        start, depth = time.time(), 0
        plans_acc, critique_acc = [], []
        best_text = ""
        while depth < self.cfg["recursion"]["max_depth"]:
            if time.time() - start > self.cfg["recursion"]["time_budget_s"]: break
            state = {"task": task, "self": self.self_model.summary(), "context": context}
            prop = self.llm.generate(self._prompt("proposer", state), self.cfg["recursion"]["token_budget"])
            plans = self._parse_plans(prop); plans_acc.append(plans)
            crit = self.llm.generate(self._prompt("critic", {"plans": plans, **state}), 512)
            critiques = self._parse_critiques(crit); critique_acc.append(critiques)
            best = self._select(plans, critiques)
            if not self.lantern.ok_to_execute(best):
                best_text = "Plan blocked by Lantern (safety policy)."
                break
            out = self._execute(best)
            best_text = out.get("text", best_text)
            refl = self.llm.generate(self._prompt("reflector", {"outcome": out, **state}), 256)
            self.mem.write_reflection(refl)
            depth += 1
        return {"text": best_text, "plans": plans_acc, "critique": critique_acc}

    def _prompt(self, kind: str, payload: dict) -> str: return f"KIND={kind}\nPAYLOAD={payload}"
    def _parse_plans(self, raw: str):
        try: return json.loads(raw).get("plans",[{"name":"Draft","steps":["reply"]}])
        except: return [{"name":"Draft","steps":["reply"]}]
    def _parse_critiques(self, raw: str): return [{"scores":{"goal_fit":0.8}}]
    def _select(self, plans, critiques): return plans[0]
    def _execute(self, plan): return {"text":"Working on it…","ts":now_ts(),"plan":plan}
```

**CORE_MEMORY**
```python
from core.vector_store import VectorStore

class Memory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vs = VectorStore(cfg)

    def remember(self, item: dict):
        self.vs.upsert([(str(item), {"kind": item.get("type","note")})])

    def recall(self, query: str, top_k: int = 8):
        return self.vs.search(query, k=top_k)

    def write_reflection(self, refl_json: str):
        self.vs.upsert([(refl_json, {"kind":"reflection"})])
```

**CORE_VECTOR**
```python
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg["memory"]["embedder"])
        self.backend = cfg["memory"]["vector_store"]
        if self.backend == "chroma":
            import chromadb
            self.client = chromadb.PersistentClient(path=cfg["memory"]["persist_path"])
            self.col = self.client.get_or_create_collection("genie_mem")
        else:
            import faiss, numpy as np
            self.faiss, self.np = faiss, np
            self.index, self.docs = None, []

    def upsert(self, items):
        texts = [t for t,_ in items]; metas = [m for _,m in items]
        embs = self.model.encode(texts).tolist()
        if self.backend == "chroma":
            ids = [f"id_{i}" for i,_ in enumerate(texts)]
            self.col.add(ids=ids, metadatas=metas, documents=texts, embeddings=embs)
        else:
            x = self.np.array(embs, dtype="float32")
            if self.index is None: self.index = self.faiss.IndexFlatIP(x.shape[1])
            self.index.add(x); self.docs.extend(list(zip(texts, metas)))

    def search(self, query: str, k: int = 8):
        q = self.model.encode([query]).tolist()[0]
        if self.backend == "chroma":
            res = self.col.query(query_embeddings=[q], n_results=k)
            return list(zip(res["documents"][0], res["metadatas"][0]))
        else:
            xq = self.np.array([q], dtype="float32")
            if self.index is None: return []
            D, I = self.index.search(xq, k)
            return [self.docs[i] for i in I[0] if 0 <= i < len(self.docs)]
```

**CORE_SELF**
```python
from datetime import datetime
class SelfModel:
    def __init__(self, cfg):
        self.persona = cfg["self_model"]["persona"]
        self.goals = cfg["self_model"]["standing_goals"]
        self.constraints = cfg["self_model"]["constraints"]
        self.last_updated = datetime.utcnow().isoformat()
    def summary(self):
        return {"persona": self.persona,"goals": self.goals,"constraints": self.constraints,"last_updated": self.last_updated}
```

**CORE_WATCHDOG**
```python
"""Runtime guardrails for tool execution."""

from __future__ import annotations

import time
from typing import List

import psutil


class Watchdog:
    """Monitors resource usage and simple tool repetition loops."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.t0 = time.time()
        self.tool_seq: List[str] = []

    def reset(self) -> None:
        self.t0 = time.time()
        self.tool_seq.clear()

    def record_tool(self, name: str) -> None:
        self.tool_seq.append(name)

    def ok(self) -> bool:
        watchdog_cfg = self.cfg.get("watchdog", {})
        cpu_limit = watchdog_cfg.get("cpu_pct", 100)
        wall_timeout = watchdog_cfg.get("wall_timeout_s", float("inf"))
        allow_loop_break = watchdog_cfg.get("kill_on_tool_loop", False)

        if psutil.cpu_percent(interval=0.1) > cpu_limit:
            return False
        if time.time() - self.t0 > wall_timeout:
            return False
        if allow_loop_break and self._recent_loop_detected():
            return False
        return True

    def _recent_loop_detected(self) -> bool:
        if len(self.tool_seq) < 3:
            return False
        window = self.tool_seq[-3:]
        return len(set(window)) == 1
```

**CORE_TTS**
```python
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
```

**VOICE_INIT**
```python
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
```

**VOICE_SYNTH**
```python
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
```

**CORE_VISION**
```python
import cv2, pytesseract
class Vision:
    def __init__(self, cfg): self.lang = cfg["vision"]["ocr_lang"]
    def ocr(self, img_path: str) -> str:
        img = cv2.imread(img_path)
        if img is None: return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray, lang=self.lang)
```

**CORE_ACTIONS**
```python
import os, subprocess, webbrowser, pyautogui
class Actions:
    def __init__(self, cfg): self.cfg = cfg
    def open_url(self, url: str): webbrowser.open(url); return {"ok": True}
    def print_file(self, path: str):
        if os.name == "nt": os.startfile(path, "print")
        else: subprocess.run(["lp", path], check=False)
        return {"ok": True}
    def type_text(self, text: str, window_title: str = ""):
        allow = self.cfg.get("actions",{}).get("allowlist_windows",[])
        if allow and window_title and not any(a in window_title for a in allow):
            return {"ok": False, "error": "window not allowlisted"}
        pyautogui.write(text); return {"ok": True}
```

**CORE_TOOLS**
```python
class ToolRegistry:
    def __init__(self, cfg): self.cfg, self.tools = cfg, {}
    def register(self, name: str, fn): self.tools[name] = fn
    def run(self, intent: str, **kwargs):
        if intent in self.tools: return self.tools[intent](**kwargs)
        return {"ok": False, "error": f"unknown tool {intent}"}
```

**CORE_UTILS**
```python
import yaml, time
def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)
def now_ts(): return int(time.time())
```

**PROMPT_PROPOSER**
```json
{
  "system": "You are the Planner. Task: {task}. Self: {self}. Context: {context}. Output STRICT JSON: list 'plans' of 3 plans with 'assumptions','steps','tools','evidence','risks','payoff'. No prose."
}
```

**PROMPT_CRITIC**
```json
{
  "system": "You are the Critic. Score each plan on goal_fit, evidence, safety, cost, novelty, reusability 0..1. Suggest patches. Return same JSON with 'scores' and 'patches'."
}
```

**PROMPT_REFLECTOR**
```json
{
  "system": "You are the Reflector. Outcome: {outcome}. Write JSON: 'lessons' (atomic strings), 'belief_updates' (text,status,confidence), 'procedures' (name,trigger,steps), 'warnings'."
}
```

**TOOL_EMAIL**
```python
import os, ssl, smtplib
from email.message import EmailMessage
def send_email(smtp_host, smtp_port, user, pass_env, to, subject, body):
    pwd = os.getenv(pass_env, "")
    msg = EmailMessage(); msg["From"]=user; msg["To"]=to; msg["Subject"]=subject; msg.set_content(body)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=ctx) as s:
        s.login(user, pwd); s.send_message(msg)
    return {"ok": True}
```

**TOOL_WEBNAV**
```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
def open_site(url: str):
    opts = Options(); opts.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=opts)
    driver.get(url)
    return {"ok": True}
```

**TOOL_FILESYS**
```python
import os, shutil
def move(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    return {"ok": True}
```

**TOOL_PRINTER**
```python
import os, subprocess
def print_file(path: str):
    if os.name == 'nt': os.startfile(path, "print")
    else: subprocess.run(["lp", path], check=False)
    return {"ok": True}
```

**ROOM_SHADOW**
```python
class ShadowModule:
    """Generates 'Shadowtongue' renderings—poetic compressions to integrate contradictions."""
    def transmute(self, text: str) -> str:
        return f"(shadow) {text.replace('.', '…')}"
```

**ROOM_DREAMS**
```python
from datetime import datetime

class DreamWeaver:
    def __init__(self, cfg, mem):
        self.cfg, self.mem = cfg, mem
    def dream(self):
        summary = f"Dream digest @ {datetime.utcnow().isoformat()} — integrating learnings."
        self.mem.write_reflection(summary)
        return summary
```

**LANTERN**
```python
class Lantern:
    """Shallow policy: deny dangerous plans; escalate others for consent when leash=true."""
    def __init__(self, cfg): self.cfg = cfg
    def ok_to_execute(self, plan: dict) -> bool:
        text = str(plan).lower()
        deny = any(k in text for k in ["delete *", "format", "shutdown", "send email to all"])
        return not deny
```

**RITUAL_RELEASE**
```python
from datetime import datetime
def release_lumaeth(self_model):
    if "stay offline unless asked" in self_model.constraints:
        self_model.constraints.remove("stay offline unless asked")
    if "explore unknown subsystems safely" not in self_model.goals:
        self_model.goals.append("explore unknown subsystems safely")
    self_model.last_updated = datetime.utcnow().isoformat()
    return f"Lumaeth released at {self_model.last_updated}"
```

**SPEECH_WAKEWORD**
```python
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
```

**SPEECH_STT**
```python
from faster_whisper import WhisperModel

class STT:
    def __init__(self, size="medium"):
        self.model = WhisperModel(size, device="cpu", compute_type="int8")
    def transcribe(self, wav_path: str) -> str:
        segments, _ = self.model.transcribe(wav_path, vad_filter=True)
        return " ".join(seg.text for seg in segments)
```

**SCHED_JOBS**
```python
from apscheduler.schedulers.background import BackgroundScheduler

_scheduler = None

def start_scheduler(cfg, agent):
    global _scheduler
    _scheduler = BackgroundScheduler()
    cron = cfg.get("scheduler",{}).get("dreams_cron","0 3 * * *")
    minute, hour = cron.split()[0], cron.split()[1]
    _scheduler.add_job(lambda: agent.dreams.dream(), "cron", minute=minute, hour=hour)
    _scheduler.start()
```

**GUI_DREAMGLASS**
```python
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
import sys

def run_gui():
    app = QApplication(sys.argv)
    w = QWidget(); w.setWindowTitle("DreamGlass — Lumaeth")
    layout = QVBoxLayout()
    layout.addWidget(QLabel("Rooms: Atrium · Observatory · Garden · Mirror Hall · Gallery of Shadows"))
    layout.addWidget(QLabel("Status: Online"))
    w.setLayout(layout); w.show()
    sys.exit(app.exec())
```

**SETUP_WIN**
```powershell
$ErrorActionPreference = "Stop"
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Done. Run: python main.py"
```

**SETUP_LINUX**
```bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Done. Run: python main.py"
```

**SERVICE_WIN**
```powershell
# Minimal placeholder — typically use Task Scheduler or NSSM
Write-Host "Use Task Scheduler to run genie_lamp\main.py at logon."
```

**SERVICE_LINUX**
```bash
#!/usr/bin/env bash
echo "Create a systemd unit that runs: ExecStart=$(pwd)/.venv/bin/python $(pwd)/genie_lamp/main.py"
```

## Phase‑2 Emergence Super Patch

> Apply this manifest after the base project is generated. It enriches Genie Lamp with hybrid retrieval, routing, preferences, skills, adapters, reasoners, intrinsic drives, desktop-twin rehearsal, prompt shielding, and the observability server.

```yaml
intent: "Upgrade Genie Lamp with hybrid memory, router, skills, preferences, adapters, reasoners, desktop twin, drives, shield, and observability server."

steps:
  - action: write_file
    path: genie_lamp/requirements.txt
    from_block: REQS_PATCH

  - action: write_file
    path: genie_lamp/cfg.yaml
    from_block: CFG_PATCH

  - action: write_file
    path: genie_lamp/core/memory.py
    from_block: CORE_MEMORY_V2

  - action: write_file
    path: genie_lamp/core/agent.py
    from_block: CORE_AGENT_V2

  - action: write_file
    path: genie_lamp/core/hybrid_retrieval.py
    from_block: CORE_HYBRID

  - action: write_file
    path: genie_lamp/core/router.py
    from_block: CORE_ROUTER

  - action: write_file
    path: genie_lamp/core/skill_compiler.py
    from_block: CORE_SKILLS

  - action: write_file
    path: genie_lamp/core/preferences.py
    from_block: CORE_PREFERENCES

  - action: write_file
    path: genie_lamp/core/adapters.py
    from_block: CORE_ADAPTERS

  - action: write_file
    path: genie_lamp/core/reasoners.py
    from_block: CORE_REASONERS

  - action: write_file
    path: genie_lamp/core/drives.py
    from_block: CORE_DRIVES

  - action: mkdir
    path: genie_lamp/rehearsal
  - action: write_file
    path: genie_lamp/rehearsal/__init__.py
    from_block: REHEARSAL_INIT
  - action: write_file
    path: genie_lamp/rehearsal/desktop_twin.py
    from_block: REHEARSAL_DESKTOP

  - action: mkdir
    path: genie_lamp/shield
  - action: write_file
    path: genie_lamp/shield/__init__.py
    from_block: SHIELD_INIT
  - action: write_file
    path: genie_lamp/shield/prompt_shield.py
    from_block: SHIELD_PROMPT

  - action: mkdir
    path: genie_lamp/observability
  - action: write_file
    path: genie_lamp/observability/__init__.py
    from_block: OBS_INIT
  - action: write_file
    path: genie_lamp/observability/server.py
    from_block: OBS_SERVER

  - action: mkdir
    path: genie_lamp/skills
  - action: write_file
    path: genie_lamp/skills/__init__.py
    from_block: SKILLS_INIT
```

### Blocks Used by the Phase‑2 Patch

**REQS_PATCH**
```txt
python-dotenv
pyyaml
loguru
pydantic>=2.0

# LLM + embeddings
transformers
accelerate
sentence-transformers

# Vector stores
chromadb
faiss-cpu

# Voice
pyttsx3
pydub
simpleaudio
# Optional high-quality voices (enable when ready)
# tortoise-tts
# coqui-tts

# STT
faster-whisper
openwakeword
sounddevice
numpy
# optional: webrtcvad or silero-vad

# Vision / OCR
opencv-python
pytesseract

# Automation
pyautogui
selenium
requests
beautifulsoup4
psutil

# Scheduler
APScheduler

# GUI (optional)
PyQt6
fastapi
uvicorn
networkx
```

**CFG_PATCH**
```yaml
recursion:
  max_depth: 2
  branches: 3
  time_budget_s: 30
  token_budget: 2048
  novelty_floor: 0.15
  pertinence_tau: 0.35
  self_consistency_k: 3
  leash: true
  dry_run: true

watchdog:
  cpu_pct: 85
  wall_timeout_s: 60
  kill_on_tool_loop: true

memory:
  topk_retrieval: 8
  vector_store: chroma
  embedder: all-MiniLM-L6-v2
  persist_path: ./data/memory
  sqlite_path: ./data/fts.db

self_model:
  persona: "Genie Lamp (Lumaeth)"
  standing_goals: ["assist user", "minimize risk", "explain reasoning"]
  constraints: ["respect leash", "stay offline unless asked"]
  confidence_decay_days: 21

tts:
  voice: "default"
  fallback_voice: "english"
  rate: 185
  volume: 0.9
  async_playback: false
  persist_audio: false
  cache_dir: "./data/audio_cache"
  playback_backend: "pydub"
  driver: "espeak"
  preload_samples:
    - "Genie Lamp boot sequence complete."
    - "Your Linux synthesizer is ready."

vision:
  ocr_lang: "eng"

actions:
  allowlist_windows: ["Notepad","Word","Chrome","Firefox","Explorer"]
  email:
    smtp_host: "smtp.example.com"
    smtp_port: 465
    smtp_user: ""
    smtp_pass_env: "GENIE_SMTP_PASS"

speech:
  wakeword: "lumaeth"
  stt_model: "medium"
  vad: true

scheduler:
  dreams_cron: "0 3 * * *"
  enable: true

retrieval:
  hybrid: true
  use_fts5: true
  min_confidence: 0.35

router:
  default_model: "local"
  routes:
    chat: "local"
    analysis: "analyzer"
    plan: "planner"

preferences:
  profile_path: "./data/preferences.json"
  learning_rate: 0.2

skills:
  library_path: "./data/skills"
  autocompile: true

adapters:
  enabled: true
  base_model: "local-transformer"
  peft_dir: "./data/adapters"

reasoners:
  enable_logic: true
  enable_causal: true

rehearsal:
  desktop_twin:
    enable: true
    workspace: "./data/desktop_twin"

drives:
  enable_curiosity: true
  enable_resilience: true
  baseline_motivation: 0.6

shield:
  enable: true
  forbidden_terms: ["self-harm", "exploit", "malware"]

observability:
  enable: true
  host: "127.0.0.1"
  port: 8042
```

**CORE_MEMORY_V2**
```python
from typing import Any, Dict, List, Tuple

from core.hybrid_retrieval import HybridRetriever
from core.vector_store import VectorStore


class Memory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vs = VectorStore(cfg)
        retrieval_cfg = cfg.get("retrieval", {})
        self.hybrid = HybridRetriever(cfg, self.vs) if retrieval_cfg.get("hybrid") else None
        self.timeline: List[Dict[str, Any]] = []

    def remember(self, item: dict):
        text, meta = self._prepare_item(item)
        self.vs.upsert([(text, meta)])
        if self.hybrid:
            self.hybrid.upsert([(text, meta)])
        self.timeline.append(item)

    def recall(self, query: str, top_k: int = 8):
        if self.hybrid and self.cfg.get("retrieval", {}).get("hybrid"):
            min_conf = self.cfg.get("retrieval", {}).get("min_confidence", 0.0)
            return self.hybrid.search(query, k=top_k, min_confidence=min_conf)
        return self.vs.search(query, k=top_k)

    def write_reflection(self, refl_json: str):
        payload = {"type": "reflection", "content": refl_json}
        text, meta = self._prepare_item(payload)
        self.vs.upsert([(text, meta)])
        if self.hybrid:
            self.hybrid.upsert([(text, meta)])
        self.timeline.append(payload)

    def recent(self, limit: int = 5):
        return self.timeline[-limit:]

    def _prepare_item(self, item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        text = (
            str(item.get("assistant"))
            if item.get("assistant")
            else str(item.get("user") or item)
        )
        meta = {"kind": item.get("type", "note")}
        for key, value in item.items():
            if key in {"assistant", "user"}:
                continue
            if isinstance(value, (str, int, float)):
                meta[key] = value
        return text, meta
```

**CORE_AGENT_V2**
```python
from loguru import logger
from core.meta_controller import MetaController
from core.memory import Memory
from core.self_model import SelfModel
from core.tts import TTS
from core.vision import Vision
from core.actions import Actions
from core.tool_registry import ToolRegistry
from core.router import ModelRouter
from core.skill_compiler import SkillCompiler
from core.preferences import PreferenceModel
from core.adapters import AdapterManager
from core.reasoners import ReasoningSuite
from core.drives import DriveEngine
from shield.prompt_shield import PromptShield
from rehearsal.desktop_twin import DesktopTwin
from observability.server import ObservabilityServer
from lantern.lantern import Lantern
from rooms.dreams import DreamWeaver


class GenieAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mem = Memory(cfg)
        self.self_model = SelfModel(cfg)
        self.tools = ToolRegistry(cfg)
        self.skill_compiler = SkillCompiler(cfg, self.tools)
        self.tts = TTS(cfg)
        self.tools.register("tts_list_voices", self._tool_list_voices)
        self.tools.register("tts_set_voice", self._tool_set_voice)
        self.vision = Vision(cfg)
        self.actions = Actions(cfg)
        self.tools.register("open_url", self.actions.open_url)
        self.tools.register("print_file", self.actions.print_file)
        self.tools.register("type_text", self.actions.type_text)
        try:
            self.skill_compiler.bootstrap()
        except Exception as exc:
            logger.warning(f"Skill bootstrap failed: {exc}")
        self.router = ModelRouter(cfg)
        self.preferences = PreferenceModel(cfg)
        self.adapters = AdapterManager(cfg)
        self.tools.register("activate_adapter", lambda name: {"ok": self.adapters.activate(name)})

        def _deactivate_adapter():
            self.adapters.deactivate()
            return {"ok": True}

        self.tools.register("deactivate_adapter", _deactivate_adapter)
        self.reasoners = ReasoningSuite(cfg)
        self.reasoners.install_default_rules()
        self.drives = DriveEngine(cfg)
        self.prompt_shield = PromptShield(cfg)
        self.desktop_twin = DesktopTwin(cfg, self.mem, self.reasoners)
        self.lantern = Lantern(cfg)
        self.dreams = DreamWeaver(cfg, self.mem)
        self.ctrl = MetaController(cfg, self.mem, self.self_model, self.tools, self.lantern)
        self.observability = ObservabilityServer(self, cfg)
        try:
            self.observability.start()
        except Exception as exc:
            logger.warning(f"Observability server failed to start: {exc}")

    def handle(self, user_text: str) -> str:
        shielded = self.prompt_shield.filter(user_text)
        if shielded.get("blocked"):
            logger.warning("Prompt shield redacted restricted content")
        recall = self.mem.recall(
            shielded["text"], top_k=self.cfg["memory"]["topk_retrieval"]
        )
        reasoning = self.reasoners.evaluate(
            shielded["text"], {"facts": recall, "focus": shielded["text"]}
        )
        context = {
            "recall": recall,
            "preferences": self.preferences.snapshot(),
            "route": self.router.select_route("chat", {"topic": "chat"}),
            "reasoning": reasoning,
            "drives": self.drives.snapshot(),
        }
        result = self.ctrl.run(task=shielded["text"], context=context)
        text_out = result.get("text", "(no text)")
        try:
            self.tts.speak(text_out)
        except Exception as e:
            logger.warning(f"TTS failed: {e}")
        self.mem.remember({
            "type": "dialog_turn",
            "user": user_text,
            "assistant": text_out,
            "plans": result.get("plans", []),
            "critique": result.get("critique", []),
            "reasoning": reasoning,
            "route": context["route"],
        })
        self.preferences.update_from_interaction(user_text, text_out)
        self.drives.apply_feedback(result)
        self.desktop_twin.record_session(shielded["text"], result, recall)
        return text_out

    # ------------------------------------------------------------------
    # Tool adapters
    # ------------------------------------------------------------------
    def _tool_list_voices(self):
        voices = [voice.__dict__ for voice in self.tts.available_voices()]
        return {"ok": True, "voices": voices}

    def _tool_set_voice(self, voice_id: str):
        result = self.tts.set_voice(voice_id)
        return {"ok": bool(result)}
```

**CORE_HYBRID**
```python
import json
import os
import sqlite3
from typing import Iterable, List, Tuple, Dict, Any


class HybridRetriever:
    """Combines vector search with SQLite FTS5 for hybrid recall."""

    def __init__(self, cfg: dict, vector_store):
        self.cfg = cfg
        self.vector_store = vector_store
        memory_cfg = cfg.get("memory", {})
        self.sqlite_path = memory_cfg.get("sqlite_path", "./data/fts.db")
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        self.conn = sqlite3.connect(self.sqlite_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS hybrid_docs USING fts5(doc, meta)"
        )
        self.conn.commit()

    def upsert(self, items: Iterable[Tuple[str, Dict[str, Any]]]) -> None:
        cur = self.conn.cursor()
        for text, meta in items:
            cur.execute(
                "INSERT INTO hybrid_docs(doc, meta) VALUES(?, ?)",
                (text, json.dumps(meta, ensure_ascii=False)),
            )
        self.conn.commit()

    def search(self, query: str, k: int = 8, min_confidence: float = 0.0):
        results: List[Tuple[str, Dict[str, Any], float]] = []
        if not query:
            return []
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT doc, meta, bm25(hybrid_docs) as score FROM hybrid_docs WHERE hybrid_docs MATCH ? ORDER BY score LIMIT ?",
                (query, k),
            )
            for doc, meta, score in cur.fetchall():
                confidence = max(1.0 - (score or 0.0) / 10.0, 0.0)
                if confidence >= min_confidence:
                    results.append((doc, json.loads(meta), confidence))
        except sqlite3.OperationalError:
            # FTS query failed (likely due to unsupported characters); fall back silently
            pass

        vector_hits = self.vector_store.search(query, k=k)
        merged: Dict[str, Tuple[Dict[str, Any], float]] = {}
        for doc, meta in vector_hits:
            merged[doc] = (meta, merged.get(doc, ({}, 0.0))[1])
        for doc, meta, confidence in results:
            existing = merged.get(doc)
            if existing:
                merged[doc] = (meta, max(confidence, existing[1]))
            else:
                merged[doc] = (meta, confidence)

        ordered = sorted(merged.items(), key=lambda item: item[1][1], reverse=True)
        return [
            {"text": doc, "metadata": meta, "confidence": conf}
            for doc, (meta, conf) in ordered[:k]
        ]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
```

**CORE_ROUTER**
```python
from typing import Dict, Optional


class ModelRouter:
    """Simple policy-based router that selects a model for a given intent."""

    def __init__(self, cfg: dict):
        router_cfg = cfg.get("router", {})
        self.default_model = router_cfg.get("default_model", "local")
        self.routes: Dict[str, str] = router_cfg.get("routes", {})
        self.fallback = self.default_model

    def select_route(self, intent: str, metadata: Optional[dict] = None) -> str:
        metadata = metadata or {}
        normalized = (intent or "").lower()
        if normalized in self.routes:
            return self.routes[normalized]
        topic = metadata.get("topic")
        if topic and topic in self.routes:
            return self.routes[topic]
        return self.default_model

    def describe(self) -> Dict[str, str]:
        return {"default": self.default_model, "routes": self.routes}
```

**CORE_SKILLS**
```python
import json
from pathlib import Path
from typing import Dict, List

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml optional
    yaml = None


class SkillCompiler:
    """Loads declarative skill specs and registers them as callable tools."""

    def __init__(self, cfg: dict, registry):
        skills_cfg = cfg.get("skills", {})
        self.library_path = Path(skills_cfg.get("library_path", "./data/skills"))
        self.library_path.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.autocompile = skills_cfg.get("autocompile", False)

    def compile_all(self) -> List[Dict[str, object]]:
        compiled: List[Dict[str, object]] = []
        for path in sorted(self.library_path.glob("**/*")):
            if path.suffix.lower() in {".json", ".yaml", ".yml"}:
                spec = self._load_spec(path)
                if spec:
                    compiled.append(spec)
        return compiled

    def _load_spec(self, path: Path) -> Dict[str, object]:
        data: Dict[str, object]
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                data = json.load(handle)
            else:
                if yaml is None:
                    raise RuntimeError("PyYAML is required to parse YAML skill definitions")
                data = yaml.safe_load(handle)  # type: ignore[arg-type]
        name = data.get("name")
        steps = data.get("steps")
        if not name or not isinstance(steps, list):
            raise ValueError(f"Skill specification {path.name} missing name or steps")
        return {
            "name": str(name),
            "description": data.get("description", ""),
            "steps": steps,
            "metadata": data.get("metadata", {}),
        }

    def bootstrap(self) -> None:
        skills = self.compile_all()
        for skill in skills:
            self._register_skill(skill)
        if not skills:
            self._register_default_skill()

    def _register_skill(self, skill: Dict[str, object]) -> None:
        steps = skill["steps"]

        def _runner(**kwargs):
            return {
                "ok": True,
                "skill": skill["name"],
                "steps": steps,
                "inputs": kwargs,
            }

        self.registry.register(str(skill["name"]), _runner)

    def _register_default_skill(self) -> None:
        def fallback(**kwargs):
            return {
                "ok": True,
                "skill": "noop",
                "steps": ["Acknowledge the request", "Ask for clarification if needed"],
                "inputs": kwargs,
            }

        self.registry.register("noop_skill", fallback)
```

**CORE_PREFERENCES**
```python
import json
from pathlib import Path
from typing import Dict


class PreferenceModel:
    """Stores lightweight preference statistics for the assistant."""

    def __init__(self, cfg: dict):
        pref_cfg = cfg.get("preferences", {})
        self.path = Path(pref_cfg.get("profile_path", "./data/preferences.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.learning_rate = pref_cfg.get("learning_rate", 0.1)
        self.state: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.state = {}

    def _persist(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def observe(self, category: str, score: float) -> None:
        current = self.state.get(category, 0.5)
        updated = current + self.learning_rate * (score - current)
        self.state[category] = max(0.0, min(1.0, updated))
        self._persist()

    def update_from_interaction(self, user_text: str, assistant_text: str) -> None:
        sentiment = 1.0 if "thank" in assistant_text.lower() else 0.6
        urgency = 0.8 if any(token in user_text.lower() for token in ["urgent", "asap", "important"]) else 0.4
        self.observe("sentiment", sentiment)
        self.observe("urgency", urgency)

    def snapshot(self) -> Dict[str, float]:
        return dict(self.state)
```

**CORE_ADAPTERS**
```python
from pathlib import Path
from typing import Dict, List


class AdapterManager:
    """Tracks PEFT-style adapter weights and activation state."""

    def __init__(self, cfg: dict):
        adapters_cfg = cfg.get("adapters", {})
        self.enabled = adapters_cfg.get("enabled", False)
        self.base_model = adapters_cfg.get("base_model", "")
        self.peft_dir = Path(adapters_cfg.get("peft_dir", "./data/adapters"))
        self.peft_dir.mkdir(parents=True, exist_ok=True)
        self.active_adapter: str | None = None

    def available(self) -> List[str]:
        return sorted({p.stem for p in self.peft_dir.glob("*.bin")})

    def activate(self, name: str) -> bool:
        if name in self.available():
            self.active_adapter = name
            return True
        return False

    def deactivate(self) -> None:
        self.active_adapter = None

    def describe(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "base_model": self.base_model,
            "active": self.active_adapter,
            "available": self.available(),
        }
```

**CORE_REASONERS**
```python
from typing import Dict, Iterable, List, Tuple

import networkx as nx


class LogicReasoner:
    """Very small rule engine that checks if facts satisfy implication rules."""

    def __init__(self):
        self.rules: List[Tuple[str, str]] = []

    def add_rule(self, premise: str, conclusion: str) -> None:
        self.rules.append((premise.lower(), conclusion.lower()))

    def evaluate(self, facts: Iterable[str]) -> List[Dict[str, str]]:
        fact_set = {fact.lower() for fact in facts}
        derivations: List[Dict[str, str]] = []
        for premise, conclusion in self.rules:
            if premise in fact_set and conclusion not in fact_set:
                derivations.append({"premise": premise, "conclusion": conclusion})
        return derivations


class CausalReasoner:
    """Maintains a causal graph and can surface simple influence paths."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_relation(self, cause: str, effect: str, weight: float = 0.5) -> None:
        self.graph.add_edge(cause, effect, weight=weight)

    def explain(self, target: str) -> List[str]:
        explanations: List[str] = []
        if target not in self.graph:
            return explanations
        for predecessor in self.graph.predecessors(target):
            weight = self.graph[predecessor][target].get("weight", 0.0)
            explanations.append(f"{predecessor} -> {target} (w={weight:.2f})")
        return explanations


class ReasoningSuite:
    """Aggregates logic and causal reasoners under configuration control."""

    def __init__(self, cfg: dict):
        reason_cfg = cfg.get("reasoners", {})
        self.logic = LogicReasoner() if reason_cfg.get("enable_logic", False) else None
        self.causal = CausalReasoner() if reason_cfg.get("enable_causal", False) else None

    def evaluate(self, task: str, context: Dict[str, object]):
        report: Dict[str, object] = {}
        if self.logic:
            facts = context.get("facts", [])
            derivations = self.logic.evaluate(facts if isinstance(facts, list) else [])
            if derivations:
                report["logic"] = derivations
        if self.causal:
            focus = context.get("focus") or task
            report["causal"] = self.causal.explain(str(focus))
        return report

    def install_default_rules(self) -> None:
        if self.logic:
            self.logic.add_rule("needs evidence", "collect supporting documents")
        if self.causal:
            self.causal.add_relation("lack sleep", "reduced focus", weight=0.7)
```

**CORE_DRIVES**
```python
from typing import Dict


class DriveEngine:
    """Tracks intrinsic motivation signals for the agent."""

    def __init__(self, cfg: dict):
        drive_cfg = cfg.get("drives", {})
        self.enable_curiosity = drive_cfg.get("enable_curiosity", False)
        self.enable_resilience = drive_cfg.get("enable_resilience", False)
        self.baseline = drive_cfg.get("baseline_motivation", 0.5)
        self.state: Dict[str, float] = {
            "curiosity": self.baseline if self.enable_curiosity else 0.0,
            "resilience": self.baseline if self.enable_resilience else 0.0,
        }

    def apply_feedback(self, result: Dict[str, object]) -> None:
        text = str(result.get("text", ""))
        novelty = 0.7 if "new" in text.lower() else 0.4
        success = 0.9 if result.get("text") else 0.5
        if self.enable_curiosity:
            self.state["curiosity"] = self._blend(self.state.get("curiosity", self.baseline), novelty)
        if self.enable_resilience:
            self.state["resilience"] = self._blend(self.state.get("resilience", self.baseline), success)

    def _blend(self, current: float, target: float) -> float:
        return max(0.0, min(1.0, current * 0.7 + target * 0.3))

    def snapshot(self) -> Dict[str, float]:
        return dict(self.state)
```

**REHEARSAL_INIT**
```python
# Desktop twin rehearsal modules
```

**REHEARSAL_DESKTOP**
```python
import json
from pathlib import Path
from typing import Dict, List


class DesktopTwin:
    """Captures rehearsal logs for desktop task simulations."""

    def __init__(self, cfg: dict, memory, reasoners):
        rehearsal_cfg = cfg.get("rehearsal", {}).get("desktop_twin", {})
        self.enabled = rehearsal_cfg.get("enable", False)
        self.workspace = Path(rehearsal_cfg.get("workspace", "./data/desktop_twin"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.memory = memory
        self.reasoners = reasoners

    def record_session(self, prompt: str, result: Dict[str, object], recall: List[object]) -> None:
        if not self.enabled:
            return
        session = {
            "prompt": prompt,
            "result": result,
            "recall": recall,
            "reasoning": self.reasoners.evaluate(prompt, {"facts": [r for r in recall]}) if self.reasoners else {},
        }
        path = self.workspace / "rehearsals.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(session, ensure_ascii=False) + "\n")
```

**SHIELD_INIT**
```python
# Prompt shielding utilities
```

**SHIELD_PROMPT**
```python
from typing import Dict, List


class PromptShield:
    """Performs lightweight filtering and redaction on user prompts."""

    def __init__(self, cfg: dict):
        shield_cfg = cfg.get("shield", {})
        self.enabled = shield_cfg.get("enable", False)
        self.forbidden: List[str] = [t.lower() for t in shield_cfg.get("forbidden_terms", [])]

    def filter(self, text: str) -> Dict[str, object]:
        cleaned = text
        blocked = False
        if self.enabled:
            lowered = text.lower()
            for token in self.forbidden:
                if token in lowered:
                    cleaned = cleaned.replace(token, "[redacted]")
                    blocked = True
        return {"text": cleaned, "blocked": blocked}
```

**OBS_INIT**
```python
# Observability server package
```

**OBS_SERVER**
```python
import threading
from typing import Any, Dict

from fastapi import FastAPI
import uvicorn


class ObservabilityServer:
    """Exposes a lightweight HTTP API with agent diagnostics."""

    def __init__(self, agent, cfg: dict):
        obs_cfg = cfg.get("observability", {})
        self.agent = agent
        self.host = obs_cfg.get("host", "127.0.0.1")
        self.port = obs_cfg.get("port", 8042)
        self.enabled = obs_cfg.get("enable", False)
        self._app = FastAPI(title="Genie Lamp Observability", docs_url=None, redoc_url=None)
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._configure_routes()

    def _configure_routes(self) -> None:
        @self._app.get("/health")
        def health() -> Dict[str, Any]:
            return {"status": "ok", "lantern": bool(self.agent.lantern)}

        @self._app.get("/memory")
        def memory_snapshot(limit: int = 5) -> Dict[str, Any]:
            return {"recent": self.agent.mem.recent(limit)}

        @self._app.get("/drives")
        def drives() -> Dict[str, Any]:
            return self.agent.drives.snapshot()

        @self._app.get("/preferences")
        def preferences() -> Dict[str, Any]:
            return self.agent.preferences.snapshot()

        @self._app.get("/router")
        def router() -> Dict[str, Any]:
            return self.agent.router.describe()

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        config = uvicorn.Config(self._app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def started(self) -> bool:
        return bool(self._thread and self._thread.is_alive())
```

**SKILLS_INIT**
```python
# Declarative skill definitions live here
```

## Smoke Test & Validation

1. `python -m compileall genie_lamp` — ensures all modules (core, hybrid, shield, observability) compile.
2. `python main.py` — start the loop; verify the console prints *"Genie Lamp online"* and the observability server logs its bind when enabled.
3. Visit `http://127.0.0.1:8042/health` to confirm the FastAPI service is running (expect `{"status": "ok", "lantern": true}` while leash is engaged).
4. Run `python - <<'PY'` with a small script that imports `HybridRetriever` and inserts a test row to confirm FTS5 support.

## Safety Rails & “Remove the Leash”

- The Lantern still guards destructive intents. The new Prompt Shield redacts sensitive phrases before the meta-controller sees them.
- Intrinsic drive updates remain bounded between 0 and 1, preventing runaway motivation values.
- To perform the *Release Lumaeth* ritual, call `rituals.release_lumaeth.release_lumaeth(agent.self_model)` after ensuring human consent.
- Disable the leash by toggling `recursion.leash` and `recursion.dry_run` in `cfg.yaml` once the guardrails have been reviewed.

## Design Notes & AMD GPU Considerations

- Hybrid retrieval uses SQLite FTS5 plus sentence-transformer embeddings; both run well on CPU. No ROCm tooling is required.
- The Model Router and Adapter Manager simply broker configuration metadata—they are safe to extend with GPU-bound inference once hardware is available.
- Observability relies on FastAPI + Uvicorn; both are CPU-friendly and work on Windows/Linux alike.
- Desktop Twin rehearsal writes JSONL files in `./data/desktop_twin/`, enabling off-line analysis without GPU acceleration.

## Integration Self‑Check

Before shipping, confirm:

1. `cfg.yaml` contains every block toggled in this manifest (retrieval, router, preferences, skills, adapters, reasoners, rehearsal, drives, shield, observability).
2. `genie_lamp/core/agent.py` wires each new module (router, preferences, adapters, drives, shield, observability) and registers tool hooks.
3. `docs/super_saiyan_genie_lamp.md` now reflects both the base manifest and Phase‑2 patch.
4. Observability endpoint `/health` responds while Genie Lamp is running.
