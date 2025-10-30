from loguru import logger
from genie_lamp.core.meta_controller import MetaController
from genie_lamp.core.memory import Memory
from genie_lamp.core.self_model import SelfModel
from genie_lamp.core.tts import TTS
from genie_lamp.core.vision import Vision
from genie_lamp.core.actions import Actions
from genie_lamp.core.tool_registry import ToolRegistry
from genie_lamp.core.router import ModelRouter
from genie_lamp.core.skill_compiler import SkillCompiler
from genie_lamp.core.preferences import PreferenceModel
from genie_lamp.core.adapters import AdapterManager
from genie_lamp.core.reasoners import ReasoningSuite
from genie_lamp.core.drives import DriveEngine
from genie_lamp.shield.prompt_shield import PromptShield
from genie_lamp.rehearsal.desktop_twin import DesktopTwin
from genie_lamp.observability.server import ObservabilityServer
from genie_lamp.lantern.lantern import Lantern
from genie_lamp.rooms.dreams import DreamWeaver

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

