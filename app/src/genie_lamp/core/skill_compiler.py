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
