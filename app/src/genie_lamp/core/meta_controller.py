import json
import time

from genie_lamp.core.utils import now_ts

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
