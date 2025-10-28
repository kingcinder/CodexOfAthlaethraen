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
