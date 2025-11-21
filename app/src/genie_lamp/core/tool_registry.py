class ToolRegistry:
    def __init__(self, cfg): self.cfg, self.tools = cfg, {}
    def register(self, name: str, fn): self.tools[name] = fn
    def run(self, intent: str, **kwargs):
        if intent in self.tools: return self.tools[intent](**kwargs)
        return {"ok": False, "error": f"unknown tool {intent}"}
