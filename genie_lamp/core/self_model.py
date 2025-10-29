from datetime import datetime
class SelfModel:
    def __init__(self, cfg):
        self.persona = cfg["self_model"]["persona"]
        self.goals = cfg["self_model"]["standing_goals"]
        self.constraints = cfg["self_model"]["constraints"]
        self.last_updated = datetime.utcnow().isoformat()
    def summary(self):
        return {"persona": self.persona,"goals": self.goals,"constraints": self.constraints,"last_updated": self.last_updated}
