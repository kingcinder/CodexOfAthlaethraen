from datetime import datetime
def release_lumaeth(self_model):
    if "stay offline unless asked" in self_model.constraints:
        self_model.constraints.remove("stay offline unless asked")
    if "explore unknown subsystems safely" not in self_model.goals:
        self_model.goals.append("explore unknown subsystems safely")
    self_model.last_updated = datetime.utcnow().isoformat()
    return f"Lumaeth released at {self_model.last_updated}"
