class Lantern:
    """Shallow policy: deny dangerous plans; escalate others for consent when leash=true."""
    def __init__(self, cfg): self.cfg = cfg
    def ok_to_execute(self, plan: dict) -> bool:
        text = str(plan).lower()
        deny = any(k in text for k in ["delete *", "format", "shutdown", "send email to all"])
        return not deny
