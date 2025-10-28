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
