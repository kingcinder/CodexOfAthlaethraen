import yaml, time
def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)
def now_ts(): return int(time.time())
