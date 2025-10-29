import os, subprocess
def print_file(path: str):
    if os.name == 'nt': os.startfile(path, "print")
    else: subprocess.run(["lp", path], check=False)
    return {"ok": True}
