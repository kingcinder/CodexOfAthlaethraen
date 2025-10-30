import os, subprocess, webbrowser, pyautogui
class Actions:
    def __init__(self, cfg): self.cfg = cfg
    def open_url(self, url: str): webbrowser.open(url); return {"ok": True}
    def print_file(self, path: str):
        if os.name == "nt": os.startfile(path, "print")
        else: subprocess.run(["lp", path], check=False)
        return {"ok": True}
    def type_text(self, text: str, window_title: str = ""):
        allow = self.cfg.get("actions",{}).get("allowlist_windows",[])
        if allow and window_title and not any(a in window_title for a in allow):
            return {"ok": False, "error": "window not allowlisted"}
        pyautogui.write(text); return {"ok": True}
