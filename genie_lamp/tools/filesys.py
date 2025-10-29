import os, shutil
def move(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    return {"ok": True}
