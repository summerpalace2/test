"""离屏渲染辅助：用法 python scripts/shot.py <脚本>；脚本里可用 win/state/app/grab(name)。
仅用于开发时验证界面（无显示器环境）。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication  # noqa: E402

from medminer.gui.main_window import MainWindow  # noqa: E402
from medminer.gui.theme import apply_theme  # noqa: E402

OUT = Path(os.environ.get("SHOT_DIR", "/tmp/shots"))
OUT.mkdir(parents=True, exist_ok=True)
app = QApplication.instance() or QApplication([])
apply_theme(app)


def pump(n=5):
    for _ in range(n):
        app.processEvents()


def grab(win, name):
    pump()
    win.grab().save(str(OUT / f"{name}.png"))
    print("saved", OUT / f"{name}.png")


def wait_for(cond, timeout=60):
    import time
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    return cond()
