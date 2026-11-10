"""防回归：自己的控件类不得用属性/方法名遮蔽 Qt 成员。

真实事故：ModelPage 里 `self.metric = QComboBox()` 遮蔽了虚函数 QWidget::metric()，
Qt 在计算设备度量时回调它，得到 QComboBox 而不是可调用对象，直接抛 TypeError
（取决于 Qt 何时查询，所以只在特定环境/顺序下才暴露）。
"""
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QWidget

# 有意覆盖的 Qt 虚函数
INTENTIONAL = {"keyPressEvent", "dropEvent", "dragEnterEvent", "rowCount", "columnCount", "data", "headerData", "sort", "run"}


def _is_signal(v) -> bool:
    return type(v).__name__ in ("SignalInstance", "Signal")


def test_no_member_shadows_qt_api(monkeypatch, tmp_path):
    from medminer.ai import config as aicfg
    monkeypatch.setattr(aicfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(aicfg, "CONFIG_FILE", tmp_path / "c.json")
    app = QApplication.instance() or QApplication([])
    from medminer.gui.main_window import MainWindow
    win = MainWindow()
    qt_names = set(dir(QWidget)) | set(dir(QObject))
    problems = []
    for obj in [win] + win.findChildren(QObject):
        cls = type(obj)
        if not cls.__module__.startswith("medminer"):
            continue  # 只检查我们自己写的类
        for k, v in vars(obj).items():
            if not k.startswith("_") and k in qt_names and not _is_signal(v):
                problems.append(f"{cls.__name__}.{k}（实例属性遮蔽 Qt 成员）")
        for k, v in cls.__dict__.items():
            if not k.startswith("_") and k in qt_names and callable(v) and k not in INTENTIONAL:
                problems.append(f"{cls.__name__}.{k}()（方法与 Qt 成员同名）")
    assert not problems, "\n".join(problems)
