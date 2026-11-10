"""页面基类与小工具。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QLabel, QMessageBox, QScrollArea, QVBoxLayout, QWidget

from ..state import AppState


def h1(text: str) -> QLabel:
    l = QLabel(text)
    l.setObjectName("h1")
    return l


def hint(text: str, warn: bool = False, ok: bool = False) -> QLabel:
    l = QLabel(text)
    l.setObjectName("warn" if warn else "ok" if ok else "hint")
    l.setWordWrap(True)
    return l


def group(title: str, layout) -> QGroupBox:
    g = QGroupBox(title)
    g.setLayout(layout)
    return g


def scroll(widget: QWidget, width: int | None = None) -> QScrollArea:
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setWidget(widget)
    s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    if width:
        s.setFixedWidth(width)
    return s


class Page(QWidget):
    key = ""
    title = ""

    def __init__(self, state: AppState):
        super().__init__()
        self.setObjectName("page")
        self.state = state

    def need_data(self, quiet: bool = False) -> bool:
        if self.state.ds is None:
            if not quiet:
                QMessageBox.information(self, "请先载入数据", "请先在“数据管理”页打开数据文件，或加载示例数据。")
            return False
        return True

    def toast(self, msg: str):
        self.state.status.emit(msg)

    def error(self, title: str, msg: str):
        QMessageBox.warning(self, title, msg)

    def root_layout(self, margins=(16, 14, 16, 12)) -> QVBoxLayout:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(*margins)
        lay.setSpacing(8)
        return lay
