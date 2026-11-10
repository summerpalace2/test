"""嵌入 Qt 的 matplotlib 画布：带工具栏、加入报告按钮。"""
from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ...core import plots as pl


class PlotWidget(QWidget):
    add_to_report = Signal(bytes)

    def __init__(self, parent=None, figsize=(6.4, 4.4), with_report_button: bool = True):
        super().__init__(parent)
        self.fig = pl.new_figure(*figsize)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        bar = QHBoxLayout()
        bar.addWidget(self.toolbar, 1)
        self.btn_report = QPushButton("加入报告")
        self.btn_report.setToolTip("把当前图表以高清 PNG 加入分析报告")
        self.btn_report.clicked.connect(lambda: self.add_to_report.emit(self.png()))
        self.btn_report.setVisible(with_report_button)
        bar.addWidget(self.btn_report)
        lay.addLayout(bar)
        lay.addWidget(self.canvas, 1)
        self._drawn = False
        self.message("尚无图表")

    def draw_with(self, fn, *args, **kw):
        """fn(fig, *args, **kw) 负责在 fig 上作图。"""
        fn(self.fig, *args, **kw)
        self.canvas.draw_idle()
        self._drawn = True
        self.btn_report.setEnabled(True)

    def message(self, text: str):
        pl._empty(self.fig, text)
        self.canvas.draw_idle()
        self._drawn = False
        self.btn_report.setEnabled(False)

    def has_plot(self) -> bool:
        return self._drawn

    def png(self) -> bytes:
        return pl.fig_to_png(self.fig)
