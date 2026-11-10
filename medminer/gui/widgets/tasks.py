"""后台任务：把耗时计算放到线程，界面保持响应，可取消。"""
from __future__ import annotations

import traceback
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QLabel, QMessageBox, QProgressBar, QPushButton, QWidget

from ...core.evaluate import Cancelled


class Worker(QThread):
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = self._fn(lambda a, b, m="": self.progress.emit(int(a), int(b), str(m)), lambda: self._cancel)
            if self._cancel:
                self.failed.emit("已取消")
            else:
                self.done.emit(res)
        except Cancelled:
            self.failed.emit("已取消")
        except Exception as e:  # 异常转成界面可显示的文字，同时保留完整堆栈到控制台
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


class TaskRunner(QObject):
    """绑定进度条/按钮的任务启动器。同一时刻只运行一个任务。"""

    def __init__(self, owner: QWidget, bar: QProgressBar | None = None, label: QLabel | None = None,
                 run_btn: QPushButton | None = None, cancel_btn: QPushButton | None = None):
        super().__init__(owner)
        self.owner, self.bar, self.label, self.run_btn, self.cancel_btn = owner, bar, label, run_btn, cancel_btn
        self.worker: Worker | None = None
        if cancel_btn:
            cancel_btn.clicked.connect(self.cancel)
            cancel_btn.setEnabled(False)
        if bar:
            bar.setVisible(False)

    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            if self.label:
                self.label.setText("正在取消…")

    def start(self, fn: Callable, on_done: Callable, on_fail: Callable | None = None, busy_text: str = "计算中…"):
        if self.busy():
            return
        self.worker = Worker(fn, self)
        if self.bar:
            self.bar.setVisible(True)
            self.bar.setRange(0, 0)  # 先显示忙碌态，收到进度后切换为确定态
        if self.label:
            self.label.setText(busy_text)
        if self.run_btn:
            self.run_btn.setEnabled(False)
        if self.cancel_btn:
            self.cancel_btn.setEnabled(True)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(lambda r: self._finish(on_done, r))
        self.worker.failed.connect(lambda m: self._fail(on_fail, m))
        self.worker.start()

    def _progress(self, a, b, msg):
        if self.bar and b > 0:
            self.bar.setRange(0, b)
            self.bar.setValue(a)
        if self.label and msg:
            self.label.setText(msg)

    def _reset(self):
        if self.bar:
            self.bar.setVisible(False)
        if self.run_btn:
            self.run_btn.setEnabled(True)
        if self.cancel_btn:
            self.cancel_btn.setEnabled(False)

    def _finish(self, cb, res):
        self._reset()
        if self.label:
            self.label.setText("完成")
        cb(res)

    def _fail(self, cb, msg):
        self._reset()
        if self.label:
            self.label.setText(msg)
        if cb:
            cb(msg)
        elif msg != "已取消":
            QMessageBox.warning(self.owner, "计算失败", msg)
