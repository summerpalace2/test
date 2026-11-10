"""由 ParamSpec 自动生成的参数表单，以及列选择控件。"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from ...core.params import ParamSpec


class ColumnList(QListWidget):
    """带复选框的列列表。"""
    changed = Signal()

    def __init__(self, parent=None, height: int | None = None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.itemChanged.connect(lambda *_: self.changed.emit())
        if height:
            self.setMaximumHeight(height)
            self.setMinimumHeight(min(height, 90))

    def set_columns(self, cols: list[str], checked: list[str] | None = None, labels: dict[str, str] | None = None):
        keep = set(self.checked()) if checked is None else set(checked)
        self.blockSignals(True)
        self.clear()
        for c in cols:
            it = QListWidgetItem((labels or {}).get(c, c))
            it.setData(Qt.UserRole, c)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if c in keep else Qt.Unchecked)
            self.addItem(it)
        self.blockSignals(False)
        self.changed.emit()

    def checked(self) -> list[str]:
        return [self.item(i).data(Qt.UserRole) for i in range(self.count()) if self.item(i).checkState() == Qt.Checked]

    def set_checked(self, cols: list[str]):
        s = set(cols)
        self.blockSignals(True)
        for i in range(self.count()):
            self.item(i).setCheckState(Qt.Checked if self.item(i).data(Qt.UserRole) in s else Qt.Unchecked)
        self.blockSignals(False)
        self.changed.emit()

    def check_all(self, on: bool = True):
        self.set_checked([self.item(i).data(Qt.UserRole) for i in range(self.count())] if on else [])


class ColumnPickerRow(QWidget):
    """ColumnList + 全选/清空按钮。"""
    changed = Signal()

    def __init__(self, height: int = 150, parent=None):
        super().__init__(parent)
        self.list = ColumnList(height=height)
        self.list.changed.connect(self.changed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        for text, on in (("全选", True), ("清空", False)):
            b = QPushButton(text)
            b.setStyleSheet("padding:2px 8px")
            b.clicked.connect(lambda _=False, o=on: self.list.check_all(o))
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

    def set_columns(self, *a, **k):
        self.list.set_columns(*a, **k)

    def checked(self):
        return self.list.checked()

    def set_checked(self, c):
        self.list.set_checked(c)


class ColumnsButton(QPushButton):
    """点击弹出对话框选择多列；按钮文字显示已选数量。"""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("（不选）", parent)
        self._cols: list[str] = []
        self._sel: list[str] = []
        self.clicked.connect(self._open)

    def set_columns(self, cols: list[str], selected: list[str] | None = None):
        self._cols = list(cols)
        self._sel = [c for c in (selected if selected is not None else self._sel) if c in cols]
        self._refresh()

    def selected(self) -> list[str]:
        return list(self._sel)

    def _refresh(self):
        self.setText("（不选）" if not self._sel else (", ".join(self._sel) if len(self._sel) <= 3 else f"已选 {len(self._sel)} 列"))

    def _open(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("选择列")
        dlg.resize(320, 420)
        lay = QVBoxLayout(dlg)
        cl = ColumnPickerRow(height=10000)
        cl.set_columns(self._cols, self._sel)
        lay.addWidget(cl)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec():
            self._sel = cl.checked()
            self._refresh()
            self.changed.emit()


class ParamForm(QWidget):
    """根据 ParamSpec 列表生成表单。"""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setLabelAlignment(Qt.AlignLeft)
        self._widgets: dict[str, tuple[ParamSpec, Any]] = {}
        self._columns: list[str] = []

    def set_specs(self, specs: list[ParamSpec], values: dict | None = None, columns: list[str] | None = None):
        if columns is not None:
            self._columns = columns
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()
        values = values or {}
        for s in specs:
            w = self._make(s, values.get(s.name, s.default))
            label = QLabel(s.label)
            if s.help:
                label.setToolTip(s.help)
                w.setToolTip(s.help)
            self._form.addRow(label, w)
            self._widgets[s.name] = (s, w)

    def _make(self, s: ParamSpec, v):
        if s.kind in ("int", "float"):
            sp = QSpinBox() if s.kind == "int" else QDoubleSpinBox()
            lo = -1e9 if s.min is None else s.min
            hi = 1e9 if s.max is None else s.max
            if s.kind == "int":
                sp.setRange(int(lo), int(hi))
                sp.setSingleStep(int(s.step or 1))
            else:
                step = s.step or 0.1
                sp.setDecimals(10 if step < 0.001 else 4 if step < 0.1 else 2)
                sp.setRange(lo, hi)
                sp.setSingleStep(step)
            sp.setKeyboardTracking(False)
            if v is not None:
                sp.setValue(v)
            if s.nullable:
                box = QWidget()
                h = QHBoxLayout(box)
                h.setContentsMargins(0, 0, 0, 0)
                cb = QCheckBox("不限/空")
                cb.setChecked(v is None)
                sp.setEnabled(v is not None)
                cb.toggled.connect(lambda on, sp=sp: (sp.setEnabled(not on), self.changed.emit()))
                h.addWidget(sp, 1)
                h.addWidget(cb)
                box._sp, box._cb = sp, cb
                sp.valueChanged.connect(lambda *_: self.changed.emit())
                return box
            sp.valueChanged.connect(lambda *_: self.changed.emit())
            return sp
        if s.kind == "choice":
            cb = QComboBox()
            for c in s.choices:
                cb.addItem("无" if c is None else str(c), c)
            i = cb.findData(v)
            cb.setCurrentIndex(max(0, i))
            cb.currentIndexChanged.connect(lambda *_: self.changed.emit())
            return cb
        if s.kind == "bool":
            c = QCheckBox()
            c.setChecked(bool(v))
            c.toggled.connect(lambda *_: self.changed.emit())
            return c
        if s.kind == "text":
            e = QLineEdit("" if v is None else str(v))
            e.textChanged.connect(lambda *_: self.changed.emit())
            return e
        if s.kind == "column":
            cb = QComboBox()
            cb.addItems(self._columns)
            if v in self._columns:
                cb.setCurrentText(v)
            cb.currentIndexChanged.connect(lambda *_: self.changed.emit())
            return cb
        if s.kind == "columns":
            b = ColumnsButton()
            b.set_columns(self._columns, list(v or []))
            b.changed.connect(self.changed)
            return b
        raise ValueError(s.kind)

    def values(self) -> dict[str, Any]:
        out = {}
        for name, (s, w) in self._widgets.items():
            if s.kind in ("int", "float"):
                if s.nullable and w._cb.isChecked():
                    out[name] = None
                else:
                    sp = w._sp if s.nullable else w
                    out[name] = sp.value()
            elif s.kind == "choice":
                out[name] = w.currentData()
            elif s.kind == "bool":
                out[name] = w.isChecked()
            elif s.kind == "text":
                out[name] = w.text()
            elif s.kind == "column":
                out[name] = w.currentText() or None
            elif s.kind == "columns":
                out[name] = w.selected()
        return out
