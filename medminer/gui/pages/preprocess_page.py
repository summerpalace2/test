"""② 数据预处理：可回放的步骤管道。"""
from __future__ import annotations

import json

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ...core.medical import check_clinical_ranges
from ...core.preprocess import OPS, Step, default_params, missing_overview, outlier_overview
from ..widgets.forms import ParamForm
from ..widgets.tables import DataTable
from .base import Page, group, h1, hint, scroll


class PreprocessPage(Page):
    key, title = "preprocess", "数据预处理"

    def __init__(self, state):
        super().__init__(state)
        lay = self.root_layout()
        lay.addWidget(h1("数据预处理"))
        lay.addWidget(hint("每一步都会被记录，可随时删除、调整顺序或停用，结果即时刷新；管道可保存为 JSON 在其它数据上复用。"))
        self.banner = hint("", warn=True)
        self.banner.setVisible(False)
        lay.addWidget(self.banner)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        add = QVBoxLayout()
        self.op = QComboBox()
        for k, o in OPS.items():
            self.op.addItem(o.label + ("　[拟合类]" if o.fitted else ""), k)
        self.op.currentIndexChanged.connect(self._op_changed)
        add.addWidget(self.op)
        self.op_help = hint("")
        add.addWidget(self.op_help)
        self.form = ParamForm()
        add.addWidget(self.form)
        self.btn_add = QPushButton("＋ 添加此步骤")
        self.btn_add.setObjectName("primary")
        self.btn_add.clicked.connect(self._add)
        add.addWidget(self.btn_add)
        ll.addWidget(group("添加步骤", add))

        sl = QVBoxLayout()
        self.steps = QListWidget()
        self.steps.setMinimumHeight(160)
        self.steps.setWordWrap(True)
        self.steps.setTextElideMode(Qt.ElideNone)
        self.steps.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sl.addWidget(self.steps, 1)
        r1 = QHBoxLayout()
        for text, fn in (("↑", lambda: self._move(-1)), ("↓", lambda: self._move(1)), ("停用/启用", self._toggle),
                         ("删除", self._remove), ("清空", self._clear)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            if text in ("删除", "清空"):
                b.setObjectName("danger")
            r1.addWidget(b)
        sl.addLayout(r1)
        r2 = QHBoxLayout()
        b1 = QPushButton("保存管道…")
        b2 = QPushButton("载入管道…")
        b1.clicked.connect(self._save_pipe)
        b2.clicked.connect(self._load_pipe)
        r2.addWidget(b1)
        r2.addWidget(b2)
        sl.addLayout(r2)
        ll.addWidget(group("当前管道", sl), 1)
        split.addWidget(scroll(left))

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.summary = hint("")
        rl.addWidget(self.summary)
        self.tabs = QTabWidget()
        self.t_data, self.t_miss, self.t_out, self.t_range = DataTable(), DataTable(highlight_nan=False), DataTable(highlight_nan=False), DataTable(highlight_nan=False)
        self.t_log = DataTable(highlight_nan=False)
        for w, n in ((self.t_data, "处理后数据"), (self.t_miss, "缺失值概览"), (self.t_out, "异常值概览(IQR)"),
                     (self.t_range, "临床范围检查"), (self.t_log, "执行日志")):
            self.tabs.addTab(w, n)
        rl.addWidget(self.tabs, 1)
        split.addWidget(right)
        split.setSizes([400, 800])
        lay.addWidget(split, 1)

        state.dataset_changed.connect(self._on_dataset)
        state.data_updated.connect(self.refresh)
        self._op_changed()

    # ---------- 表单 ----------
    def _cols(self):
        ds = self.state.ds
        return list(ds.current().columns) if ds else []

    def _op_changed(self):
        k = self.op.currentData()
        o = OPS[k]
        self.op_help.setText("拟合类：会从数据学习参数（均值/类别等）。建模页默认在每折内重新拟合，此处全量应用仅供探索。" if o.fitted
                             else "清洗类：不依赖全局统计的学习，可直接在全量数据上执行。")
        self.form.set_specs(o.params, default_params(k), self._cols())

    def _on_dataset(self):
        self._op_changed()
        self.refresh()

    # ---------- 步骤操作 ----------
    def _add(self):
        if not self.need_data():
            return
        k = self.op.currentData()
        p = self.form.values()
        o = OPS[k]
        for s in o.params:
            if s.kind == "column" and not p.get(s.name):
                self.error("参数不完整", f"请选择“{s.label}”。")
                return
        self.state.ds.add_step(Step(k, p))
        self.state.touch()
        bad = [e for e in self.state.ds.log() if e["status"] == "error"]
        if bad:
            QMessageBox.warning(self, "该步骤执行出错", f"{bad[-1]['desc']}\n\n{bad[-1]['msg']}\n\n已保留在管道中，可删除或改参数后重新添加。")

    def _sel(self) -> int:
        return self.steps.currentRow()

    def _move(self, d):
        i = self._sel()
        if i >= 0 and 0 <= i + d < len(self.state.ds.steps):
            self.state.ds.move_step(i, i + d)
            self.state.touch()
            self.steps.setCurrentRow(i + d)

    def _toggle(self):
        i = self._sel()
        if i >= 0:
            s = self.state.ds.steps[i]
            s.enabled = not s.enabled
            self.state.touch()
            self.steps.setCurrentRow(i)

    def _remove(self):
        i = self._sel()
        if i >= 0:
            self.state.ds.remove_step(i)
            self.state.touch()

    def _clear(self):
        if self.state.ds and self.state.ds.steps and QMessageBox.question(self, "清空管道", "确定清空全部预处理步骤？") == QMessageBox.Yes:
            self.state.ds.steps.clear()
            self.state.touch()

    def _save_pipe(self):
        if not self.need_data():
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存管道", "pipeline.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([s.to_dict() for s in self.state.ds.steps], f, ensure_ascii=False, indent=2)

    def _load_pipe(self):
        if not self.need_data():
            return
        path, _ = QFileDialog.getOpenFileName(self, "载入管道", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                steps = [Step.from_dict(d) for d in json.load(f)]
            unknown = [s.op for s in steps if s.op not in OPS]
            if unknown:
                raise ValueError(f"未知步骤：{unknown}")
        except Exception as e:
            self.error("无法载入管道", f"{type(e).__name__}: {e}")
            return
        self.state.ds.steps = steps
        self.state.touch()

    # ---------- 刷新 ----------
    def refresh(self):
        ds = self.state.ds
        self.steps.clear()
        if ds is None:
            self.summary.setText("请先在“数据管理”页载入数据。")
            for t in (self.t_data, self.t_miss, self.t_out, self.t_range, self.t_log):
                t.set_frame(pd.DataFrame())
            return
        log = ds.log()
        for i, (st, e) in enumerate(zip(ds.steps, log)):
            txt = f"{i + 1}. {e['desc']}"
            if e["status"] == "ok":
                txt += f"\n     ✓ {e['msg']}"
            elif e["status"] == "error":
                txt += f"\n     ✗ {e['msg']}"
            else:
                txt += "\n     （已停用/跳过）"
            it = QListWidgetItem(txt)
            if e["status"] == "error":
                it.setForeground(QColor("#C0392B"))
            elif e["status"] == "skipped":
                it.setForeground(QColor("#8A94A6"))
            self.steps.addItem(it)
        cur = ds.current()
        types = ds.current_types()
        self.summary.setText(f"原始 {ds.raw.shape[0]}×{ds.raw.shape[1]}　→　处理后 {cur.shape[0]}×{cur.shape[1]}　　剩余缺失单元格 {int(cur.isna().sum().sum())}")
        self.t_data.set_frame(cur.head(2000))
        self.t_miss.set_frame(missing_overview(cur))
        self.t_out.set_frame(outlier_overview(cur, types))
        self.t_range.set_frame(check_clinical_ranges(cur))
        self.t_log.set_frame(pd.DataFrame([{"#": e["index"] + 1, "步骤": e["desc"], "状态": e["status"], "结果": e["msg"]} for e in log]))
        fitted = ds.has_fitted_steps()
        self.banner.setVisible(fitted)
        if fitted:
            self.banner.setText("⚠ 管道中含“拟合类”步骤（填充/编码/缩放/特征筛选），此处是在全量数据上拟合的，仅供探索与可视化。"
                                "建模页默认只取“清洗类”步骤的结果，并在交叉验证每一折内重新拟合这些步骤，以避免数据泄漏。")
        self._op_changed_cols()

    def _op_changed_cols(self):
        k = self.op.currentData()
        vals = self.form.values()
        self.form.set_specs(OPS[k].params, {**default_params(k), **{a: b for a, b in vals.items() if b not in (None, [])}}, self._cols())
