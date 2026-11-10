"""① 数据管理：导入、预览、列类型/说明/目标列设置。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                               QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from ...core import io as mio
from ...core.dataset import Dataset, load_project, save_project
from ...core.medical import check_clinical_ranges
from ...core.sample_data import SAMPLE_NAME, make_heart_sample
from ...core.schema import IGNORE, LABEL_TO_TYPE, TYPE_LABELS
from ..widgets.tables import DataTable
from .base import Page, group, h1, hint

TYPE_ORDER = ["numeric", "categorical", "binary", "text", "datetime", "ignore"]


class DataPage(Page):
    key, title = "data", "数据管理"

    def __init__(self, state):
        super().__init__(state)
        self._updating = False
        self._path: str | None = None
        lay = self.root_layout()
        lay.addWidget(h1("数据管理"))
        lay.addWidget(hint("支持 CSV / TSV / TXT / DATA / Excel / JSON，自动识别编码、分隔符与表头，“?”等占位符按缺失值处理。也可直接把文件拖进窗口。"))

        bar = QHBoxLayout()
        self.btn_open = QPushButton("打开数据文件…")
        self.btn_open.setObjectName("primary")
        self.btn_sample = QPushButton("加载示例数据（心脏病·模拟）")
        self.btn_open_proj = QPushButton("打开项目…")
        self.btn_save_proj = QPushButton("保存项目…")
        self.sheet = QComboBox()
        self.sheet.setVisible(False)
        self.sheet.setToolTip("Excel 工作表")
        for b in (self.btn_open, self.btn_sample, self.btn_open_proj, self.btn_save_proj):
            bar.addWidget(b)
        bar.addWidget(QLabel("工作表："))
        self.lbl_sheet = bar.itemAt(bar.count() - 1).widget()
        self.lbl_sheet.setVisible(False)
        bar.addWidget(self.sheet)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.info = hint("尚未载入数据。")
        lay.addWidget(self.info)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.table = DataTable()
        ll.addWidget(QLabel("数据预览（红底=缺失；点击表头可排序，右键可复制/导出）"))
        ll.addWidget(self.table, 1)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.cols = QTableWidget(0, 5)
        self.cols.setHorizontalHeaderLabels(["列名", "说明（可编辑）", "类型", "缺失%", "唯一值"])
        self.cols.setAlternatingRowColors(True)
        self.cols.verticalHeader().setVisible(False)
        hh = self.cols.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.cols.itemChanged.connect(self._desc_edited)
        rl.addWidget(QLabel("列设置：类型会影响预处理、统计检验与建模方式（如 cp 这类编码应设为“分类”）"))
        rl.addWidget(self.cols, 1)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("目标列（预测/分组对象）："))
        self.target = QComboBox()
        self.target.setMinimumWidth(160)
        self.target.currentIndexChanged.connect(self._target_changed)
        trow.addWidget(self.target)
        trow.addStretch(1)
        self.btn_check = QPushButton("数据质量检查")
        self.btn_check.clicked.connect(self._quality)
        trow.addWidget(self.btn_check)
        rl.addLayout(trow)
        self.report_box = QTextBrowser()
        self.report_box.setMaximumHeight(120)
        self.report_box.setVisible(False)
        rl.addWidget(self.report_box)
        split.addWidget(right)
        split.setSizes([620, 560])
        lay.addWidget(split, 1)

        self.btn_open.clicked.connect(lambda: self.open_file())
        self.btn_sample.clicked.connect(self.load_sample)
        self.btn_open_proj.clicked.connect(self.open_project)
        self.btn_save_proj.clicked.connect(self.save_project)
        self.sheet.activated.connect(lambda *_: self._path and self.open_file(self._path, self.sheet.currentText()))
        state.dataset_changed.connect(self.refresh)
        state.data_updated.connect(self._refresh_table_only)

    # ---------- 载入 ----------
    def open_file(self, path: str | None = None, sheet: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "打开数据文件", "", mio.FILE_FILTER)
            if not path:
                return
        try:
            sheets = mio.list_sheets(path)
            if sheets and sheet is None:
                sheet = sheets[0]
            df, info = mio.load_table(path, sheet=sheet)
        except Exception as e:
            QMessageBox.critical(self, "无法读取文件", f"{Path(path).name}\n\n{type(e).__name__}: {e}\n\n"
                                 "如果是 .xls 请确认已安装 xlrd；如果是特殊格式，可先另存为 CSV。")
            return
        if df.empty:
            QMessageBox.warning(self, "数据为空", "文件中没有读取到数据行。")
            return
        self._path = path
        name = Path(path).stem + (f"[{sheet}]" if sheet and len(sheets) > 1 else "")
        self.sheet.blockSignals(True)
        self.sheet.clear()
        self.sheet.addItems(sheets)
        if sheet:
            self.sheet.setCurrentText(sheet)
        self.sheet.blockSignals(False)
        self.sheet.setVisible(len(sheets) > 1)
        self.lbl_sheet.setVisible(len(sheets) > 1)
        self.state.set_dataset(Dataset.from_frame(name, df, info))
        notes = info.get("notes", [])
        if notes:
            self.toast("；".join(notes))

    def load_sample(self):
        ds = Dataset.from_frame(SAMPLE_NAME, make_heart_sample(), {"sample": True})
        self._path = None
        self.sheet.setVisible(False)
        self.lbl_sheet.setVisible(False)
        self.state.set_dataset(ds)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", "MedMiner 项目 (*.medproj)")
        if not path:
            return
        try:
            self.state.set_dataset(load_project(path))
        except Exception as e:
            QMessageBox.critical(self, "无法打开项目", f"{type(e).__name__}: {e}")

    def save_project(self):
        if not self.need_data():
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存项目", f"{self.state.ds.name}.medproj", "MedMiner 项目 (*.medproj)")
        if path:
            save_project(self.state.ds, path)
            self.toast(f"项目已保存：{path}")

    # ---------- 刷新 ----------
    def refresh(self):
        ds = self.state.ds
        self._refresh_table_only()
        self._fill_cols()
        self.report_box.setVisible(False)
        if ds is None:
            return
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItem("（不指定）", None)
        for c in ds.raw.columns:
            self.target.addItem(c, c)
        i = self.target.findData(ds.target)
        self.target.setCurrentIndex(max(0, i))
        self.target.blockSignals(False)

    def _refresh_table_only(self):
        ds = self.state.ds
        if ds is None:
            return
        self.table.set_frame(ds.raw.head(2000))
        r, c = ds.raw.shape
        miss = int(ds.raw.isna().sum().sum())
        extra = f"（预览前 2000 行）" if r > 2000 else ""
        self.info.setText(f"数据集：{ds.name}　{r} 行 × {c} 列　缺失单元格 {miss}（{miss / max(1, r * c) * 100:.2f}%）"
                          f"　重复行 {int(ds.raw.duplicated().sum())}　目标列：{ds.target or '未指定'} {extra}")

    def _fill_cols(self):
        ds = self.state.ds
        self._updating = True
        self.cols.setRowCount(0)
        if ds is not None:
            self.cols.setRowCount(ds.raw.shape[1])
            for i, c in enumerate(ds.raw.columns):
                it = QTableWidgetItem(c)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.cols.setItem(i, 0, it)
                d = QTableWidgetItem(ds.descriptions.get(c, ""))
                d.setData(Qt.UserRole, c)
                self.cols.setItem(i, 1, d)
                cb = QComboBox()
                for t in TYPE_ORDER:
                    cb.addItem(TYPE_LABELS[t], t)
                cb.setCurrentIndex(TYPE_ORDER.index(ds.types.get(c, "numeric")))
                cb.currentIndexChanged.connect(lambda _, col=c, w=cb: self._type_changed(col, w))
                self.cols.setCellWidget(i, 2, cb)
                s = ds.raw[c]
                for j, v in ((3, f"{s.isna().mean() * 100:.1f}"), (4, str(s.nunique()))):
                    t = QTableWidgetItem(v)
                    t.setFlags(t.flags() & ~Qt.ItemIsEditable)
                    t.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.cols.setItem(i, j, t)
        self._updating = False

    # ---------- 编辑 ----------
    def _type_changed(self, col, widget):
        if self._updating or self.state.ds is None:
            return
        self.state.ds.set_type(col, widget.currentData())
        self.state.touch()

    def _desc_edited(self, item):
        if self._updating or item.column() != 1 or self.state.ds is None:
            return
        col = item.data(Qt.UserRole)
        text = item.text().strip()
        if text:
            self.state.ds.descriptions[col] = text
        else:
            self.state.ds.descriptions.pop(col, None)

    def _target_changed(self):
        if self._updating or self.state.ds is None:
            return
        self.state.ds.set_target(self.target.currentData())
        self.state.touch()

    def _quality(self):
        if not self.need_data():
            return
        ds = self.state.ds
        df = ds.raw
        lines = []
        miss = df.isna().mean()
        hi = miss[miss > 0.3]
        lines.append(f"<b>缺失</b>：{int(df.isna().any().sum())} 列有缺失" + (f"；缺失率>30% 的列：{', '.join(hi.index)}" if len(hi) else ""))
        lines.append(f"<b>重复行</b>：{int(df.duplicated().sum())}")
        const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        if const:
            lines.append(f"<b>常量列</b>（建议忽略）：{', '.join(const)}")
        if ds.target and ds.target in df.columns:
            vc = df[ds.target].value_counts(normalize=True)
            if len(vc) <= 10:
                lines.append("<b>目标分布</b>：" + "，".join(f"{k}: {v * 100:.1f}%" for k, v in vc.items())
                             + ("　⚠ 类别不平衡" if vc.min() < 0.2 and len(vc) == 2 else ""))
        rng = check_clinical_ranges(df)
        bad = rng[rng["超范围个数"] > 0] if not rng.empty else rng
        if len(bad):
            lines.append("<b>超出临床合理范围</b>：" + "；".join(f"{r['变量']} {r['超范围个数']} 个（合理 {r['合理范围']}）" for _, r in bad.iterrows()))
        elif not rng.empty:
            lines.append("<b>临床范围</b>：所检查的变量均在合理范围内 ✓")
        lines.append(f"<b>样本量</b>：n={len(df)}" + ("　⚠ 样本较小，建议使用交叉验证并关注置信区间" if len(df) < 500 else ""))
        self.report_box.setHtml("<br>".join(lines))
        self.report_box.setVisible(True)
