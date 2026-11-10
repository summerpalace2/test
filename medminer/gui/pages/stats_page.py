"""④ 统计分析：描述统计、基线特征表(Table 1)、相关性、逻辑回归 OR。"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton, QSplitter,
                               QTabWidget, QVBoxLayout, QWidget)

from ...core import plots as pl
from ...core import stats
from ...core.medical import value_labels
from ...core.schema import BINARY, CATEGORICAL, NUMERIC
from ..widgets.forms import ColumnPickerRow
from ..widgets.plotwidget import PlotWidget
from ..widgets.tables import DataTable
from .base import Page, group, h1, hint


class StatsPage(Page):
    key, title = "stats", "统计分析"

    def __init__(self, state):
        super().__init__(state)
        lay = self.root_layout()
        lay.addWidget(h1("统计分析"))
        lay.addWidget(hint("医学论文常用的统计：描述统计、分组基线特征表（自动选择 t 检验/Mann-Whitney/卡方/Fisher 等）、相关性、逻辑回归比值比（OR）。"))
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        self._build_describe()
        self._build_table1()
        self._build_corr()
        self._build_or()
        state.dataset_changed.connect(self.refresh)
        state.data_updated.connect(self.refresh)

    # ---------- 描述统计 ----------
    def _build_describe(self):
        w = QWidget()
        l = QVBoxLayout(w)
        bar = QHBoxLayout()
        b1 = QPushButton("计算描述统计")
        b1.setObjectName("primary")
        b2 = QPushButton("正态性检验 (Shapiro-Wilk)")
        b3 = QPushButton("加入报告")
        for b in (b1, b2, b3):
            bar.addWidget(b)
        bar.addStretch(1)
        l.addLayout(bar)
        self.t_desc = DataTable(highlight_nan=False)
        self.t_norm = DataTable(highlight_nan=False)
        self.t_norm.setMaximumHeight(200)
        l.addWidget(self.t_desc, 3)
        l.addWidget(QLabel("正态性检验（p>0.05 近似正态，决定基线表使用均值±SD 还是中位数[IQR]）"))
        l.addWidget(self.t_norm, 1)
        b1.clicked.connect(self._describe)
        b2.clicked.connect(self._normality)
        b3.clicked.connect(lambda: self._to_report(self.t_desc, "描述统计"))
        self.tabs.addTab(w, "描述统计")

    def _describe(self):
        if not self.need_data():
            return
        ds = self.state.ds
        self.t_desc.set_frame(stats.describe(ds.current(), ds.current_types()))

    def _normality(self):
        if not self.need_data():
            return
        ds = self.state.ds
        t = ds.current_types()
        cols = [c for c in ds.current().columns if t.get(c) == NUMERIC]
        self.t_norm.set_frame(stats.normality(ds.current(), cols))

    # ---------- 基线表 ----------
    def _build_table1(self):
        w = QWidget()
        l = QHBoxLayout(w)
        ctl = QWidget()
        ctl.setFixedWidth(260)
        cl = QVBoxLayout(ctl)
        cl.setContentsMargins(0, 0, 0, 0)
        self.t1_group = QComboBox()
        cl.addWidget(QLabel("分组变量（如是否患病）"))
        cl.addWidget(self.t1_group)
        cl.addWidget(QLabel("纳入的变量"))
        self.t1_vars = ColumnPickerRow(height=10000)
        cl.addWidget(self.t1_vars, 1)
        self.t1_btn = QPushButton("生成基线特征表")
        self.t1_btn.setObjectName("primary")
        self.t1_btn.clicked.connect(self._table1)
        cl.addWidget(self.t1_btn)
        l.addWidget(ctl)
        right = QVBoxLayout()
        self.t1_note = hint("连续变量：近似正态 → 均值±SD + Welch t / ANOVA；否则 中位数[Q1,Q3] + Mann-Whitney / Kruskal-Wallis。"
                            "分类变量：n(%) + 卡方（小期望频数的 2×2 用 Fisher）。多变量比较未做多重校正，解读时请注意。")
        right.addWidget(self.t1_note)
        self.t1 = DataTable(highlight_nan=False)
        right.addWidget(self.t1, 1)
        row = QHBoxLayout()
        b1 = QPushButton("导出表格…")
        b2 = QPushButton("加入报告")
        b1.clicked.connect(self.t1.export_csv)
        b2.clicked.connect(lambda: self._to_report(self.t1, "基线特征表"))
        row.addStretch(1)
        row.addWidget(b1)
        row.addWidget(b2)
        right.addLayout(row)
        l.addLayout(right, 1)
        self.tabs.addTab(w, "基线特征表 (Table 1)")

    def _table1(self):
        if not self.need_data():
            return
        ds = self.state.ds
        g = self.t1_group.currentData()
        if not g:
            self.error("请选择分组变量", "基线表需要一个分类/二分类的分组变量。")
            return
        df, types = ds.current(), ds.current_types()
        if df[g].nunique() > 8:
            self.error("分组过多", f"{g} 有 {df[g].nunique()} 个取值，不适合作为分组变量。")
            return
        labels = value_labels(ds.raw, g)
        cols = [c for c in self.t1_vars.checked() if c != g]
        if not cols:
            self.error("请选择变量", "至少选择一个要比较的变量。")
            return
        try:
            tab = stats.table_one(df, g, types, cols, labels or None)
        except Exception as e:
            self.error("生成失败", f"{type(e).__name__}: {e}")
            return
        self.t1.set_frame(tab, stretch=False)
        self.state.cache["table1"] = tab
        self.state.cache["table1_group"] = g
        self.toast("基线特征表已生成")

    # ---------- 相关性 ----------
    def _build_corr(self):
        w = QWidget()
        l = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.corr_method = QComboBox()
        self.corr_method.addItems(["pearson", "spearman", "kendall"])
        b = QPushButton("计算")
        b.setObjectName("primary")
        b.clicked.connect(self._corr)
        bar.addWidget(QLabel("相关方法："))
        bar.addWidget(self.corr_method)
        bar.addWidget(b)
        bar.addStretch(1)
        l.addLayout(bar)
        sp = QSplitter(Qt.Horizontal)
        self.corr_plot = PlotWidget(figsize=(6, 5))
        self.corr_plot.add_to_report.connect(lambda p: self.state.report_add_figure(p, "相关性热力图", "统计分析"))
        sp.addWidget(self.corr_plot)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("各变量与目标列的关联强度（点二列相关 / Cramér V）"))
        self.t_assoc = DataTable(highlight_nan=False)
        rl.addWidget(self.t_assoc, 1)
        sp.addWidget(right)
        sp.setSizes([650, 450])
        l.addWidget(sp, 1)
        self.tabs.addTab(w, "相关性")

    def _corr(self):
        if not self.need_data():
            return
        ds = self.state.ds
        df, t = ds.current(), ds.current_types()
        self.corr_plot.draw_with(pl.draw_chart, "corr", df, t, cols=[c for c in df.columns if t.get(c) in (NUMERIC, BINARY)],
                              corr_method=self.corr_method.currentText())
        tgt = ds.target
        if tgt and tgt in df.columns and t.get(tgt) in (BINARY, CATEGORICAL) and df[tgt].nunique() == 2:
            self.t_assoc.set_frame(stats.target_association(df, tgt, t).round(4))
        else:
            self.t_assoc.set_frame(pd.DataFrame({"提示": ["目标列为二分类时才显示关联强度（可在预处理中把诊断二值化）"]}))

    # ---------- OR ----------
    def _build_or(self):
        w = QWidget()
        l = QHBoxLayout(w)
        ctl = QWidget()
        ctl.setFixedWidth(260)
        cl = QVBoxLayout(ctl)
        cl.setContentsMargins(0, 0, 0, 0)
        self.or_target = QComboBox()
        cl.addWidget(QLabel("结局变量（必须二分类）"))
        cl.addWidget(self.or_target)
        cl.addWidget(QLabel("自变量"))
        self.or_vars = ColumnPickerRow(height=10000)
        cl.addWidget(self.or_vars, 1)
        self.or_uni = QRadioButton("单因素（逐个变量）")
        self.or_multi = QRadioButton("多因素（同时放入模型）")
        self.or_uni.setChecked(True)
        cl.addWidget(self.or_uni)
        cl.addWidget(self.or_multi)
        b = QPushButton("运行逻辑回归")
        b.setObjectName("primary")
        b.clicked.connect(self._or)
        cl.addWidget(b)
        l.addWidget(ctl)
        sp = QSplitter(Qt.Vertical)
        self.t_or = DataTable(highlight_nan=False)
        sp.addWidget(self.t_or)
        self.or_plot = PlotWidget(figsize=(6, 4))
        self.or_plot.add_to_report.connect(lambda p: self.state.report_add_figure(p, "OR 森林图", "统计分析"))
        sp.addWidget(self.or_plot)
        sp.setSizes([260, 380])
        l.addWidget(sp, 1)
        self.tabs.addTab(w, "逻辑回归 (OR)")

    def _or(self):
        if not self.need_data():
            return
        ds = self.state.ds
        df, t = ds.current(), ds.current_types()
        y = self.or_target.currentData()
        feats = [c for c in self.or_vars.checked() if c != y]
        if not y or not feats:
            self.error("参数不完整", "请选择结局变量和至少一个自变量。")
            return
        if df[y].nunique() != 2:
            self.error("结局必须是二分类", f"“{y}”有 {df[y].nunique()} 个取值。请先到“数据预处理”用“二值化”把诊断变成 0/1。")
            return
        try:
            res = stats.logit_or(df, y, feats, t, multivariable=self.or_multi.isChecked())
        except Exception as e:
            self.error("拟合失败", f"{type(e).__name__}: {e}")
            return
        if res.empty:
            self.error("没有结果", "模型未能拟合（可能存在完全分离或样本过少）。")
            return
        self.t_or.set_frame(res.round(4))
        self.or_plot.draw_with(pl.plot_forest, res, f"OR 森林图（{'多因素' if self.or_multi.isChecked() else '单因素'}）")
        self.state.cache["or"] = res

    # ---------- 公共 ----------
    def _to_report(self, table: DataTable, title: str):
        df = table.frame()
        if df.empty:
            self.error("没有内容", "请先生成表格。")
            return
        self.state.report_add_table(df, title, "统计分析")

    def refresh(self):
        ds = self.state.ds
        if ds is None:
            return
        df, t = ds.current(), ds.current_types()
        cats = [c for c in df.columns if t.get(c) in (BINARY, CATEGORICAL) and df[c].nunique() <= 8]
        usable = [c for c in df.columns if t.get(c) in (NUMERIC, BINARY, CATEGORICAL)]
        for combo in (self.t1_group, self.or_target):
            prev = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            pool = cats if combo is self.t1_group else [c for c in cats if df[c].nunique() == 2]
            for c in pool:
                combo.addItem(ds.label(c), c)
            want = prev if prev in pool else (ds.target if ds.target in pool else (pool[0] if pool else None))
            if want is not None:
                combo.setCurrentIndex(combo.findData(want))
            combo.blockSignals(False)
        g = self.t1_group.currentData()
        self.t1_vars.set_columns(usable, [c for c in usable if c != g and c != ds.target], {c: ds.label(c) for c in usable})
        y = self.or_target.currentData()
        self.or_vars.set_columns(usable, [c for c in usable if c != y and c != ds.target and c != g][:12], {c: ds.label(c) for c in usable})
