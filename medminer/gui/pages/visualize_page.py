"""③ 数据可视化。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QSplitter,
                               QVBoxLayout, QWidget)

from ...core import plots as pl
from ...core.schema import BINARY, CATEGORICAL, NUMERIC
from ..widgets.forms import ColumnsButton
from ..widgets.plotwidget import PlotWidget
from .base import Page, group, h1, hint, scroll

ROLE_TEXT = {"num": "数值", "cat": "分类", "any": "任意"}


class VisualizePage(Page):
    key, title = "visualize", "数据可视化"

    def __init__(self, state):
        super().__init__(state)
        lay = self.root_layout()
        lay.addWidget(h1("数据可视化"))
        lay.addWidget(hint("选择图表类型与变量后点击“绘制”。图表可缩放/平移，用工具栏保存图片，或“加入报告”。分类变量用“按谁着色”区分组别。"))
        split = QSplitter(Qt.Horizontal)

        ctl = QWidget()
        form = QFormLayout(ctl)
        form.setLabelAlignment(Qt.AlignLeft)
        self.kind = QComboBox()
        for k, (label, _) in pl.CHART_KINDS.items():
            self.kind.addItem(label, k)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.x_box, self.y_box, self.hue = QComboBox(), QComboBox(), QComboBox()
        self.lx, self.ly, self.lh = QLabel("X"), QLabel("Y"), QLabel("按谁着色")
        self.bins = QSpinBox()
        self.bins.setRange(5, 100)
        self.bins.setValue(20)
        self.method = QComboBox()
        self.method.addItems(["pearson", "spearman", "kendall"])
        self.cols = ColumnsButton()
        self.annot = QCheckBox("显示数值")
        self.annot.setChecked(True)
        self.lc = QLabel("变量集合")
        self.lb, self.lm = QLabel("分箱数"), QLabel("相关方法")
        form.addRow("图表类型", self.kind)
        form.addRow(self.lx, self.x_box)
        form.addRow(self.ly, self.y_box)
        form.addRow(self.lh, self.hue)
        form.addRow(self.lb, self.bins)
        form.addRow(self.lm, self.method)
        form.addRow(self.lc, self.cols)
        form.addRow("", self.annot)
        self.btn = QPushButton("绘制")
        self.btn.setObjectName("primary")
        self.btn.clicked.connect(self.draw)
        form.addRow(self.btn)
        self.msg = hint("")
        form.addRow(self.msg)
        split.addWidget(scroll(ctl, 300))

        self.plot = PlotWidget(figsize=(7, 5))
        self.plot.add_to_report.connect(lambda png: self.state.report_add_figure(png, self._title(), "可视化"))
        split.addWidget(self.plot)
        split.setSizes([300, 900])
        lay.addWidget(split, 1)

        state.dataset_changed.connect(self.refresh)
        state.data_updated.connect(self.refresh)
        self._kind_changed()

    def _title(self):
        return f"{self.kind.currentText()}：{self.x_box.currentData() or ''} {self.y_box.currentData() or ''}".strip()

    # ---------- 变量候选 ----------
    def _candidates(self, role: str):
        ds = self.state.ds
        if ds is None:
            return []
        types = ds.current_types()
        cols = [c for c in ds.current().columns if types.get(c) not in ("ignore", "text", "datetime")]
        r = role.rstrip("?")
        if r == "num":
            return [c for c in cols if types.get(c) == NUMERIC]
        if r == "cat":
            return [c for c in cols if types.get(c) in (CATEGORICAL, BINARY)]
        return cols

    def _fill(self, combo: QComboBox, role: str | None, default: str | None = None, keep: bool = True):
        combo.blockSignals(True)
        prev = combo.currentData() if keep else None
        combo.clear()
        if role is None:
            combo.blockSignals(False)
            return
        if role.endswith("?"):
            combo.addItem("（无）", None)
        ds = self.state.ds
        for c in self._candidates(role):
            combo.addItem(ds.label(c), c)
        for want in (prev, default):
            i = combo.findData(want) if want else -1
            if i >= 0:
                combo.setCurrentIndex(i)
                break
        combo.blockSignals(False)

    def _kind_changed(self):
        k = self.kind.currentData()
        spec = pl.CHART_KINDS[k][1]
        for lab, w, key in ((self.lx, self.x_box, "x"), (self.ly, self.y_box, "y"), (self.lh, self.hue, "hue")):
            on = key in spec
            lab.setVisible(on)
            w.setVisible(on)
            if on:
                lab.setText({"x": "X", "y": "Y", "hue": "按谁着色"}[key] + f"（{ROLE_TEXT[spec[key].rstrip('?')]}）")
        for w in (self.bins, self.lb):
            w.setVisible(k in ("histogram",))
        for w in (self.method, self.lm, self.annot):
            w.setVisible(k == "corr")
        for w in (self.cols, self.lc):
            w.setVisible(k in ("corr", "pairs"))
        self._populate(keep=False)  # 换图表类型：重新给出合理默认值，而不是沿用上一张图的选择

    def _populate(self, keep: bool = True):
        ds = self.state.ds
        spec = pl.CHART_KINDS[self.kind.currentData()][1]
        target = ds.target if ds else None
        nums = self._candidates("num")
        cats = self._candidates("cat")
        first_num = next((c for c in nums if c != target), None)
        second_num = next((c for c in nums if c not in (target, first_num)), first_num)
        tgt_cat = target if target in cats else None
        x_role = spec.get("x", "")
        kind = self.kind.currentData()
        other_cat = next((c for c in cats if c != target), None)
        if x_role.startswith("cat"):
            # 计数/构成比图：X=某个分类变量、着色=目标列；箱线/小提琴/均值图：X=目标列（按疾病状态分组）
            flip = kind in ("count", "stacked")
            x_default = (other_cat or tgt_cat) if flip else (tgt_cat or other_cat)
        elif kind == "line":
            x_default = None  # 折线图默认按行序
        else:
            x_default = first_num
        # 数值 X 配数值 Y（散点/回归线）时 Y 取第二个数值变量；分类 X 配数值 Y（箱线/小提琴）时 Y 取第一个
        y_default = second_num if x_role.startswith("num") else first_num
        self._fill(self.x_box, spec.get("x"), x_default, keep)
        self._fill(self.y_box, spec.get("y"), y_default, keep)
        self._fill(self.hue, spec.get("hue") if "hue" in spec else None, None, keep)
        if "hue" in spec:
            used = {self.x_box.currentData(), self.y_box.currentData()} - {None}
            required = not spec["hue"].endswith("?")
            cur = self.hue.currentData()
            if (not keep or cur is None) and tgt_cat and tgt_cat not in used and self.hue.findData(tgt_cat) >= 0:
                self.hue.setCurrentIndex(self.hue.findData(tgt_cat))  # 着色默认用目标列
            elif cur in used or (required and cur is None):
                # 着色不能与 X/Y 相同（退化图）；必选时选第一个不冲突的变量，可选时置为“无”
                for i in range(self.hue.count()):
                    v = self.hue.itemData(i)
                    if v is not None and v not in used:
                        self.hue.setCurrentIndex(i)
                        break
                else:
                    self.hue.setCurrentIndex(0)
        cols = [c for c in (ds.current().columns if ds else []) if c in nums]
        self.cols.set_columns(cols, cols[:6] if self.kind.currentData() == "pairs" else cols)

    def refresh(self):
        self._populate()
        if self.state.ds is None:
            self.plot.message("请先载入数据")

    # ---------- 绘制 ----------
    def draw(self):
        if not self.need_data():
            return
        ds = self.state.ds
        k = self.kind.currentData()
        spec = pl.CHART_KINDS[k][1]
        x, y, hue = self.x_box.currentData(), self.y_box.currentData(), self.hue.currentData()
        if spec.get("x") and not spec["x"].endswith("?") and not x:
            self.msg.setText("没有可用于 X 的变量（类型不匹配）。请到“数据管理”页检查列类型。")
            return
        if spec.get("y") and not spec["y"].endswith("?") and not y:
            self.msg.setText("没有可用于 Y 的变量（类型不匹配）。")
            return
        self.msg.setText("")
        self.plot.draw_with(pl.draw_chart, k, ds.current(), ds.current_types(), x=x, y=y, hue=hue,
                         cols=self.cols.selected(), bins=self.bins.value(), corr_method=self.method.currentText(),
                         annotate=self.annot.isChecked())
